#!/usr/bin/env python3
"""One iteration of the voice loop: subset → fine-tune → export → test.

    experiment.py --data data/karlo --minutes 5 --train-minutes 30 --name k5

1. Takes the first --minutes of clips from <data>/metadata.csv.
2. Fine-tunes Piper from the CC BY 4.0 `_base_model` checkpoint for at most
   --train-minutes of wall-clock time (CPU is fine for this).
3. Exports <runs>/<name>/eo.onnx (+ .onnx.json).
4. Runs evaltest.py on it and appends one row to <runs>/results.tsv:

    name  data_min  train_min  epochs  cer  passed

So every experiment, however small, lands as one comparable line.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Shapes and settings of rhasspy/piper-checkpoints/_base_model (LibriTTS-R,
# CC BY 4.0). Warm-starting copies only parameters whose shapes match, so the
# vocoder layout must be identical.
BASE_ARGS = [
    "--model.sample_rate", "22050",
    "--model.resblock", "2",
    "--model.resblock_kernel_sizes", "[3,5,7]",
    "--model.resblock_dilation_sizes", "[[1,2],[2,6],[3,12]]",
    "--model.upsample_rates", "[8,8,4]",
    "--model.upsample_initial_channel", "256",
    "--model.upsample_kernel_sizes", "[16,16,8]",
    "--model.use_mrd", "true",
    "--model.use_sdp", "true",
    "--model.mos_metric", "none",
    "--data.vowel_clusters", '[["a","ɪ"],["a","ʊ"],["e","ɪ"],["o","ʊ"],["ɔ","ɪ"]]',
]


def subset(data_dir, source_csv, minutes, max_sec, out_csv):
    """Clips from source_csv (ids with or without .wav), skipping any longer
    than max_sec, up to `minutes` of audio (0 = all)."""
    import soundfile as sf
    rows, total = [], 0.0
    with open(source_csv, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cid, text = line.rstrip("\n").split("|", 1)
            wav = cid if cid.endswith(".wav") else cid + ".wav"
            dur = sf.info(os.path.join(data_dir, "wav", wav)).duration
            if dur > max_sec:
                continue   # VITS memory grows with the longest clip in a batch
            if minutes and total + dur > minutes * 60:
                break
            rows.append((wav, text))
            total += dur
    with open(out_csv, "w", encoding="utf-8") as f:
        for wav, text in rows:
            f.write(f"{wav}|{text}\n")
    return len(rows), total / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--minutes", type=float, default=0,
                    help="take the first N minutes of clips (ignored with --csv)")
    ap.add_argument("--csv", help="use this clip list instead, e.g. from select_coverage.py")
    ap.add_argument("--train-minutes", type=int, default=30)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--base", default="base/base_model.ckpt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-clip-sec", type=float, default=12,
                    help="skip longer clips; long clips dominate training memory")
    ap.add_argument("--phase", choices=["full", "text"], default="full",
                    help="text: train only the language side (see piper_train.py); "
                         "much faster per step, voice timbre stays the base's")
    ap.add_argument("--fast", action="store_true",
                    help="full phase speedups: no MRD discriminator, half-length "
                         "training segments")
    ap.add_argument("--val-every", type=int, default=1,
                    help="validate (and checkpoint) every N epochs")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="further piper.train arguments, passed through")
    args = ap.parse_args()

    run = os.path.join(args.runs, args.name)
    os.makedirs(run, exist_ok=True)
    n, got_min = subset(args.data, args.csv or os.path.join(args.data, "metadata.csv"),
                        args.minutes, args.max_clip_sec, os.path.join(run, "metadata.csv"))
    print(f"[{args.name}] {n} clips, {got_min:.1f} min of audio", flush=True)

    py = sys.executable
    t0 = time.time()
    cmd = [py, os.path.join(HERE, "piper_train.py"), "fit",
           "--trainer.enable_progress_bar", "false",
           "--data.voice_name", f"eo-{args.name}",
           "--data.csv_path", os.path.join(run, "metadata.csv"),
           "--data.audio_dir", os.path.join(args.data, "wav"),
           "--data.espeak_voice", "eo",
           "--data.cache_dir", os.path.join(run, "cache"),
           "--data.config_path", os.path.join(run, "config.json"),
           "--data.batch_size", str(args.batch_size),
           "--data.num_test_examples", "0",
           # Worker processes buy nothing at this data size, and Python 3.14's
           # forkserver needs Unix sockets that sandboxes often forbid.
           "--data.num_workers", "0",
           "--model.warmstart_ckpt", args.base,
           "--trainer.accelerator", "cpu",
           "--trainer.max_time", f"00:00:{args.train_minutes:02d}:00"
           if args.train_minutes < 60 else
           f"00:{args.train_minutes // 60:02d}:{args.train_minutes % 60:02d}:00",
           "--trainer.default_root_dir", run,
           "--trainer.check_val_every_n_epoch", str(args.val_every),
           ] + BASE_ARGS
    if args.fast:
        # Later flags override BASE_ARGS. segment_size must be a multiple of
        # hop_length (256).
        cmd += ["--model.use_mrd", "false", "--model.segment_size", "4096"]
    cmd += args.extra
    env = dict(os.environ, PIPER_PHASE=args.phase)
    with open(os.path.join(run, "train.log"), "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    train_min = (time.time() - t0) / 60
    ckpts = sorted(glob.glob(os.path.join(run, "lightning_logs", "*", "checkpoints", "last.ckpt")),
                   key=os.path.getmtime)
    if not ckpts:
        sys.exit(f"[{args.name}] training produced no checkpoint (exit {rc}); see {run}/train.log")
    ckpt = ckpts[-1]
    epoch = re.search(r"epoch=(\d+)", " ".join(os.listdir(os.path.dirname(ckpt))))
    epochs = epoch.group(1) if epoch else "?"

    onnx = os.path.join(run, "eo.onnx")
    # torch >= 2.9 defaults torch.onnx.export to the dynamo exporter, which
    # can't trace VITS; the TorchScript exporter (dynamo=False) can.
    exporter = ("import sys, functools, torch; "
                "torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False); "
                "import runpy; sys.argv[0] = 'export_onnx'; "
                "runpy.run_module('piper.train.export_onnx', run_name='__main__')")
    exp = subprocess.run([py, "-c", exporter, "--checkpoint", ckpt, "--output-file", onnx],
                         capture_output=True, text=True)
    if exp.returncode != 0:
        sys.exit(f"[{args.name}] export failed:\n{exp.stderr[-2000:]}")
    shutil.copy(os.path.join(run, "config.json"), onnx + ".json")

    res = subprocess.run([py, os.path.join(HERE, "evaltest.py"), "--out",
                          os.path.join(run, "eval"), f"piper:{onnx}"],
                         capture_output=True, text=True)
    summary = [l for l in res.stdout.splitlines() if l.startswith(("ok", "FAIL"))]
    m = re.search(r"CER\s+([\d.]+)%\s+(\d+/\d+)", summary[-1] if summary else "")
    cer, passed = (m.group(1), m.group(2)) if m else ("?", "?")
    print(res.stdout, flush=True)

    new = not os.path.exists(os.path.join(args.runs, "results.tsv"))
    with open(os.path.join(args.runs, "results.tsv"), "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new:
            w.writerow(["name", "data_min", "train_min", "epochs", "cer", "passed"])
        w.writerow([args.name, f"{got_min:.1f}", f"{train_min:.0f}", epochs, cer, passed])
    print(f"[{args.name}] data {got_min:.1f} min, trained {train_min:.0f} min, "
          f"epochs {epochs}: CER {cer}% ({passed})", flush=True)


if __name__ == "__main__":
    main()
