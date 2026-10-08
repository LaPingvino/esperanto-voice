#!/usr/bin/env python3
"""Score how strongly each clip's r's are rolled (trilled).

A rolled (alveolar) r is 2-4 tongue-tip taps at roughly 25-35 per second, so
the loudness inside the r pulses at that rate. Uvular ("French") and
approximant ("English") r's lack that clean pulse. Speakers also compensate a
weak r in the neighbouring vowels — fine for human listeners, but a voice
trained on such clips learns the compensation instead of the r. Choosing the
clips where the narrator really rolls makes a computer voice clearer.

    trill_score.py data/oz [--limit 200] > oz-trill.tsv

For each clip: locate every 'r' with CTC alignment on an Esperanto character
ASR, measure the 20-40 Hz modulation of the amplitude envelope inside each r
(relative to all modulation energy 2-60 Hz), and report per clip:

    clip  n_r  trill_mean  trill_max

Higher = more trill-like. Compare clips of one speaker with each other, and
speakers by their median; absolute values depend on recording conditions.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def trill_strength(seg, sr):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt, hilbert
    if len(seg) < int(0.04 * sr):
        return None
    sos = butter(4, [300, 4000], btype="band", fs=sr, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, seg)))
    env = env - env.mean()
    # pad to ~0.5 s so low modulation frequencies are resolvable
    n = max(len(env), int(0.5 * sr))
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)), n=n)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    band = spec[(f >= 20) & (f <= 40)].sum()
    total = spec[(f >= 2) & (f <= 60)].sum() + 1e-12
    return float(band / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pad", type=float, default=0.02, help="seconds added around each r")
    args = ap.parse_args()

    import numpy as np
    import librosa
    import torch
    from ctc_segmentation import (CtcSegmentationParameters, ctc_segmentation,
                                  prepare_text)
    import evaltest as E

    asr = E.ASR()
    vocab = asr.proc.tokenizer.get_vocab()
    char_list = [None] * len(vocab)
    for ch, i in vocab.items():
        char_list[i] = ch
    blank = vocab.get(asr.proc.tokenizer.pad_token, 0)
    sep = asr.proc.tokenizer.word_delimiter_token or "|"

    rows = [l.rstrip("\n").split("|", 1) for l in open(os.path.join(args.data, "metadata.csv"), encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print("clip\tn_r\ttrill_mean\ttrill_max", flush=True)
    for cid, text in rows:
        path = os.path.join(args.data, "wav", cid + ".wav")
        y, sr = librosa.load(path, sr=16000)
        norm = E.norm(text)
        if "r" not in norm:
            continue
        with torch.no_grad():
            inp = asr.proc(y, sampling_rate=16000, return_tensors="pt").input_values
            lp = torch.log_softmax(asr.model(inp).logits[0], dim=-1).numpy()
        params = CtcSegmentationParameters()
        params.char_list = char_list
        params.blank = blank
        params.index_duration = len(y) / lp.shape[0] / 16000
        chars = norm.replace(" ", sep)
        try:
            gt, _ = prepare_text(params, [chars])
            timings, _, _ = ctc_segmentation(params, lp, gt)
        except Exception as e:
            print(f"# {cid}: alignment failed ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        # timings[k] = start time of the k-th character of the ground truth
        # (prepare_text prepends one blank, so character k sits at index k+1).
        scores = []
        for k, ch in enumerate(chars):
            if ch != "r" or k + 2 >= len(timings):
                continue
            t0 = max(0.0, timings[k + 1] - args.pad)
            t1 = timings[k + 2] + args.pad
            s = trill_strength(y[int(t0 * sr): int(t1 * sr)], sr)
            if s is not None:
                scores.append(s)
        if scores:
            print(f"{cid}\t{len(scores)}\t{np.mean(scores):.3f}\t{np.max(scores):.3f}", flush=True)


if __name__ == "__main__":
    main()
