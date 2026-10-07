#!/usr/bin/env python3
"""Turn one Common Voice speaker's clips into a Piper training dataset.

    cv_prepare.py dataset/cv27-full/speakers/<id> tts-work/data/cv-<name> --max-clips 2000

Reads <speaker>/metadata.csv + <speaker>/mp3/*.mp3 (from `cv_stream.py clips`)
and writes <out>/wav/<clip>.wav (mono, 22.05 kHz, 16-bit, leading/trailing
silence trimmed) and <out>/metadata.csv (<clip>|<sentence>), the same layout
`prepare_librivox.py` produces, so experiment.py and select_coverage.py work
on it unchanged.
"""

import argparse
import os

SR = 22050


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker_dir")
    ap.add_argument("out")
    ap.add_argument("--max-clips", type=int, default=2000)
    ap.add_argument("--exclude", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "..", "eval", "sentences.txt"),
                    help="sentences never to include (default: the test set)")
    ap.add_argument("--top-db", type=float, default=35,
                    help="silence trim threshold; Common Voice clips have long silent edges")
    args = ap.parse_args()

    import librosa
    import numpy as np
    import soundfile as sf

    import re
    import unicodedata

    def key(s):
        s = unicodedata.normalize("NFC", s.lower())
        return re.sub(r"\s+", " ", re.sub(r"[^a-zĉĝĥĵŝŭ ]", " ", s)).strip()

    # Never train on the test set: some test sentences also exist in Common
    # Voice ("Esperanto estas internacia lingvo." was recorded once).
    held_out = {key(l) for l in open(args.exclude, encoding="utf-8") if l.strip()}

    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    rows = [r for r in (l.rstrip("\n").split("|", 1)
                        for l in open(os.path.join(args.speaker_dir, "metadata.csv"), encoding="utf-8")
                        if "|" in l)
            if key(r[1]) not in held_out][: args.max_clips]
    n = total = 0
    with open(os.path.join(args.out, "metadata.csv"), "w", encoding="utf-8") as meta:
        for cid, text in rows:
            try:
                y, _ = librosa.load(os.path.join(args.speaker_dir, "mp3", cid + ".mp3"),
                                    sr=SR, mono=True)
            except Exception:
                continue
            y, _ = librosa.effects.trim(y, top_db=args.top_db)
            if len(y) < SR * 0.8:
                continue
            peak = float(np.max(np.abs(y))) or 1.0
            sf.write(os.path.join(args.out, "wav", cid + ".wav"), (y * (0.9 / peak)).astype("float32"),
                     SR, subtype="PCM_16")
            meta.write(f"{cid}|{text.strip()}\n")
            n += 1
            total += len(y) / SR
    print(f"{n} clips, {total/60:.1f} min → {args.out}")


if __name__ == "__main__":
    main()
