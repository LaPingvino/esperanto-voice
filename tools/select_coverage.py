#!/usr/bin/env python3
"""Pick the most informative N minutes of clips, using only the transcripts.

Greedy set cover over phoneme bigrams (as Piper's espeak produces them),
weighted so rare units count most: each step takes the clip that adds the
most not-yet-seen weight per second of audio. With very little data, this
beats "the first N minutes", which over-samples whatever the first chapter
happens to talk about.

    select_coverage.py data/karlo --minutes 5 > data/karlo/cover5.csv

Writes Piper CSV lines (<id>.wav|<text>) to stdout and a coverage summary to
stderr. experiment.py takes the result with --csv.
"""

import argparse
import collections
import math
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--max-sec", type=float, default=12,
                    help="ignore longer clips (match experiment.py --max-clip-sec)")
    args = ap.parse_args()

    import soundfile as sf
    from piper.phonemize_espeak import EspeakPhonemizer
    ph = EspeakPhonemizer()

    clips = []
    for line in open(os.path.join(args.data, "metadata.csv"), encoding="utf-8"):
        cid, text = line.rstrip("\n").split("|", 1)
        dur = sf.info(os.path.join(args.data, "wav", cid + ".wav")).duration
        if dur > args.max_sec:
            continue
        seq = [p for sent in ph.phonemize("eo", text) for p in sent]
        units = set(seq) | {a + b for a, b in zip(seq, seq[1:])}
        clips.append((cid, text, dur, units))

    freq = collections.Counter(u for c in clips for u in c[3])
    weight = {u: 1.0 / math.sqrt(n) for u, n in freq.items()}   # rare = valuable

    budget, used, seen, chosen = args.minutes * 60, 0.0, set(), []
    remaining = list(clips)
    while remaining:
        best = max(remaining, key=lambda c: sum(weight[u] for u in c[3] - seen) / c[2])
        if used + best[2] > budget:
            remaining.remove(best)
            continue
        chosen.append(best)
        used += best[2]
        seen |= best[3]
        remaining.remove(best)

    for cid, text, _, _ in chosen:
        print(f"{cid}.wav|{text}")
    single = {u for u in freq if len(u) == 1}
    print(f"{len(chosen)} clips, {used/60:.1f} min; phonemes {len(seen & single)}/{len(single)}, "
          f"units {len(seen)}/{len(freq)}", file=sys.stderr)


if __name__ == "__main__":
    main()
