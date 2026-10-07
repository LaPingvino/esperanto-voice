#!/usr/bin/env python3
"""Choose voice candidates from Common Voice speaker profiles.

    pick_speakers.py dataset/cv27-meta/speakers.tsv dataset/profiles.tsv \
        --male 3 --female 2 > picked.txt

Joins the per-speaker metadata (hours, gender) with profile_speakers.py's
measurements, drops speakers who fail any hard filter, ranks the rest by
phoneme error rate (correctness first), and prints the chosen full
client_ids — one per line, followed by a tab and a short description — so
`cv_stream.py clips --speakers picked.txt` can use the file directly. A
readable shortlist of everyone who passed goes to stderr.
"""

import argparse
import csv
import sys


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speakers")
    ap.add_argument("profiles")
    ap.add_argument("--male", type=int, default=3)
    ap.add_argument("--female", type=int, default=2)
    ap.add_argument("--min-hours", type=float, default=5)
    ap.add_argument("--max-per", type=float, default=0.12)
    ap.add_argument("--max-nasal", type=float, default=5)
    ap.add_argument("--min-trill", type=float, default=0.7)
    ap.add_argument("--max-breath", type=float, default=0.25)
    ap.add_argument("--min-snr", type=float, default=30)
    ap.add_argument("--min-rolloff", type=float, default=3.5)
    args = ap.parse_args()

    meta = {r["client_id"][:16]: r for r in read(args.speakers)}
    passed = []
    for p in read(args.profiles):
        m = meta.get(p["speaker"])
        if not m:
            continue
        f = lambda k: float(p[k])
        ok = (float(m["hours"]) >= args.min_hours and f("per") <= args.max_per
              and f("nasal") <= args.max_nasal and f("r_trill") >= args.min_trill
              and f("breath") <= args.max_breath and f("snr_db") >= args.min_snr
              and f("rolloff") >= args.min_rolloff)
        if ok:
            passed.append((f("per"), m, p))
    passed.sort(key=lambda t: t[0])

    print(f"{'speaker':16} {'gender':16} {'hours':>6} {'per':>6} {'cer':>6} {'nasal':>5} "
          f"{'trill':>5} {'breath':>6} {'snr':>5} {'kHz':>4}", file=sys.stderr)
    for per, m, p in passed:
        print(f"{p['speaker']:16} {m['gender'] or '?':16} {float(m['hours']):6.1f} {per:6.3f} "
              f"{float(p['cer']):6.3f} {p['nasal']:>5} {p['r_trill']:>5} {p['breath']:>6} "
              f"{p['snr_db']:>5} {p['rolloff']:>4}", file=sys.stderr)

    want = {"male": args.male, "female": args.female}
    for per, m, p in passed:
        g = "female" if m["gender"].startswith("female") else "male" if m["gender"].startswith("male") else None
        if g and want[g] > 0:
            want[g] -= 1
            print(f"{m['client_id']}\t{g} {float(m['hours']):.1f}h per={per:.3f}")


if __name__ == "__main__":
    main()
