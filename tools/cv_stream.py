#!/usr/bin/env python3
"""Work with a Common Voice release archive without unpacking it.

A Common Voice language archive is tens of gigabytes of MP3s plus a few
small TSV files. Unpacking doubles the disk use for no benefit, so this tool
streams the .tar.gz in one sequential pass, twice at most:

    cv_stream.py meta ARCHIVE OUTDIR
        Extract only the *.tsv metadata (validated.tsv, clip_durations.tsv, …).

    cv_stream.py speakers OUTDIR
        Per-speaker summary from the metadata: validated clips, hours,
        up/down votes, gender/age/accent fields. Writes OUTDIR/speakers.tsv,
        sorted by validated hours.

    cv_stream.py clips ARCHIVE OUTDIR --speakers ids.txt [--min-up 2]
        Extract only the validated clips of the listed client_ids, plus a
        Piper-style OUTDIR/<speaker>/metadata.csv (clip|sentence).

The archive is treated purely as data: member names are reduced to their
basename and nothing from it is executed.
"""

import argparse
import collections
import csv
import os
import sys
import tarfile

csv.field_size_limit(10 ** 9)


def stream(archive):
    # "r|gz": strictly sequential reading, no seeking, constant memory.
    return tarfile.open(archive, mode="r|gz")


def cmd_meta(args):
    os.makedirs(args.outdir, exist_ok=True)
    n = 0
    with stream(args.archive) as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".tsv"):
                name = os.path.basename(m.name)
                with tar.extractfile(m) as src, open(os.path.join(args.outdir, name), "wb") as dst:
                    dst.write(src.read())
                n += 1
                print(f"  {name} ({m.size/1e6:.1f} MB)", file=sys.stderr, flush=True)
    print(f"{n} metadata files → {args.outdir}", file=sys.stderr)


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def cmd_speakers(args):
    dur = {}
    dpath = os.path.join(args.outdir, "clip_durations.tsv")
    if os.path.exists(dpath):
        for r in read_tsv(dpath):
            dur[r["clip"]] = float(r["duration[ms]"]) / 1000
    stats = collections.defaultdict(lambda: {"clips": 0, "sec": 0.0, "up": 0, "down": 0,
                                             "gender": collections.Counter(),
                                             "age": collections.Counter(),
                                             "accent": collections.Counter()})
    for r in read_tsv(os.path.join(args.outdir, "validated.tsv")):
        s = stats[r["client_id"]]
        s["clips"] += 1
        s["sec"] += dur.get(r["path"], 0.0)
        s["up"] += int(r.get("up_votes") or 0)
        s["down"] += int(r.get("down_votes") or 0)
        for k, col in (("gender", "gender"), ("age", "age"), ("accent", "accents")):
            v = (r.get(col) or "").strip()
            if v:
                s[k][v] += 1
    out = os.path.join(args.outdir, "speakers.tsv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("client_id\tclips\thours\tup\tdown\tgender\tage\taccent\n")
        for cid, s in sorted(stats.items(), key=lambda kv: -kv[1]["sec"]):
            top = lambda c: c.most_common(1)[0][0] if c else ""
            f.write(f"{cid}\t{s['clips']}\t{s['sec']/3600:.2f}\t{s['up']}\t{s['down']}\t"
                    f"{top(s['gender'])}\t{top(s['age'])}\t{top(s['accent'])}\n")
    total = sum(s["sec"] for s in stats.values()) / 3600
    print(f"{len(stats)} speakers, {total:.0f} validated hours → {out}", file=sys.stderr)


def cmd_clips(args):
    wanted = {l.strip() for l in open(args.speakers) if l.strip()}
    keep = {}
    for r in read_tsv(os.path.join(args.outdir, "validated.tsv")):
        if r["client_id"] in wanted and int(r.get("up_votes") or 0) >= args.min_up \
                and int(r.get("down_votes") or 0) == 0:
            keep[r["path"]] = (r["client_id"], r["sentence"])
    print(f"extracting {len(keep)} clips from {len(wanted)} speakers", file=sys.stderr)
    metas = {}
    n = 0
    with stream(args.archive) as tar:
        for m in tar:
            base = os.path.basename(m.name)
            if not m.isfile() or base not in keep:
                continue
            cid, sentence = keep[base]
            d = os.path.join(args.outdir, "speakers", cid[:16])
            os.makedirs(os.path.join(d, "mp3"), exist_ok=True)
            with tar.extractfile(m) as src, open(os.path.join(d, "mp3", base), "wb") as dst:
                dst.write(src.read())
            if cid not in metas:
                metas[cid] = open(os.path.join(d, "metadata.csv"), "w", encoding="utf-8")
            metas[cid].write(f"{os.path.splitext(base)[0]}|{sentence}\n")
            n += 1
            if n % 1000 == 0:
                print(f"  {n}/{len(keep)}", file=sys.stderr, flush=True)
    for f in metas.values():
        f.close()
    print(f"done: {n} clips", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("meta"); a.add_argument("archive"); a.add_argument("outdir")
    a = sub.add_parser("speakers"); a.add_argument("outdir")
    a = sub.add_parser("clips"); a.add_argument("archive"); a.add_argument("outdir")
    a.add_argument("--speakers", required=True)
    a.add_argument("--min-up", type=int, default=2)
    args = ap.parse_args()
    {"meta": cmd_meta, "speakers": cmd_speakers, "clips": cmd_clips}[args.cmd](args)


if __name__ == "__main__":
    main()
