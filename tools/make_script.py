#!/usr/bin/env python3
"""Design a recording script for a new Esperanto TTS voice.

    make_script.py cv27-meta/validated_sentences.tsv --n 1200 --out recording/

Picks N sentences from a CC0 sentence pool (Common Voice's sentence list) by
greedy coverage of phonemes and phoneme pairs, as Piper's espeak produces
them. Unlike select_coverage.py (which scores per second of existing audio and
so drifts toward short clips), this scores per *sentence* within a fixed
length band, because a reader's time goes to sentences, not seconds.

Extra weight goes to the sounds earlier Esperanto voices got wrong: the
trilled r, word-final n (accusative/plural), ĵ, ĥ, "sc", and j/ŭ before a
vowel. Sentences with digits, all-caps words, or letters outside the
Esperanto alphabet (q w x y, "th") are skipped, as are the evaluation
sentences, so the new voice can still be tested fairly.

Writes:
    <out>/script.tsv          id, session, sentence
    <out>/session-NN.txt      one numbered sentence per line, 100 per session
"""

import argparse
import csv
import math
import os
import random
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
csv.field_size_limit(10 ** 9)

FOREIGN = re.compile(r"[qwxyQWXYäöüéèàçñøåÄÖÜÉ]|th|Th", re.U)
BONUS = {"r": 3.0, "ʒ": 4.0, "x": 4.0, "sts": 3.0}   # ĵ, ĥ, sc


def key(s):
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-zĉĝĥĵŝŭ ]", " ", s)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences", help="TSV with a 'sentence' column (Common Voice sentence list)")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--per-session", type=int, default=100)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=16)
    ap.add_argument("--pool", type=int, default=60000, help="random candidates to phonemize")
    ap.add_argument("--seed", type=int, default=1887)
    ap.add_argument("--exclude", default=os.path.join(HERE, "..", "eval", "sentences.txt"))
    ap.add_argument("--avoid", nargs="*", default=[],
                    help="metadata.csv files of existing recordings whose sentences to skip")
    ap.add_argument("--unrecorded", action="store_true",
                    help="only Common Voice sentences nobody has recorded yet (clips_count == 0)")
    ap.add_argument("--extra-text", nargs="*", default=[],
                    help="plain-text files, one sentence per line, added to the pool (e.g. a PD novel)")
    ap.add_argument("--include-extra", action="store_true",
                    help="give every --extra-text sentence a guaranteed place, spread across sessions")
    ap.add_argument("--extra-boost", type=float, default=1.5,
                    help="score multiplier for --extra-text sentences (natural prose)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    held_out = {key(l) for l in open(args.exclude, encoding="utf-8") if l.strip()}
    for path in args.avoid:
        # Sentences already recorded elsewhere (other books' metadata.csv): a
        # new voice should add new material, not re-record existing material.
        for l in open(path, encoding="utf-8"):
            held_out.add(key(l.split("|", 1)[-1]))

    def eligible(s, names_ok=False):
        nw = len(s.split())
        if not (args.min_words <= nw <= args.max_words):
            return False
        if re.search(r"\d", s) or FOREIGN.search(s) or re.search(r"\b[A-ZĈĜĤĴŜŬ]{2,}\b", s):
            return False
        # Hyphenated coinages (hind-azadiraĥta) are rare-sound traps: awkward
        # to read, teach little. Mid-sentence proper names (Timnat-Ĥeres) are
        # too in an encyclopedic pool like Common Voice's, but in a novel they
        # are its characters (Ernesto, onklo Vik) and read naturally.
        if re.search(r"\w-\w", s):
            return False
        # Typos and transliterations: ŭ only follows a or e in Esperanto
        # ("hŭanglongbingo" is a loanword); ǔ (caron) is a mistyped ŭ.
        if "ǔ" in s or "Ǔ" in s or re.search(r"(?<![aeAE])[ŭŬ]", s):
            return False
        # Novels play with spelling in dialogue: lisps ("aĉetiŝ ŝole"), broken
        # English ("Pa’doŭnu men"), trailing-off fragments. Keep it readable:
        # no long ellipses, no apostrophes inside words except the elided l’,
        # at most one quotation.
        if ".." in s or "…" in s:
            return False
        if re.search(r"\w[’']\w|[’'](?!\s)(?<!l’)(?<!l')", s.replace("l’ ", "").replace("l' ", "")):
            return False
        if s.count("„") + s.count("“") + s.count('"') > 2:
            return False
        if not names_ok and re.search(r"(?<![.!?«\"—–]\s)(?<!^)\b[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+", s[1:]):
            return False
        return True

    pool, seen = [], set()
    sources = {}
    with open(args.sentences, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            s = (r.get("sentence") or "").strip()
            k = key(s)
            if not s or k in seen or k in held_out or not eligible(s):
                continue
            if args.unrecorded and int(r.get("clips_count") or 0) > 0:
                continue
            seen.add(k)
            pool.append(s)
            sources[s] = "cv"
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.pool]
    for path in args.extra_text:
        extra = 0
        for s in open(path, encoding="utf-8"):
            s = s.strip()
            k = key(s)
            if s and k not in seen and k not in held_out and eligible(s, names_ok=True):
                seen.add(k)
                pool.append(s)
                sources[s] = os.path.basename(path)
                extra += 1
        print(f"{extra} eligible sentences from {path}", file=sys.stderr)
    print(f"phonemizing {len(pool)} candidates", file=sys.stderr)

    from piper.phonemize_espeak import EspeakPhonemizer
    ph = EspeakPhonemizer()
    cand = []
    for s in pool:
        seq = [p for sent in ph.phonemize("eo", s) for p in sent if p not in "ˈˌ"]
        units = set(seq) | {a + b for a, b in zip(seq, seq[1:])} \
            | {a + b + c for a, b, c in zip(seq, seq[1:], seq[2:])}
        # word-final n and j/w before a vowel, from the spelling
        for w in key(s).split():
            if w.endswith("n"):
                units.add("n#")
            if re.search(r"[jŭ][aeiou]", w[1:]):
                units.add("glide+V")
        cand.append((s, units))

    freq = {}
    for _, u in cand:
        for x in u:
            freq[x] = freq.get(x, 0) + 1

    def weight(x):
        w = 1.0 / math.sqrt(freq[x])
        for b, m in BONUS.items():
            if b in x:
                w *= m
        if x in ("n#", "glide+V"):
            w *= 3.0
        return w

    W = {x: weight(x) for x in freq}
    covered = {}          # unit -> times covered; value decays so units keep getting practice

    # Cap any single unit's weight so one rare sound can't drag in an odd sentence.
    cap = sorted(W.values())[int(len(W) * 0.98)]
    W = {x: min(w, cap) for x, w in W.items()}

    def gain(i):
        g = sum(W[x] / (1 + covered.get(x, 0)) for x in cand[i][1])
        return g * (args.extra_boost if sources.get(cand[i][0], "cv") != "cv" else 1.0)

    # Lazy greedy: a candidate's gain only shrinks as coverage grows, so a
    # stale heap entry is an upper bound; re-score only the top until it holds.
    import heapq
    # --include-extra: every --extra-text sentence gets a guaranteed place
    # (written for flair, they would otherwise lose to rarer-sound sentences);
    # the greedy then fills the rest for coverage around them.
    seeded = []
    if args.include_extra:
        for i, (s, units) in enumerate(cand):
            if sources.get(s, "cv") != "cv":
                seeded.append(s)
                for x in units:
                    covered[x] = covered.get(x, 0) + 1
    seeded_set = set(seeded)
    heap = [(-gain(i), i) for i in range(len(cand)) if cand[i][0] not in seeded_set]
    heapq.heapify(heap)
    chosen = []
    while heap and len(chosen) + len(seeded) < args.n:
        neg, i = heapq.heappop(heap)
        g = gain(i)
        if heap and g < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-g, i))
            continue
        chosen.append(cand[i][0])
        for x in cand[i][1]:
            covered[x] = covered.get(x, 0) + 1

    # Spread the seeded sentences evenly, so every session has some flair
    # (the greedy order keeps the most coverage-valuable ones early).
    if seeded:
        merged, step = [], (len(chosen) + len(seeded)) / len(seeded)
        si = ci = 0
        for pos in range(len(chosen) + len(seeded)):
            if si < len(seeded) and (ci >= len(chosen) or pos >= si * step):
                merged.append(seeded[si]); si += 1
            else:
                merged.append(chosen[ci]); ci += 1
        chosen = merged

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "script.tsv"), "w", encoding="utf-8") as f:
        f.write("id\tsession\tsource\tsentence\n")
        for i, s in enumerate(chosen, 1):
            f.write(f"eo{i:04d}\t{(i - 1) // args.per_session + 1}\t{sources.get(s, 'cv')}\t{s}\n")
    for sess in range(0, len(chosen), args.per_session):
        n = sess // args.per_session + 1
        with open(os.path.join(args.out, f"session-{n:02d}.txt"), "w", encoding="utf-8") as f:
            for i, s in enumerate(chosen[sess: sess + args.per_session], sess + 1):
                f.write(f"eo{i:04d}  {s}\n")
    single = {x for x in freq if len(x) == 1}
    print(f"{len(chosen)} sentences in {math.ceil(len(chosen)/args.per_session)} sessions; "
          f"phonemes {len([x for x in single if x in covered])}/{len(single)}, "
          f"units {len(covered)}/{len(freq)}", file=sys.stderr)


if __name__ == "__main__":
    main()
