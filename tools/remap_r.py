#!/usr/bin/env python3
"""Choose how a voice realises Esperanto r — by editing only its config.

A Piper voice config maps each phoneme symbol to one *or more* token ids, and
lists symbol sequences to merge ("vowel_clusters", used for any sequence).
espeak's Esperanto writes word-initial and post-consonant r as ɾ and other
r's as r, so the realisation can be chosen per position without retraining:

    r            (middle, end: kara, kvar)       → ɹ
    ɾ            (word-initial: rozo, reĝo)       → ɹ ɹ
    C + ɾ        (after a consonant: tri, granda) → C ɹ   (merged as a cluster)

These were picked by ear from token-substitution menus (Joop, 2026-10-09):
the English-trained ɹ sounded best in the middle, a doubled ɹɹ word-initially,
and the doubled form was too heavy after consonants.

    remap_r.py voice.onnx.json out.onnx.json [--initial ɹ,ɹ] [--middle ɹ] [--after-consonant ɹ]

Works with Piper 1.5+ (cluster merging). The model file is unchanged.
"""

import argparse
import json

VOICELESS = ["p", "t", "k", "f", "x", "s", "ʃ"]
VOICED = ["b", "d", "ɡ", "v", "z", "ʒ", "m", "n", "l"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    # Specs: tokens joined by "," get the pause token between them (as Piper
    # normally does), joined by "+" they are adjacent (no pause).
    ap.add_argument("--initial", default="ɹ+ɹ")
    ap.add_argument("--middle", default="ɹ")
    ap.add_argument("--after-voiceless", default="ɾ",
                    help="after p t k f ĥ s ŝ: the English ɹ carries an h-like puff there (try/pray)")
    ap.add_argument("--after-voiced", default="ɹ")
    ap.add_argument("--keep-glide", action="store_true",
                    help="keep the ʲ glide Piper's espeak inserts in kiel/tiu/lia/hodiaŭ; by "
                         "default it is mapped to nothing (Joop: clearly better, esp. hodiaŭ)")
    args = ap.parse_args()

    cfg = json.load(open(args.src, encoding="utf-8"))
    m = cfg["phoneme_id_map"]
    pad = m["_"][0]
    # Token ids as trained, captured before r/ɾ get remapped below (otherwise
    # a spec mentioning ɾ would pick up its *new* mapping).
    orig = {k: v[0] for k, v in m.items()}

    def seq(spec):
        out = []
        for i, group in enumerate(spec.split(",")):
            if i:
                out.append(pad)
            out += [orig[p] for p in group.split("+")]
        return out

    m["r"] = seq(args.middle)
    m["ɾ"] = seq(args.initial)
    if not args.keep_glide and "ʲ" in m:
        m["ʲ"] = []          # a symbol may map to no ids at all: the glide simply disappears
    clusters = {tuple(c) for c in cfg.get("vowel_clusters", [])}
    for group, spec in ((VOICELESS, args.after_voiceless), (VOICED, args.after_voiced)):
        after = seq(spec)
        for c in group:
            if c in m:
                clusters.add((c, "ɾ"))
                m[c + "ɾ"] = [m[c][0], pad] + after
    cfg["vowel_clusters"] = [list(c) for c in sorted(clusters)]
    json.dump(cfg, open(args.dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"r → {args.middle}; ɾ → {args.initial}; voiceless+ɾ → C {args.after_voiceless}; "
          f"voiced+ɾ → C {args.after_voiced}  ({args.dst})")


if __name__ == "__main__":
    main()
