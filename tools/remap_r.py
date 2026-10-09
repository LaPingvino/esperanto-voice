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

CONSONANTS = ["p", "b", "t", "d", "k", "ɡ", "f", "v", "x", "s", "z", "ʃ", "ʒ", "m", "n", "l"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--initial", default="ɹ,ɹ")
    ap.add_argument("--middle", default="ɹ")
    ap.add_argument("--after-consonant", default="ɹ")
    args = ap.parse_args()

    cfg = json.load(open(args.src, encoding="utf-8"))
    m = cfg["phoneme_id_map"]
    pad = m["_"][0]

    def seq(spec):
        out = []
        for i, p in enumerate(spec.split(",")):
            if i:
                out.append(pad)
            out.append(m[p][0])
        return out

    m["r"] = seq(args.middle)
    m["ɾ"] = seq(args.initial)
    clusters = {tuple(c) for c in cfg.get("vowel_clusters", [])}
    after = seq(args.after_consonant)
    for c in CONSONANTS:
        if c in m:
            clusters.add((c, "ɾ"))
            m[c + "ɾ"] = [m[c][0], pad] + after
    cfg["vowel_clusters"] = [list(c) for c in sorted(clusters)]
    json.dump(cfg, open(args.dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"r → {args.middle}; ɾ → {args.initial}; C+ɾ → C {args.after_consonant}  ({args.dst})")


if __name__ == "__main__":
    main()
