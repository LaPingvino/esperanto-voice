#!/usr/bin/env python3
"""Text-only model surgery: give rarely-trained phonemes a sensible start.

The base checkpoint learned English. Phonemes that English espeak never
produces (the Esperanto trill `r`, Piper-espeak's hiatus glide `ʲ`, `x` for ĥ)
have embeddings that saw little or no training. With minutes of data, the
model may never learn them well from noise. This copies each such embedding
from its nearest well-trained English neighbour, so fine-tuning starts from
"roughly the right sound".

    seed_embeddings.py base/base_model.ckpt base/base_eo_seeded.ckpt

Pass the result to experiment.py with --base. No audio is involved.
"""

import argparse
import json
import os

# target phoneme  <-  source phonemes (averaged), all from the base id map
SEEDS = {
    "r": ["ɹ", "ɾ"],      # trill: English has the approximant and the tap
    # espeak's Esperanto writes about half of all r's as ɾ (reĝo, tre,
    # rapide), and the base learned ɾ as the American flap — the d-like sound
    # of "butter". In Esperanto r and ɾ are one sound, so ɾ gets the same
    # start as r. Without this, "reĝo" came out with a d-like r that blurred
    # the following ĝ toward ĵ.
    "ɾ": ["ɹ", "ɾ"],
    "x": ["h", "k"],      # ĥ: velar fricative, between the two
    # English espeak almost never emits plain a/e/o (it uses æ ɛ ɑ ɔ and
    # diphthongs), so in the base model these three — Esperanto's most
    # frequent vowels — have near-empty embeddings (norm ~0.23 vs ~0.9).
    "a": ["ɑ"],
    "e": ["ɛ"],
    "o": ["ɔ"],
    # Hiatus glide Piper's espeak inserts in lia, kiel, hodiaŭ. Seeding it
    # from j made consonants sound palatalized ("Russian"); from i it stays
    # a smooth transition.
    "ʲ": ["i"],
    # A lone ʊ only occurs as the glide of eŭ (aŭ becomes the merged aʊ
    # token). The base learned ʊ as the English vowel of "book", which next
    # to e drifts toward ü; Esperanto wants e + a short w-like glide.
    "ʊ": ["w"],
}
# v0.2 candidate used: r ← ɹ,ɾ; ʲ ← j; x ← h,k   (--only r,ʲ,x reproduces it
# except for the ʲ donor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--config", default=None,
                    help="config.json with phoneme_id_map (default: next to src)")
    ap.add_argument("--only", default="",
                    help="comma-separated subset of target phonemes to seed (default: all)")
    args = ap.parse_args()

    import torch
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    cfg = json.load(open(args.config or os.path.join(os.path.dirname(args.src), "config.json")))
    idmap = cfg["phoneme_id_map"]
    sd = ck["state_dict"]
    key = next(k for k in sd if k.endswith("enc_p.emb.weight"))
    emb = sd[key]
    norms = emb.norm(dim=1)
    only = set(filter(None, args.only.split(",")))
    for dst, srcs in SEEDS.items():
        if only and dst not in only:
            continue
        if dst not in idmap or not all(s in idmap for s in srcs):
            print(f"skip {dst}: not in phoneme map")
            continue
        d = idmap[dst][0]
        vec = torch.stack([emb[idmap[s][0]] for s in srcs]).mean(0)
        print(f"{dst!r}: norm {norms[d]:.3f} -> {vec.norm():.3f}  (from {', '.join(srcs)})")
        emb[d] = vec
    torch.save(ck, args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
