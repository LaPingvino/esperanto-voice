#!/usr/bin/env python3
"""Make ĝ, ĉ and c single sounds instead of two-token sequences.

espeak writes Esperanto's affricates with a tie bar (d͡ʒ, t͡ʃ, t͡s), but Piper
drops it and feeds the model two tokens: d + ʒ, t + ʃ, t + s. The model then
learns them as sequences and may leave a gap or extra length between the
halves; Esperanto speakers hear ĝ as one merged sound, not d + ĵ.

Piper's (generically named) vowel-cluster mechanism merges any listed token
sequence into one token, and the cluster list lives in the voice config, so
training and every Piper install (1.5+) apply the same merge. This tool:

  1. writes a phoneme map = Piper's default map + dʒ, tʃ, ts appended
     (ids 166-168; existing ids are untouched),
  2. seeds those three embedding rows in a checkpoint from the mean of their
     halves (rows exist: the model has 256 symbol slots).

    add_affricates.py base/base_eo_vowels_r2.ckpt base/base_eo_vafr.ckpt \
        --map-out base/phonemes_affricates.json

Then train with:
    --data.phonemes_path base/phonemes_affricates.json
    --data.vowel_clusters '[["a","ɪ"],["a","ʊ"],["e","ɪ"],["o","ʊ"],["ɔ","ɪ"],
                            ["d","ʒ"],["t","ʃ"],["t","s"]]'
"""

import argparse
import json

AFFRICATES = {"dʒ": ("d", "ʒ"), "tʃ": ("t", "ʃ"), "ts": ("t", "s")}   # ĝ, ĉ, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--map-out", required=True)
    args = ap.parse_args()

    import torch
    from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP

    m = {k: list(v) for k, v in DEFAULT_PHONEME_ID_MAP.items()}
    nid = max(v[0] for v in m.values()) + 1
    for tok in AFFRICATES:
        if tok not in m:
            m[tok] = [nid]
            nid += 1
    with open(args.map_out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=0)

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    emb = next(v for k, v in ck["state_dict"].items() if k.endswith("enc_p.emb.weight"))
    if nid > emb.shape[0]:
        raise SystemExit(f"model has only {emb.shape[0]} symbol slots")
    for tok, (a, b) in AFFRICATES.items():
        i = m[tok][0]
        emb[i] = (emb[m[a][0]] + emb[m[b][0]]) / 2
        print(f"{tok} → id {i}, seeded from {a}+{b} (norm {float(emb[i].norm()):.2f})")
    torch.save(ck, args.dst)
    print(f"wrote {args.dst} and {args.map_out}")


if __name__ == "__main__":
    main()
