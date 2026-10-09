#!/usr/bin/env python3
"""Build the phoneme map + cluster list a voice is TRAINED with.

remap_r.py edits a finished voice's config so it realises r (and drops the ʲ
glide) the way Joop picked by ear. Applied only at synthesis, the model still
uses its English-trained tokens there, which drifts English in running speech.
Training with the same mapping lets every step on the narrator's audio pull
those tokens toward *his* sounds, in every context.

This writes, from Piper's default map:
  * the affricates dʒ / tʃ / ts as single tokens (ids 166-168),
  * the r / ɾ / consonant+ɾ mapping and glide removal (via remap_r's logic),
and prints the --data.phonemes_path and --data.vowel_clusters arguments for
experiment.py.

    training_map.py base/phonemes_train.json [remap_r options...]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DIPHTHONGS = [["a", "ɪ"], ["a", "ʊ"], ["e", "ɪ"], ["o", "ʊ"], ["ɔ", "ɪ"]]
AFFRICATES = {"dʒ": ["d", "ʒ"], "tʃ": ["t", "ʃ"], "ts": ["t", "s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args, remap_args = ap.parse_known_args()

    from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP
    m = {k: list(v) for k, v in DEFAULT_PHONEME_ID_MAP.items()}
    nid = max(v[0] for v in m.values()) + 1
    for tok in AFFRICATES:
        if tok not in m:
            m[tok] = [nid]
            nid += 1
    clusters = DIPHTHONGS + list(AFFRICATES.values())

    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.json"), os.path.join(td, "out.json")
        json.dump({"phoneme_id_map": m, "vowel_clusters": clusters}, open(src, "w"), ensure_ascii=False)
        subprocess.run([sys.executable, os.path.join(HERE, "remap_r.py"), src, dst] + remap_args, check=True)
        cfg = json.load(open(dst, encoding="utf-8"))

    json.dump(cfg["phoneme_id_map"], open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(cfg["vowel_clusters"], open(args.out + ".clusters.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("--data.phonemes_path", args.out)
    print("--data.vowel_clusters '" + json.dumps(cfg["vowel_clusters"], ensure_ascii=False) + "'")


if __name__ == "__main__":
    main()
