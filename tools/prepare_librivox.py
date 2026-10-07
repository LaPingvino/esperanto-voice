#!/usr/bin/env python3
"""Cut a LibriVox Esperanto audiobook into sentence clips for TTS training.

Public-domain audio (LibriVox) + public-domain text (Project Gutenberg) in,
a Piper-style dataset out:

    <out>/wav/<id>.wav         mono 22.05 kHz 16-bit clips, one sentence each
    <out>/metadata.csv         <id>|<text>   (Piper's LJSpeech-style format)
    <out>/rejected.csv         clips dropped by the confidence filter, for review

Alignment is CTC segmentation (ctc-segmentation) against an Esperanto
wav2vec2 model (cpierse/wav2vec2-large-xlsr-53-esperanto, Apache-2.0, trained
on Common Voice). The whole chapter text is aligned, including parts the
narrator may have skipped (exercise questions, word lists): those get low
confidence scores and are dropped, so the text does not need hand-editing.

Usage:
    prepare_librivox.py --text karlo.txt --audio-dir karlo_mp3/ --out data/karlo \
        [--chapter-re '^\\d+\\. [A-ZĈĜĤĴŜŬ]'] [--min-score -1.5]

Chapter i of the text (after splitting on --chapter-re; the part before the
first match is chapter 0) is aligned against the i-th audio file in sorted
order. Use --skip-audio / --skip-text to drop unmatched leading sections.
"""

import argparse
import os
import re
import sys
import unicodedata

XSYS = {"cx": "ĉ", "gx": "ĝ", "hx": "ĥ", "jx": "ĵ", "sx": "ŝ", "ux": "ŭ",
        "Cx": "Ĉ", "Gx": "Ĝ", "Hx": "Ĥ", "Jx": "Ĵ", "Sx": "Ŝ", "Ux": "Ŭ",
        "CX": "Ĉ", "GX": "Ĝ", "HX": "Ĥ", "JX": "Ĵ", "SX": "Ŝ", "UX": "Ŭ"}

SAMPLE_RATE = 22050  # Piper medium voices


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def from_xsystem(s):
    # "ux" only means ŭ after a vowel (aŭ, eŭ); elsewhere leave it alone.
    s = re.sub(r"([aeAE])(ux|uX|UX|Ux)",
               lambda m: m.group(1) + ("Ŭ" if m.group(2)[0] == "U" else "ŭ"), s)
    return re.sub(r"[cghjsCGHJS][xX]", lambda m: XSYS.get(m.group(0), m.group(0)), s)


def gutenberg_body(raw):
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF.*?\*\*\*", raw)
    return raw[start.end() if start else 0: end.start() if end else len(raw)]


def split_sentences(chapter):
    text = re.sub(r"_", "", chapter)                 # Gutenberg italics
    text = re.sub(r"\s*--\s*", " — ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?…])\s+(?=[\"«(—]?[A-ZĈĜĤĴŜŬ0-9])", text)
    out = []
    for p in parts:
        p = p.strip(" —")
        if len(re.sub(r"[^\wĉĝĥĵŝŭ]", "", p.lower())) >= 3:
            out.append(p)
    return out


def ctc_text(s, vocab):
    """Normalize a sentence to the ASR model's character set."""
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[’']", "'", s)
    keep = "".join(ch if ch in vocab else " " for ch in s)
    return re.sub(r"\s+", " ", keep).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    # A heading is a short numbered line on its own ending in a period
    # ("1. La familio."), followed by a blank line. Numbered exercise
    # questions also start a line ("9. Ĉu vi jam provis ekzamenon?") but end
    # in "?", and table-of-contents entries read "1. -- La familio".
    ap.add_argument("--chapter-re", default=r"^\d{1,2}\. [^-\n][^\n]{1,40}\.\n[ \t]*\n")
    ap.add_argument("--skip-text", type=int, default=0)
    ap.add_argument("--skip-audio", type=int, default=0)
    ap.add_argument("--min-score", type=float, default=-1.5,
                    help="minimum mean log-prob per frame to keep a clip")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--model", default="cpierse/wav2vec2-large-xlsr-53-esperanto")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from ctc_segmentation import (CtcSegmentationParameters, ctc_segmentation,
                                  determine_utterance_segments, prepare_text)
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    raw = from_xsystem(open(args.text, encoding="utf-8").read())
    body = gutenberg_body(raw)
    chapters = re.split(args.chapter_re, body, flags=re.M)
    # re.split drops the matched heading prefix; that's fine — headings are
    # short and the aligner tolerates the missing words.
    chapters = chapters[args.skip_text:]
    audio = sorted(f for f in os.listdir(args.audio_dir)
                   if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg")))
    audio = audio[args.skip_audio:]
    n = min(len(chapters), len(audio))
    log(f"{len(chapters)} text chapters, {len(audio)} audio files → aligning {n}")

    proc = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).eval()
    vocab_dict = proc.tokenizer.get_vocab()
    blank = vocab_dict.get(proc.tokenizer.pad_token, 0)
    char_list = [None] * len(vocab_dict)
    for ch, i in vocab_dict.items():
        char_list[i] = ch
    char_set = set(c for c in char_list if c and len(c) == 1)
    word_sep = proc.tokenizer.word_delimiter_token or "|"

    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    meta = open(os.path.join(args.out, "metadata.csv"), "a", encoding="utf-8")
    rej = open(os.path.join(args.out, "rejected.csv"), "a", encoding="utf-8")
    kept = dropped = 0
    kept_sec = 0.0

    for ci in range(n):
        sents = split_sentences(chapters[ci])
        norm = [ctc_text(s, char_set | {" "}).replace(" ", word_sep) for s in sents]
        pairs = [(s, t) for s, t in zip(sents, norm) if t]
        if not pairs:
            continue
        sents, norm = [p[0] for p in pairs], [p[1] for p in pairs]

        wav16, _ = librosa.load(os.path.join(args.audio_dir, audio[ci]), sr=16000, mono=True)
        wav22, _ = librosa.load(os.path.join(args.audio_dir, audio[ci]), sr=SAMPLE_RATE, mono=True)

        # Long chapters don't fit wav2vec2 in one pass on modest RAM: run it
        # in 30 s windows and concatenate the frame log-probs.
        win = 16000 * 30
        logps = []
        with torch.no_grad():
            for off in range(0, len(wav16), win):
                chunk = wav16[off: off + win]
                if len(chunk) < 400:
                    continue
                inp = proc(chunk, sampling_rate=16000, return_tensors="pt").input_values
                lp = torch.log_softmax(model(inp).logits[0], dim=-1)
                logps.append(lp.numpy())
        lpz = np.concatenate(logps)

        params = CtcSegmentationParameters()
        params.char_list = char_list
        params.blank = blank
        params.index_duration = len(wav16) / lpz.shape[0] / 16000
        params.replace_spaces_with_blanks = False
        gt, utt_begin = prepare_text(params, norm)
        timings, char_probs, _ = ctc_segmentation(params, lpz, gt)
        segs = determine_utterance_segments(params, utt_begin, char_probs, timings, norm)

        for si, ((start, end, score), sent) in enumerate(zip(segs, sents)):
            dur = end - start
            cid = f"{args.prefix}{ci:02d}_{si:04d}"
            ok = score >= args.min_score and args.min_sec <= dur <= args.max_sec
            if not ok:
                dropped += 1
                rej.write(f"{cid}|{score:.2f}|{dur:.1f}|{sent}\n")
                continue
            a = max(0, int((start - 0.05) * SAMPLE_RATE))
            b = min(len(wav22), int((end + 0.10) * SAMPLE_RATE))
            sf.write(os.path.join(args.out, "wav", cid + ".wav"), wav22[a:b],
                     SAMPLE_RATE, subtype="PCM_16")
            meta.write(f"{cid}|{sent}\n")
            kept += 1
            kept_sec += dur
        meta.flush()
        rej.flush()
        log(f"  {audio[ci]}: {len(sents)} sentences, kept so far {kept} "
            f"({kept_sec/60:.1f} min), dropped {dropped}")

    log(f"done: {kept} clips, {kept_sec/60:.1f} min kept, {dropped} dropped")


if __name__ == "__main__":
    main()
