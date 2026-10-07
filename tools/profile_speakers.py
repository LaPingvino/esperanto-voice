#!/usr/bin/env python3
"""Profile Common Voice speakers: who has which pronunciation traits.

Common Voice gives us hundreds of Esperanto speakers. Instead of averaging
them all into one model, we pick speakers (or clips) per trait — crisp n,
trilled r, clean vowel onsets, good microphones. This script measures those
traits on a sample of each speaker's clips:

    profile_speakers.py dataset/cv27-sample/speakers --clips 20 > profiles.tsv

Columns (all per speaker, over the sampled clips):
    cer        word-ASR character error rate        (intelligibility)
    per        phoneme error rate vs. espeak        (correctness)
    nasal      nasal vowels heard per 100 written n (nasalization habit)
    r_trill    share of r heard as r/ɾ (rest: uvular ʁ/ʀ, dropped, …)
    breath     share of vowel-initial clips heard with a leading fricative
    snr_db     rough signal-to-noise estimate
    rolloff    95% spectral rolloff in kHz (low = band-limited microphone)
    rate       phonemes per second

Input dirs are produced by `cv_stream.py clips`: <dir>/<speaker>/metadata.csv
and <dir>/<speaker>/mp3/*.mp3.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaltest as E  # noqa: E402  (ASR, PhonemeASR, normalizers)

NASAL = "̃"            # combining tilde
VOWELS = "aeiouAEIOU"
FRICATIVES = set("hxsfθʃçχ")


def audio_stats(y, sr):
    import numpy as np
    import librosa
    frame = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0] + 1e-9
    lo, hi = np.percentile(frame, 10), np.percentile(frame, 90)
    snr = 20 * np.log10(hi / lo)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95,
                                            n_fft=1024, hop_length=256)[0]
    n = min(len(roll), len(frame))
    roll, frame = roll[:n], frame[:n]
    loud = frame > np.percentile(frame, 50)
    return float(snr), float(np.median(roll[loud]) / 1000 if loud.any() else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speakers_dir")
    ap.add_argument("--clips", type=int, default=20)
    args = ap.parse_args()

    import librosa
    import numpy as np
    asr, pasr = E.ASR(), E.PhonemeASR()

    print("speaker\tclips\tcer\tper\tnasal\tr_trill\tbreath\tsnr_db\trolloff\trate", flush=True)
    for spk in sorted(os.listdir(args.speakers_dir)):
        d = os.path.join(args.speakers_dir, spk)
        meta = os.path.join(d, "metadata.csv")
        if not os.path.exists(meta):
            continue
        rows = [l.rstrip("\n").split("|", 1) for l in open(meta, encoding="utf-8") if "|" in l]
        rows = rows[: args.clips]
        ce = cl = pe = pl = 0
        n_written = nasal = r_written = r_trill = 0
        vowel_init = breathy = 0
        snrs, rolls, rates = [], [], []
        for cid, text in rows:
            path = os.path.join(d, "mp3", cid + ".mp3")
            try:
                y, sr = librosa.load(path, sr=16000, mono=True)
            except Exception:
                continue
            if len(y) < 1600:
                continue
            tmp = path  # both ASRs reload at 16 kHz; librosa decodes mp3 directly
            hyp = asr(tmp)
            ce += E.edit_distance(E.norm(text), E.norm(hyp))
            cl += len(E.norm(text))

            raw_ph = pasr(tmp)                 # space-separated phoneme tokens
            want = pasr.expected(text)
            w, g = E.phone_norm(want), E.phone_norm(raw_ph)
            pe += E.edit_distance(w, g)
            pl += len(w)

            toks = raw_ph.split()
            n_written += text.lower().count("n")
            nasal += sum(NASAL in t for t in toks)
            r_written += text.lower().count("r")
            r_trill += sum(t in ("r", "ɾ", "rː", "ɾː") for t in toks)
            first = E.norm(text)[:1]
            if first and first in VOWELS.lower():
                vowel_init += 1
                if toks and toks[0][0] in FRICATIVES:
                    breathy += 1
            s, r = audio_stats(y, 16000)
            snrs.append(s)
            rolls.append(r)
            rates.append(len(toks) / (len(y) / 16000))
        n = len(snrs)
        if not n:
            continue
        print(f"{spk}\t{n}\t{ce/max(1,cl):.3f}\t{pe/max(1,pl):.3f}\t"
              f"{100*nasal/max(1,n_written):.1f}\t{r_trill/max(1,r_written):.2f}\t"
              f"{breathy/max(1,vowel_init):.2f}\t{np.median(snrs):.1f}\t"
              f"{np.median(rolls):.1f}\t{np.median(rates):.1f}", flush=True)


if __name__ == "__main__":
    main()
