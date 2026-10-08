#!/usr/bin/env python3
"""go-test-style regression harness for Esperanto voices.

Each line of the sentence file is a test case. A voice synthesizes every
case, an Esperanto ASR model (wav2vec2, Apache-2.0, trained on Common Voice)
transcribes the audio, and the case passes when the character error rate
(CER) between transcript and sentence is at most --max-cer.

CER measures intelligibility, not beauty: a robotic voice can score well. It
is the gate every change must pass; listening decides between voices that do.

    evaltest.py espeak                         # espeak-ng -v eo
    evaltest.py piper:path/to/eo-voice.onnx    # a Piper voice
    evaltest.py dir:path/to/wavs               # pre-rendered 01.wav, 02.wav, …
    evaltest.py -v --max-cer 0.10 espeak piper:voice.onnx

Output mirrors `go test -v`:

    === RUN   espeak/03
    --- PASS: espeak/03 (CER 2.6%)
    ...
    ok    espeak    CER 4.1%  15/16 passed
    FAIL  piper:x   CER 12.0%  9/16 passed

Exit status is non-zero when any voice fails a case. Rendered audio is kept
under --out/<voice>/ for listening.
"""

import argparse
import os
import re
import subprocess
import sys
import unicodedata

ASR_MODEL = "cpierse/wav2vec2-large-xlsr-53-esperanto"
HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^a-zĉĝĥĵŝŭ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    return edit_distance(ref, hyp) / max(1, len(ref))


def render(voice, cases, outdir):
    """Write outdir/NN.wav for each case; return the paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = [os.path.join(outdir, f"{i:02d}.wav") for i in range(1, len(cases) + 1)]
    kind, _, arg = voice.partition(":")
    if kind == "espeak":
        for p, text in zip(paths, cases):
            subprocess.run(["espeak-ng", "-v", arg or "eo", "-w", p, text], check=True)
    elif kind == "piper":
        from piper import PiperVoice
        import wave
        v = PiperVoice.load(arg)
        for p, text in zip(paths, cases):
            with wave.open(p, "wb") as w:
                v.synthesize_wav(text, w)
    elif kind == "dir":
        paths = [os.path.join(arg, f"{i:02d}.wav") for i in range(1, len(cases) + 1)]
    else:
        sys.exit(f"unknown voice kind {kind!r} (want espeak, piper:<onnx>, dir:<path>)")
    return paths


class ASR:
    def __init__(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(ASR_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL).eval()

    def __call__(self, path):
        import librosa
        wav, _ = librosa.load(path, sr=16000, mono=True)
        with self.torch.no_grad():
            inp = self.proc(wav, sampling_rate=16000, return_tensors="pt").input_values
            ids = self.model(inp).logits.argmax(-1)
        return self.proc.batch_decode(ids)[0]


PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"   # Apache-2.0

# Variants that are all acceptable Esperanto collapse to one symbol; real
# errors (a breathy "x" before a vowel, a missing n, g→k) survive.
PHONE_EQUIV = [("ɾ", "r"), ("ʲ", ""), ("ɪ", "j"), ("ʊ", "w"), ("ɡ", "g"),
               ("ɛ", "e"), ("ɔ", "o"), ("ə", "e"), ("ɐ", "a"), ("ɑ", "a")]


def phone_norm(s):
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"[ˈˌːˑ͡‍\s.,;:!?'\"-]", "", s)
    for a, b in PHONE_EQUIV:
        s = s.replace(a, b)
    return s


class PhonemeASR:
    """What was actually *said*, as espeak-style phonemes."""

    def __init__(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(PHONEME_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).eval()
        from piper.phonemize_espeak import EspeakPhonemizer
        self.espeak = EspeakPhonemizer()

    def expected(self, text):
        return "".join("".join(s) for s in self.espeak.phonemize("eo", text))

    def __call__(self, path):
        import librosa
        wav, _ = librosa.load(path, sr=16000, mono=True)
        with self.torch.no_grad():
            inp = self.proc(wav, sampling_rate=16000, return_tensors="pt").input_values
            ids = self.model(inp).logits.argmax(-1)
        return self.proc.batch_decode(ids)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("voices", nargs="+")
    ap.add_argument("--cases", default=os.path.join(HERE, "..", "eval", "sentences.txt"))
    ap.add_argument("--out", default="eval-out")
    ap.add_argument("--max-cer", type=float, default=0.10)
    ap.add_argument("-v", action="store_true", help="print every case")
    ap.add_argument("-run", default="", help="only cases whose text matches this regex")
    ap.add_argument("--no-per", action="store_true",
                    help="skip the phoneme error rate (saves loading a second model)")
    args = ap.parse_args()

    cases = [l.strip() for l in open(args.cases, encoding="utf-8") if l.strip()]
    ids = [i for i, c in enumerate(cases) if re.search(args.run, c)]
    asr = ASR()
    pasr = None if args.no_per else PhonemeASR()
    failed_any = False

    for voice in args.voices:
        name = re.sub(r"[^\w.-]+", "_", voice)
        paths = render(voice, cases, os.path.join(args.out, name))
        total_err = total_len = passed = 0
        p_err = p_len = 0
        vowel_starts = breathy = 0
        for i in ids:
            hyp = asr(paths[i])
            c = cer(cases[i], hyp)
            total_err += edit_distance(norm(cases[i]), norm(hyp))
            total_len += len(norm(cases[i]))
            ok = c <= args.max_cer
            passed += ok
            per_note = ""
            if pasr:
                want_p = phone_norm(pasr.expected(cases[i]))
                raw_p = pasr(paths[i])
                got_p = phone_norm(raw_p)
                # Breathy onset: a vowel-initial sentence heard with a
                # leading fricative ("x e s p e r a …").
                if want_p[:1] in "aeiou":
                    vowel_starts += 1
                    breathy += raw_p.strip()[:1] in "xhsfçχθ"
                e = edit_distance(want_p, got_p)
                p_err += e
                p_len += len(want_p)
                per_note = f", PER {e / max(1, len(want_p)):.1%}"
            if args.v or not ok:
                if args.v:
                    print(f"=== RUN   {voice}/{i+1:02d}")
                print(f"--- {'PASS' if ok else 'FAIL'}: {voice}/{i+1:02d} (CER {c:.1%}{per_note})")
                if not ok:
                    print(f"    want: {norm(cases[i])}\n    got:  {norm(hyp)}")
                if pasr and args.v:
                    print(f"    phones want: {want_p}\n    phones got:  {got_p}")
        agg = total_err / max(1, total_len)
        per = (f"  PER {p_err / max(1, p_len):5.1%}  breathy onsets {breathy}/{vowel_starts}"
               if pasr else "")
        status = "ok  " if passed == len(ids) else "FAIL"
        failed_any |= passed != len(ids)
        print(f"{status}  {voice:40s} CER {agg:5.1%}{per}  {passed}/{len(ids)} passed", flush=True)

    sys.exit(1 if failed_any else 0)


if __name__ == "__main__":
    main()
