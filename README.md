# esperanto-voice

Open, natural-sounding Esperanto speech — for screen readers, learning apps,
and anyone else — built only from freely licensed data.

Today the only freely licensed Esperanto voice is espeak-ng: intelligible,
but robotic. The good neural voices that speak Esperanto (OmniVoice and
friends) carry non-commercial licenses inherited from their training data.
Esperanto is unusually well placed to fix this: its spelling is phonemic and
its stress is fixed, so a small model fine-tuned on a little clean audio can
go a long way.

## Goals

1. **A Piper voice for Esperanto** (`eo` doesn't exist in
   [piper-voices](https://huggingface.co/rhasspy/piper-voices) yet). Piper runs
   faster than real time on a CPU and plugs into NVDA, Orca/speech-dispatcher
   and Home Assistant, so one voice helps many projects at once.
2. **Every input and tool open**: public-domain or CC0 audio, a CC BY base
   checkpoint, MIT/Apache tooling — so the result can be released as
   CC BY 4.0 and packaged anywhere.
3. **Fast iteration over big data**: a `go test`-style harness scores every
   experiment in about a minute, so changes are judged by numbers, not hope.
4. **Speech recognition alongside**: the same aligned data serves ASR
   (e.g. pronunciation practice in [esperanto-kurso.net](https://esperanto-kurso.net)).

## Layout

| path | what |
|---|---|
| `tools/prepare_librivox.py` | Cut a LibriVox audiobook + Gutenberg text into sentence clips (CTC segmentation with an Esperanto wav2vec2 model). Unread text, like exercise questions, is dropped by confidence. |
| `tools/evaltest.py` | Regression harness: synthesize the test sentences, transcribe with Esperanto ASR, report per-case character error rate (CER) as PASS/FAIL. |
| `eval/sentences.txt` | Held-out test sentences covering every accented letter and stress pattern. |
| `espeak-ng/eo-glide-onset.patch` | *Experimental, not proposed upstream.* Makes `j`/`ŭ` before a vowel a syllable onset (`lerne-jo`) instead of a glide (`lernej-o`). Both occur in real speech, so this is a style choice, not a bug fix. |
| `docs/SOURCES.md` | Every data source, model and license used or considered. |

## The test harness

```sh
python tools/evaltest.py -v espeak                      # the baseline
python tools/evaltest.py espeak piper:eo-voice.onnx     # compare voices
```

```
--- FAIL: espeak/10 (CER 21.4%)
    want: la ĥoro kantis ĵaŭde vespere en la preĝejo
    got:  la ĵolo kantis laŭde mezreme en apreĝejo
FAIL  espeak                                   CER 15.4%  8/16 passed
```

CER measures intelligibility, not naturalness: it's the gate every change has
to pass, and listening decides between voices that pass it.

## Baseline

| voice | CER | passed |
|---|---|---|
| espeak-ng 1.52 (`-v eo`) | 15.4%* | 8/16 |

\* measured before two test sentences had their years spelled out; to be re-run.

## Status

Work in progress (October 2026). Aligning the first audiobook (*Karlo*, read
by J. D. Zero) and running data-size experiments: how little clean audio does
a fine-tuned Piper voice need to beat espeak-ng?

## License

Code: MIT. Test sentences and documentation: CC0. Trained voices will carry
the licenses of their inputs (see `docs/SOURCES.md`) — the aim is CC BY 4.0.
