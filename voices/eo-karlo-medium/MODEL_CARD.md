# Model card for eo-karlo-medium (v0.1)

* Language: eo (Esperanto)
* Speakers: 1 (male)
* Quality: medium (Piper)
* Samplerate: 22,050 Hz
* License: **CC BY 4.0**

A freely licensed neural Esperanto voice for [Piper](https://github.com/OHF-Voice/piper1-gpl).
Usable anywhere Piper voices are: the `piper` CLI, NVDA, speech-dispatcher/Orca,
Home Assistant.

```sh
echo 'Saluton, mondo!' | piper -m eo-karlo-medium.onnx -f saluton.wav
```

## Data

* Audio: [*Karlo*](https://archive.org/details/karlo_2207_librivox) by Edmond
  Privat, read by J. D. Zero for LibriVox — public domain.
* Text: [Project Gutenberg #24525](https://www.gutenberg.org/ebooks/24525) — public domain.
* Aligned into sentence clips with `tools/prepare_librivox.py`. This v0.1
  used only the first **4.9 minutes** (41 clips of at most 12 s).

## Training

Fine-tuned from Piper's `_base_model`
([rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/_base_model)),
trained from scratch on LibriTTS-R, for 3 hours on an 8-core CPU (about 900
steps, batch size 4). Recipe: `tools/experiment.py --minutes 5 --train-minutes 180`.

## Evaluation

Character error rate of an Esperanto ASR model
([cpierse/wav2vec2-large-xlsr-53-esperanto](https://huggingface.co/cpierse/wav2vec2-large-xlsr-53-esperanto))
on the 16 held-out sentences in `eval/sentences.txt`:

| voice | CER | passed (≤10%) |
|---|---|---|
| espeak-ng 1.52 | 10.0% | 8/16 |
| **eo-karlo-medium v0.1** | **6.4%** | **13/16** |

## Known limitations

* The trilled **r** is weak and sometimes heard as *t* or dropped
  (*rivero* → "tivero"): the English base never learned it. Fixes are in progress.
* Trained on one narrator and very little data: expect some odd prosody on
  long or unusual sentences.
* Text goes through espeak-ng's Esperanto rules, so anything espeak-ng
  pronounces wrong, this voice does too.

## Attribution

This voice is a derivative of the Piper base model, which was trained on
LibriTTS-R (Koizumi et al., 2023, CC BY 4.0), itself derived from LibriTTS and
LibriSpeech (CC BY 4.0). Please credit:

> eo-karlo-medium by the esperanto-voice project
> (https://github.com/LaPingvino/esperanto-voice), CC BY 4.0; based on the
> Piper base model (LibriTTS-R, CC BY 4.0) and LibriVox audio read by J. D. Zero.
