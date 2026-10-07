# Sources and licenses

Everything this project uses or considered, with the license that decides
whether it can go into a freely licensed voice.

## Used

| what | license | role |
|---|---|---|
| [LibriVox](https://librivox.org) Esperanto audiobooks | public domain | training audio |
| [Project Gutenberg](https://www.gutenberg.org) Esperanto texts | public domain | transcripts |
| [Piper `_base_model`](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/_base_model) — trained from scratch on LibriTTS-R | CC BY 4.0 | fine-tuning base (attribution required) |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) training & runtime | GPL-3.0 (code) | training, inference |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) | GPL-3.0 | phonemizer (and the baseline voice) |
| [cpierse/wav2vec2-large-xlsr-53-esperanto](https://huggingface.co/cpierse/wav2vec2-large-xlsr-53-esperanto) | Apache-2.0 | alignment and evaluation ASR |

### LibriVox Esperanto recordings

| book | reader | length |
|---|---|---|
| [Karlo](https://archive.org/details/karlo_2207_librivox) (Privat) | J. D. Zero | 2:00 |
| [La Mirinda Sorĉisto de Oz](https://archive.org/details/mirinda_sorcisto_de_oz_jdz_2304_librivox) (Baum, tr. Broadribb) | J. D. Zero | 4:54 |
| [Vivo de Zamenhof](https://archive.org/details/vivo_de_zamenhof_2207_librivox) (Privat) | Nightflush, J. D. Zero | 3:59 |
| [Vojaĝo interne de mia ĉambro](https://archive.org/details/vojagointernedemiacambro_2409_librivox) (de Maistre) | VerdaVocxo | 2:42 |
| [Fabeloj de Andersen](https://archive.org/details/fabeloj_andersen_1505_librivox) | Rosslyn Carlyle | 1:37 |
| [Aventuroj de Alicio en Mirlando](https://archive.org/details/laaventuro_de_alicio0912_librivox) (tr. Kearney) | Nicholas James Bridgewater | 4:22 |

## Available, not yet used

| what | license | notes |
|---|---|---|
| [Common Voice Esperanto](https://datacollective.mozillafoundation.org/datasets/cmflnuzw5u1plk8mv3fyiv6vf) v23 | CC0 | 1,437 validated hours, 1,884 speakers; uneven recording conditions, so per-speaker filtering needed. Download needs a Mozilla Data Collective account. |
| [Vosk `vosk-model-small-eo-0.42`](https://alphacephei.com/vosk/models) | Apache-2.0 | 41 MB offline ASR; runs in the browser via vosk-browser — candidate for pronunciation practice. |

## Considered and rejected

| what | why not |
|---|---|
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) weights | CC-BY-NC "due to constraints from its training data (e.g., Emilia)"; its tokenizer (Higgs Audio v2) adds a usage cap. The code is Apache-2.0 and third-party sites misreport the weights as Apache too. Its Esperanto data (~1,397 h) is Common Voice, i.e. CC0. A more openly licensed model is planned upstream ([#235](https://github.com/k2-fsa/OmniVoice/issues/235)). |
| Most Piper language checkpoints | fine-tuned from the `lessac` voice, whose dataset license is not open; `_base_model` avoids this. |
| Zonos, Kokoro, Chatterbox | permissive, but no Esperanto. |
| Meta MMS-TTS | no `epo` checkpoint, and CC-BY-NC anyway. |
