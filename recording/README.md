# Recording package

Everything a volunteer needs to record a new, consented, openly licensed
Esperanto voice.

| file | what |
|---|---|
| `GVIDILO.md` | Recording guide (Esperanto + English): room, microphone, format, reading style. |
| `KONSENTO.md` | Plain-language consent and licence text (CC BY 4.0 or CC0). |
| `cv-only/` | 1,200 sentences in 12 sessions of 100, chosen by `tools/make_script.py` from Common Voice's CC0 sentence list for full phoneme and phoneme-pair coverage, with extra weight on sounds earlier voices got wrong (trilled r, final -n, ĵ, ĥ, sc, j/ŭ before a vowel). Test sentences and Karlo's sentences are excluded. |

Even 3–4 sessions (30–45 minutes) are valuable: a 5-minute fine-tune already
beat espeak-ng.
