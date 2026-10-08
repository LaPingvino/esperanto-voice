# Recording package

Everything a volunteer needs to record a new, consented, openly licensed
Esperanto voice.

| file | what |
|---|---|
| `GVIDILO.md` | Recording guide (Esperanto + English): room, microphone, format, reading style. |
| `KONSENTO.md` | Plain-language consent and licence text (CC BY 4.0 or CC0). |
| `script/` | **The script:** 1,200 sentences in 12 sessions of 100 (`session-01.txt` …, `script.tsv` with ids and sources). |
| `originalaj-frazoj.txt` | 208 sentences written for this project (CC0): everyday scenes, dialogue, and original *langorompiloj* (tongue twisters) drilling r, ŝ, ĉ, ĵ, ĥ, sc, aŭ, kv, tr and the accusative -n. All of them are in the script, spread over every session. |

The other 992 sentences come from Common Voice's CC0 sentence list, chosen by
`tools/make_script.py` for full phoneme and phoneme-pair coverage, with extra
weight on sounds earlier voices got wrong. Excluded: the test sentences,
Karlo's sentences, hyphenated coinages, foreign spellings, mid-sentence
proper names, and obvious typos. Some odd sentences will remain — the guide
asks the reader to simply skip them.

Even 3–4 sessions (30–45 minutes) are valuable: a 5-minute fine-tune already
beat espeak-ng.
