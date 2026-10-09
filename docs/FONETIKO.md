# Esperanto sounds: the norm, and how a voice gets each one right

Esperanto spelling is phonemic: one letter, one sound. The norm below follows
the *Fundamento* and PMEG's description of pronunciation; the "typical
deviation" column lists what text-to-speech voices (and learners) actually do
wrong — most of it inherited from the language a voice was first trained on.
The last column says which symbol espeak-ng writes and how this project covers
it.

**Base language.** Fine-tuning starts from an existing voice. An English base
needs a patch for almost every vowel and for r, aspiration and reduction (see
the history in `README.md`). An **Italian base** already has pure a e i o u,
alveolar trill and tap, unaspirated stops and the affricates; only *h* and
*ĥ* are missing (`tools/` seeds them).

## Vowels

All five are pure (no glide toward another vowel), keep their quality in
unstressed syllables, and are never reduced to a schwa.

| letter | norm (IPA) | typical deviation | espeak | notes |
|---|---|---|---|---|
| a | [a] open, as Italian *casa* | English [æ]/[ɑ], schwa when unstressed | `a` | |
| e | [e]~[ɛ], mid | diphthong [eɪ] ("day") | `e` | |
| i | [i] | lax [ɪ] | `i` | |
| o | [o]~[ɔ], mid | diphthong [oʊ] ("go") | `o` | |
| u | [u] back, rounded | fronted toward ü [ʉ] (modern English *goose*) | `u` | |

## Stress, length, rhythm

* **Stress** always on the second-to-last vowel; espeak marks it (`ˈ`).
* **No reduction:** unstressed syllables are shorter at most, never swallowed
  (*esperanta*, not *esperant'*). Voice setting: lower `noise_w`, or slightly
  higher `length_scale`.
* **Hiatus:** two vowels in a row are two syllables, with no glide inserted:
  *ki-el*, *hodi-aŭ*, *ti-u*. Piper's espeak inserts `ʲ` here; the pipeline
  removes it (`i`+`ʲ` → `i`, `tools/remap_r.py`).
* **Double consonants** across morphemes are pronounced long:
  *mal-longa*, *ek-kuri*.

## Consonants

| letter | norm (IPA) | typical deviation | espeak | notes |
|---|---|---|---|---|
| p t k | [p t k] **unaspirated** | English aspiration [pʰ tʰ kʰ]; an h-like puff carried into a following r | `p t k` | |
| b d g | [b d ɡ] voiced in every position | devoicing (*ĝia* → ĉia, *dudek* → tudek) | `b d ɡ` | |
| c | [t͡s] **one** sound | two sounds t + s | `t͡s` | merged token `ts` (`tools/add_affricates.py`) |
| ĉ | [t͡ʃ] one sound | t + ŝ | `t͡ʃ` | merged token `tʃ` |
| ĝ | [d͡ʒ] one sound | d + ĵ, or blurred toward ĵ | `d͡ʒ` | merged token `dʒ` |
| f v | [f v] | | `f v` | |
| h | [h] always pronounced | dropped (Romance speakers) | `h` | **missing in Italian base** — seeded |
| ĥ | [x], as German *Bach* | [k] or [h] | `x` | **missing in Italian base** — seeded |
| j | [j] consonant; after a vowel the second half of *aj ej oj uj* | English diphthongs [aɪ eɪ ɔɪ] | `j` / `ɪ` | |
| ĵ | [ʒ], as French *je* | [d͡ʒ] | `ʒ` | |
| l | [l] clear | dark, velarized [ɫ] (English, word-final) | `l` | |
| m n | [m n] | **n swallowed into a nasal vowel** (*malgrãda*); fatal for the accusative -n | `m n` | the trimming tool keeps word-final n (`cv_prepare --top-db 50`) |
| r | alveolar trill [r] or tap [ɾ] | English approximant [ɹ], uvular [ʁ], or a "hazy" buzz in TTS | `r` / `ɾ` | English base: remapped by ear (`tools/remap_r.py`); Italian base: native |
| s z | [s z] | | `s z` | *sc* is [st͡s] (*scii*), *ekz* is [kz] |
| ŝ | [ʃ] | | `ʃ` | |
| ŭ | [w], only after a/e (*aŭ eŭ*) | full vowel [u], or ü-like in *eŭ* | `ʊ` | English base: seeded from `w` |

## How this list is used

* **Testing:** `tools/evaltest.py` scores intelligibility (CER) and
  correctness (PER); `eval/sentences.txt` covers every row above.
* **Listening menus:** for anything a recognizer can't judge (quality of r,
  vowel colour, reduction), the pipeline renders numbered variants and a
  speaker picks by ear.
* **Chiselling:** wherever a row has a tool, the fix is a text- or
  config-level change, not more recordings.
