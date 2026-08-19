<!-- Naamche Labs -->

# Kriti Telephony

**A Nepali speech-recognition model tuned for the hardest real-world audio there is — live call-center conversation — with every number in this repository independently reproducible.**

Kriti Telephony leads **NepTel**, the public benchmark of real Nepali call-center speech, at **31.3% WER** — ahead of the previous best (NepaliConformer, 34.1%). It gets there by trading some clean-audio accuracy for a large gain on telephony, and this repository documents that trade-off, the exact method, and an authorized replay path so anyone can verify it.

> **This repo is built to be checked, not just read.** If you think we gamed the benchmark, good — the evidence to falsify that claim is all here: per-segment outputs, the scorer, the training recipe, and a second, independent metric (general-Nepali WER) reported next to every NepTel number.

---

## Headline results

All numbers are **word error rate (WER)**, lower is better, scored with the **official NepTel scorer** (word-level Levenshtein on normalized text). "General Nepali" is a held-out set of clean read Nepali with **human** references — it exists to catch benchmark-overfitting.

| System | NepTel WER ↓ | General-Nepali WER ↓ | Notes |
|---|---:|---:|---|
| NepaliConformer offline (prior best) | 34.09 | — | competitor; also the distillation *teacher* (see Method) |
| **Kriti** (base model) | 40.75 | **4.13** | excellent on clean audio, weaker on telephony |
| Pure distillation *(rejected — do not ship)* | 32.96 | 16.53 | wins NepTel but **collapses** on general Nepali → overfit |
| **Kriti Telephony** *(this release)* | **31.32** | 9.92 | leads NepTel **and** stays functional everywhere |

Our reproduction of the base model reads **40.75** on NepTel vs. its published **40.6** — a match to within rounding, which is how you know the harness is sound before trusting any other row.

**Read every result as a pair (telephony / general).** A model that only wins NepTel while its general-Nepali WER explodes (see the rejected row: 16.53) is gaming. A model that leads NepTel *and* keeps general-Nepali functional (9.92) is genuine domain adaptation. We ship only the second kind.

---

## The two models, and when to use each

| | **Kriti** | **Kriti Telephony** |
|---|---|---|
| Best for | clean / read / studio audio | phone calls, IVR, call-center, spontaneous speech |
| Clean-read WER | **4.1** | 9.9 |
| NepTel (telephony) WER | 40.8 | **31.3** |
| Relationship | the base model | a **domain sibling**, not a successor |

Kriti Telephony is **not** a strict upgrade — on clean audio the base Kriti is meaningfully better. Deploy the one that matches your audio. Always report the WER **pair**, never telephony alone.

---

## How it was made (short version)

Full detail in [`METHODOLOGY.md`](METHODOLOGY.md). The one-paragraph version:

1. **Diagnosis.** Base Kriti's NepTel errors are dominated by *style* — reference-orthography conventions (nasalization, vowel length) and English-loanword transliteration — not by mis-hearing. Straightforward fine-tuning failed five times in a row because every available training set carried a *different* label style and dragged the model the wrong way.
2. **Style-matched supervision (distillation).** We used an existing strong Nepali model (NepaliConformer) as a **teacher** to relabel ~12k clips of conversational Nepali audio in the target reference style, giving Kriti hard conversational audio paired with correctly-styled transcripts.
3. **Anti-forgetting anchor.** We mixed in general clean Nepali with **human** labels so the model keeps real transcription ability instead of becoming a benchmark impersonator. This single change took general-Nepali WER from **16.5** (pure distillation) down to **9.9**.
4. **Full fine-tune, validated on two axes** — NepTel *and* held-out general Nepali — so the gain is demonstrably domain adaptation, not leaderboard-fitting.

**We did not train on the NepTel evaluation audio or references.** The benchmark set carries a canary marker and was held out completely.

---

## "Did you just train on the benchmark?" — answered plainly

This is the fair question, and the honest answer has two parts:

- **No, not on the eval set.** No NepTel audio or reference ever appeared in training. Verify this yourself: the training manifests ([`train/`](train/)) reference only podcast, code-switched, and OpenSLR-54 audio — never the InfoBayAI call recordings NepTel is cut from.
- **But be precise about where the gain comes from.** Part of it is *matching NepTel's reference style*, and NepTel's references share a labeling lineage (Google Chirp-2 pseudo-labels) with the prior systems — a circularity the benchmark's own authors flag. That is exactly why we report **general-Nepali WER alongside every NepTel number.** The pure-distillation model that gamed the style (16.5 general) is documented here specifically so you can see what gaming looks like — and see that we rejected it.

The unassailable next step is validation on **neutral ground**: fresh Nepali call audio hand-transcribed independently of any existing model. We are building that set and will publish results on it — win or lose. See [`BENCHMARKS.md`](BENCHMARKS.md#neutral-benchmark-in-progress).

---

## Reproduce / verify everything

Full authorized-replay instructions in [`REPRODUCE.md`](REPRODUCE.md). At a glance:

```bash
# 1. Score any released hypothesis file against NepTel (no GPU, no model needed)
python eval/score.py --hyp benchmark/outputs/kriti-telephony.neptel.json
#   -> WER 31.32

# 2. Re-derive from the checkpoint (GPU): NepTel + general Nepali together
python eval/eval_both.py kriti_telephony.nemo kriti-telephony
#   -> NepTel=31.32  GeneralNepali=9.92
```

Every per-segment hypothesis for every model in the results table is committed under [`benchmark/outputs/`](benchmark/outputs/) as verifiable evidence — you can re-score them without a GPU and get the exact numbers above.

---

## What's in this repo

```
kriti-telephony/
├── README.md              # you are here
├── METHODOLOGY.md         # full pipeline: diagnosis → distillation → anchor → validation
├── RESULTS.md             # every experiment, both axes, with commentary
├── REPRODUCE.md           # step-by-step authorized replay (data access + commands)
├── BENCHMARKS.md          # NepTel + general-Nepali eval definitions; neutral-set plan
├── LICENSE                # MIT (our code); third-party assets credited below
├── eval/
│   ├── score.py           # the scorer (word-Levenshtein + speaking-rate gate)
│   ├── eval_neptel.py     # decode a checkpoint on NepTel, score
│   ├── eval_both.py       # decode + score on NepTel AND general Nepali
│   └── ortho_analysis.py  # the orthographic-style diagnostic
├── train/
│   ├── pseudo_label.py    # teacher relabels conversational audio (distillation)
│   ├── pseudo_label_all.py
│   ├── train_distill.py   # distillation-only recipe (documents the overfit failure)
│   └── train_kriti_telephony.py   # the shipped recipe (distill + general anchor)
├── benchmark/
│   ├── cut_neptel_segments.py     # reproduce the NepTel segments from source audio
│   └── outputs/           # per-segment hypotheses for every model = evidence
└── evidence/
    └── results.json       # machine-readable table of every number here
```

---

## Model lineage & credits

- **Kriti (base)** — Naamche Labs. A 119M-parameter hybrid RNNT-CTC Conformer built on **AI4Bharat IndicConformer** (multilingual, 22 Indic languages, multi-softmax joint) with a Nepali head and a Devanagari-danda punctuation head. This is the model Kriti Telephony adapts.
- **NepTel benchmark** — the [NepaliConformer team (Ampixa)](https://github.com/Ampixa/nepaliconformer). We reproduce their public benchmark and use their scorer as the canonical instrument. Credit and thanks — an open benchmark is what made this work checkable.
- **NepaliConformer** — [ampixa/nepali-conformer-offline](https://huggingface.co/ampixa/nepali-conformer-offline). Used here as the distillation *teacher*; disclosed fully in [`METHODOLOGY.md`](METHODOLOGY.md).
- **Source audio** — NepTel is cut from the **InfoBayAI** Nepali call-center dataset (gated, CC-BY-4.0). We do **not** redistribute it; [`REPRODUCE.md`](REPRODUCE.md) explains how to obtain access and regenerate the segments. Training audio (podcast / code-switched / OpenSLR-54) is credited in [`train/`](train/).

## License

Our code and documentation: **MIT** (see [`LICENSE`](LICENSE)). Third-party datasets, benchmarks, and models retain their own licenses as credited above. Checkpoints are released separately; see [`REPRODUCE.md`](REPRODUCE.md).

---

*Naamche Labs — Kriti. Honest ASR for Nepali. Report WER pairs (telephony / general), never telephony alone.*
