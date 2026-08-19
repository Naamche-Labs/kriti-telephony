# Benchmarks

Two evaluation sets are used throughout this repo. Reporting both together is the core methodological commitment.

## NepTel (telephony, the headline metric)

- **What:** real, spontaneous Nepali call-center conversation, the hardest real-world Nepali audio with public references. 3 calls, one vendor, one domain; about 77 segments, 75 to 76 scored after gating, roughly 2,375 reference words.
- **Author:** the [NepaliConformer team (Ampixa)](https://github.com/Ampixa/nepaliconformer). This is their benchmark; we reproduce it and use their scorer.
- **Source audio:** cut from the gated InfoBayAI dual-channel Nepali call-center dataset (CC-BY-4.0). Reconstruct via [benchmark/cut_neptel_segments.py](benchmark/cut_neptel_segments.py) after obtaining access. See [REPRODUCE.md](REPRODUCE.md).
- **References:** drafted by Google Chirp-2, human-reviewed. Known circularity, flagged by the authors: systems trained on Chirp-2 pseudo-labels share error patterns with these references, which can inflate their apparent agreement. This is why we always pair NepTel with an independent metric.
- **Scorer:** word-level Levenshtein on normalized text (`nepali_normalize`), with a 6 words/sec speaking-rate gate that drops hallucinated references. See [eval/score.py](eval/score.py) for a standalone re-implementation that calls the official normalizer.
- **Caveat:** the authors report absolute levels shift by several points with call mix. Treat the direction as the finding, not the exact decimal.

Scores in this repo:

| System | NepTel WER |
|---|---:|
| Kriti (base) | 40.75 |
| **Kriti Telephony** | **32.66** |

## General Nepali (the honesty check)

- **What:** 200 held-out clips of clean read Nepali from OpenSLR-54, with human reference transcripts (not model-generated). Independent of NepTel and of any ASR model's label style.
- **Why it exists:** to catch benchmark-overfitting. A model can be tuned to win telephony while silently getting worse at transcribing Nepali generally. This metric makes that visible, and it is why the general-Nepali anchor is part of the shipped training recipe.

Scores in this repo:

| System | General-Nepali WER |
|---|---:|
| Kriti (base) | 4.13 |
| **Kriti Telephony** | **6.28** |

## Neutral benchmark (in progress)

Because NepTel's references share a Chirp-2 lineage across the systems on it, the fully unassailable comparison uses references derived from no model at all.

Plan: hand-transcribe about 30 minutes of fresh Nepali call audio (independent transcribers, no ASR pre-labeling), publish the audio manifest, references, and per-system outputs, and score every model on it. Win or lose, the result gets published here. This is a benchmark whose references cannot favor anyone.

## Reporting rule

Always report the pair (NepTel, then General). A telephony number without its general-Nepali companion is incomplete and, on its own, potentially misleading. Every table in this repo follows this rule; downstream users should too.
