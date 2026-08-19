# Benchmarks

Two evaluation sets are used throughout this repo. Reporting **both** together is the core methodological commitment.

---

## NepTel (telephony — the headline metric)

- **What:** real, spontaneous Nepali call-center conversation — the hardest real-world Nepali audio with public references. 3 calls / 1 vendor / 1 domain; ~77 segments, ~75–76 scored after gating, ~2,375 reference words.
- **Author:** the [NepaliConformer team (Ampixa)](https://github.com/Ampixa/nepaliconformer). This is *their* benchmark; we reproduce and report on it, using their scorer.
- **Source audio:** cut from the gated **InfoBayAI** dual-channel Nepali call-center dataset (CC-BY-4.0). Reconstruct via [`benchmark/cut_neptel_segments.py`](benchmark/cut_neptel_segments.py) after obtaining access — see [`REPRODUCE.md`](REPRODUCE.md).
- **References:** drafted by Google Chirp-2, human-reviewed. **Known circularity (flagged by the authors):** systems trained on Chirp-2 pseudo-labels share error patterns with these references, which can inflate their apparent agreement. This is why we always pair NepTel with an independent metric.
- **Scorer:** word-level Levenshtein on normalized text (`nepali_normalize`), with a >6 words/sec speaking-rate gate that drops hallucinated references. See [`eval/score.py`](eval/score.py) for a standalone re-implementation that calls the official normalizer.
- **Caveat:** the authors report absolute levels shift ±5–8 points with call mix. **Treat the ranking/lead as the finding, not the exact decimal.**

**Scores (this repo):**

| System | NepTel WER |
|---|---:|
| Kriti (base) | 40.75 |
| NepaliConformer offline | 34.09 |
| **Kriti Telephony** | **31.32** |

---

## General Nepali (the honesty check)

- **What:** 200 held-out clips of clean read Nepali from **OpenSLR-54**, with **human** reference transcripts (not model-generated). Independent of NepTel and of any ASR model's label style.
- **Why it exists:** to catch benchmark-overfitting. A model can be tuned to win NepTel while silently getting *worse* at transcribing Nepali. This metric makes that visible.
- **What it caught:** the pure-distillation model scored a tempting 32.96 on NepTel — but 16.53 here, a 4× regression from baseline's 4.13. That model was rejected. Kriti Telephony holds at 9.92.

**Scores (this repo):**

| System | General-Nepali WER |
|---|---:|
| Kriti (base) | 4.13 |
| Pure distillation (rejected) | 16.53 |
| **Kriti Telephony** | **9.92** |

---

## Neutral benchmark (in progress)

Because NepTel's references share a Chirp-2 lineage with the systems being compared, the *fully* unassailable comparison uses references derived from **neither** model.

**Plan:** hand-transcribe ~30 minutes of fresh Nepali call audio (independent transcribers, no ASR pre-labeling), publish the audio manifest + references + per-system outputs, and score every model — Kriti, Kriti Telephony, NepaliConformer — on it. Win or lose, the result gets published here.

This is the honest endpoint of the "did you game it?" conversation: a benchmark whose references cannot favor anyone.

---

## Reporting rule

**Always report the pair (NepTel / General).** A telephony number without its general-Nepali companion is incomplete and, on its own, potentially misleading. Every table in this repo follows this rule; downstream users should too.
