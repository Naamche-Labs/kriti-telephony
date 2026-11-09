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
| Kriti (base) | 40.51 |
| **Kriti Telephony** | **32.38** |

### NepTel 0.2

[NepTel 0.2](https://github.com/Ampixa/rani-asr/tree/master/benchmark) replaces the Chirp-2-drafted references with references corrected by a human against the audio (78 segments, 2,647 words, 8 kHz). Scored with the benchmark's own `eval/score_v02.py`.

| System | Decoding | NepTel 0.2 WER |
|---|---|---:|
| **Kriti Telephony v2** | beam search, width 4 | **32.49** |
| Kriti Telephony v2 | greedy | 33.09 |
| Rani (Ampixa Labs), published output | greedy, no language model | 33.51 |
| Kriti Telephony v1 | greedy | 37.89 |

Output files, re-scorable on CPU with no model:

- `benchmark/outputs/kriti-telephony-v2.beam4.neptel-0.2.json` (32.49)
- `benchmark/outputs/kriti-telephony-v2.greedy.neptel-0.2.json` (33.09)
- `benchmark/outputs/kriti-telephony.neptel-0.2.json` (37.89, v1 released checkpoint, decoded by the NepTel authors)

```bash
python eval/score_v02.py --hyp benchmark/outputs/kriti-telephony-v2.beam4.neptel-0.2.json
```

Notes, stated plainly:

- The v2 numbers were decoded on the AI4Bharat NeMo fork (`ENVIRONMENT.md`), the runtime Kriti is trained in. Beam search is NeMo's `maes` with beam size 4 and no language model (`train/v2/decode_lm.py`). Decoding the same checkpoint in mainline NeMo 3.0 after reducing it to a plain Nepali RNNT gives 33.62 greedy and 33.47 beam; small numerical differences between the two runtimes move about half a point on this 78-segment set.
- Paired segment bootstrap, Kriti Telephony v2 (beam) minus Rani: -1.0 points, 95% interval (-3.0, +1.0). Kriti v2 has the lower error, and the interval still includes zero.
- On the 56 real customer-support calls v2 scores 30.28 (Rani 30.39); on the 22 voice-assistant segments it scores 36.85 (Rani 39.66).

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
