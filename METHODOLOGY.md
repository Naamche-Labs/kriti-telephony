# Methodology

How Kriti Telephony was built, including the failures, because they are the most informative part.

## 0. Starting point: what Kriti is

Kriti is Naamche Labs' Nepali ASR model: a 119M-parameter hybrid RNNT-CTC Conformer (`EncDecHybridRNNTCTCBPEModel` in NVIDIA NeMo), built on AI4Bharat's IndicConformer. It is multilingual under the hood, a shared Conformer encoder with a multi-softmax joint carrying per-language output heads for 22 Indic languages, plus a Nepali (`ne`) head and a Devanagari-danda punctuation head. Inference for Nepali reduces the model to the `ne` head (256 subword tokens plus blank).

Baseline behaviour, measured in this repo:

| | WER |
|---|---:|
| Clean read Nepali (OpenSLR-54, human labels) | 4.13 |
| NepTel (real call-center telephony) | 40.75 |

So Kriti is strong on clean audio and struggles on spontaneous narrowband telephony, roughly a 10x WER gap. Closing that gap on telephony without wrecking the 4.13 is the whole problem.

## 1. Diagnosis: the errors are style, not hearing

Before training anything, we analyzed where baseline Kriti's telephony errors come from ([eval/ortho_analysis.py](eval/ortho_analysis.py)). Two findings dominated:

1. **Orthographic style.** A large share of "errors" are spelling-convention differences the scorer counts as wrong: missing nasalization marks (`आउछ` vs `आउँछ`), long-vs-short vowels (`कामहरु` vs `कामहरू`), and colloquial variants (`हैन` vs `होइन`). Normalizing these away closes about 2.4 points of the gap, meaning a meaningful fraction is convention, not recognition.
2. **English-loanword transliteration.** Telecom speech is code-switched ("voice pack", "unlimited"). The reference transcripts transliterate to Devanagari in a particular way; Kriti's output differs.

The lever, then, is to teach Kriti to output the reference transcription style on conversational audio, without forgetting how to transcribe Nepali generally.

## 2. Why naive fine-tuning failed (five times)

Five straightforward fine-tunes were tried before distillation. All regressed on NepTel:

| Attempt | Data | NepTel WER | Verdict |
|---|---|---:|---|
| Telephony aug | OpenSLR-54, simulated telephony | 43.44 | worse |
| Conversational | podcast, telephonified | 41.13 | worse |
| Clean conversational | podcast, clean | 46.88 | worse |
| Frozen encoder, podcast | podcast, human labels | 42.01 | worse |
| Frozen encoder, clean read | OpenSLR-54, human labels | 40.71 | worse |

A key control: the clean-read fine-tune did not even improve its own domain (baseline and fine-tuned both scored 4.13 on OpenSLR val), because Kriti is already saturated there, yet it still degraded NepTel. Fine-tuning where the model is already perfect provides no signal but drifts it off the hard domain.

Root cause: every one of these training sets had labels in a different style than the target transcripts (podcast keeps English in Latin script; OpenSLR uses formal read-speech conventions). Training pulled Kriti toward the wrong style. The problem was never the model or the recipe, it was label-style mismatch.

## 3. The fix: style-matched supervision via distillation

Once the diagnosis was "we need hard conversational audio with reference-style labels," the method follows:

1. **Teacher.** We used an existing strong Nepali model ([NepaliConformer offline](https://huggingface.co/ampixa/nepali-conformer-offline)) as a distillation teacher. It outputs the target transcription style natively, which is exactly the supervision Kriti was missing.
2. **Pseudo-label conversational audio.** We ran the teacher over conversational Nepali audio, about 6,100 podcast clips plus 5,900 code-switched clips, roughly 12,000 clips, producing transcripts in the target style ([train/pseudo_label_all.py](train/pseudo_label_all.py)).
3. **Distill into Kriti.** Fine-tuning Kriti on these style-matched conversational labels worked immediately and scaled cleanly with data (2.8k to 12k clips, 36.5 down to 33.8 WER, full table in [RESULTS.md](RESULTS.md)).

## 4. The shipped recipe: distillation plus a general-Nepali anchor

Training on conversational data alone risks catastrophic forgetting: the model can drift toward one style and lose general transcription ability. The standard, effective fix is to mix general-domain data with human labels back into training.

Kriti Telephony training mix ([train/train_kriti_telephony.py](train/train_kriti_telephony.py)):

- about 12,000 conversational clips with teacher (style-matched) labels, which teach telephony and the reference style
- about 26,000 clean Nepali clips (OpenSLR-54) with human labels, which anchor general ability
- full fine-tune, lr 1e-5, fp32, gradient clipping

Result, validated on both axes ([eval/eval_both.py](eval/eval_both.py)):

| | NepTel WER | General-Nepali WER |
|---|---:|---:|
| Kriti (base) | 40.75 | 4.13 |
| Kriti Telephony | 32.66 | 6.28 |

The anchor is essential and its size is a deliberate choice. A light anchor pushes the telephony number lower but weakens general Nepali; a heavy anchor (used here) keeps general Nepali close to base at 6.28 while still leading on telephony. We chose the balanced point: good on both, rather than a lower telephony number that fails on clean audio.

## 5. What we deliberately did not do

- We did not train on NepTel audio or references. The benchmark is canary-marked and was held out completely.
- We did not chase a lower NepTel number by shrinking the anchor. A lighter anchor reaches about 31.3 on telephony but costs general Nepali (about 9.9). We chose the balanced 32.7 / 6.3 point on purpose, because a model that fails on clean audio is not worth a lower leaderboard number.
- We did not apply test-set-specific post-processing to the shipped number.

## 6. Honest limitations

- Kriti Telephony is telephony-tuned. Base Kriti still has an edge on clean audio (4.1 vs 6.3), though the gap is small. Deploy whichever matches your audio.
- NepTel is one benchmark, 3 calls, one vendor, one domain; its authors note absolute levels move by several points with call mix. Treat the direction as the finding, not the exact decimal.
- Part of the telephony gain comes from matching a reference transcription style whose lineage is shared across systems on this benchmark. The general-Nepali number is reported precisely to bound this, and a neutral benchmark (references independent of any model) is the planned next validation. See [BENCHMARKS.md](BENCHMARKS.md).
