# Methodology

How Kriti Telephony was actually built — including the failures, because they are the most informative part.

---

## 0. Starting point: what Kriti is

Kriti is Naamche Labs' Nepali ASR model: a **119M-parameter hybrid RNNT-CTC Conformer** (`EncDecHybridRNNTCTCBPEModel` in NVIDIA NeMo), built on **AI4Bharat's IndicConformer**. It is multilingual under the hood — a shared Conformer encoder with a **multi-softmax joint** carrying per-language output heads for 22 Indic languages — with a Nepali (`ne`) head and a Devanagari-danda punctuation head. Inference for Nepali reduces the model to the `ne` head (256 subword tokens + blank).

**Baseline behaviour (measured, this repo):**

| | WER |
|---|---:|
| Clean read Nepali (OpenSLR-54, human labels) | **4.13** |
| NepTel (real call-center telephony) | **40.75** |

So Kriti is genuinely strong on clean audio and struggles on spontaneous narrowband telephony — a ~10× WER gap. Closing that gap on NepTel, without wrecking the 4.13, is the entire problem.

---

## 1. Diagnosis — the errors are *style*, not *hearing*

Before training anything, we analyzed **where** baseline Kriti's NepTel errors come from ([`eval/ortho_analysis.py`](eval/ortho_analysis.py)). Two findings dominated:

1. **Orthographic style.** A large share of "errors" are spelling-convention differences the scorer counts as wrong: missing nasalization marks (`आउछ` vs reference `आउँछ`), long-vs-short vowels (`कामहरु` vs `कामहरू`), and colloquial variants (`हैन` vs `होइन`). Normalizing these away closed ~2.4 points of the gap — i.e. a meaningful fraction is *convention*, not recognition.
2. **English-loanword transliteration.** Telecom speech is code-switched ("voice pack", "unlimited"). The reference style transliterates to Devanagari in a specific way; Kriti's differs.

**Why this matters:** NepTel's references were drafted by Google Chirp-2 and human-reviewed. The prior best system (NepaliConformer) trained on Chirp-2 pseudo-labels — so its output *style* already matches the references. Part of the leaderboard gap is this shared-style circularity, which the NepTel authors themselves flag. The lever, then, is to make Kriti **output the reference style** on conversational audio — without forgetting how to transcribe Nepali generally.

---

## 2. Why naive fine-tuning failed (five times)

We document every failed attempt in [`RESULTS.md`](RESULTS.md) because they rule out the obvious explanations:

| Attempt | Data | NepTel WER | Verdict |
|---|---|---:|---|
| Telephony aug | OpenSLR-54, simulated telephony | 43.44 | worse |
| Conversational | podcast, telephonified | 41.13 | worse |
| Clean conversational | podcast, clean | 46.88 | worse (overfit) |
| Frozen encoder, podcast | podcast, human labels | 42.01 | worse |
| Frozen encoder, clean read | OpenSLR-54, human labels | 40.71 | worse |

A key control: the clean-read fine-tune (row 5) did **not** even improve its *own* domain (baseline and fine-tuned both scored 4.13 on OpenSLR val) — because Kriti is already saturated there — yet it still degraded NepTel. Fine-tuning on data where the model is already perfect provides no signal but drifts it off the hard domain.

**Root cause:** every one of these training sets had labels in a *different style* than the NepTel references (podcast keeps English in Latin script; OpenSLR uses formal read-speech conventions). Training pulled Kriti toward the wrong style. The problem was never the model or the recipe — it was **label-style mismatch**.

---

## 3. The fix — knowledge distillation for style-matched supervision

Once the diagnosis was "we need hard conversational audio with *reference-style* labels," the method follows:

1. **Teacher.** [NepaliConformer offline](https://huggingface.co/ampixa/nepali-conformer-offline) reproduces to 35.6 WER on NepTel in our harness and, crucially, outputs the reference style natively.
2. **Pseudo-label conversational audio.** We ran the teacher over conversational Nepali audio — ~6.1k podcast clips + ~5.9k code-switched clips = **~12k clips** — producing transcripts in the target style ([`train/pseudo_label_all.py`](train/pseudo_label_all.py)).
3. **Distill into Kriti.** Fine-tuning Kriti on these style-matched conversational labels *worked immediately*, and scaled cleanly with data:

   | Distillation clips | NepTel WER |
   |---:|---:|
   | 2.8k (frozen) | 36.48 |
   | 5.9k (frozen) | 34.93 |
   | 5.9k (full FT) | 33.75 |
   | 12k (full FT) | **32.96** |

**But — see the trap.** The 12k full-FT model hit 32.96 on NepTel and *appeared* to win. Then we checked general Nepali: **16.53 WER**, a 4× regression from baseline's 4.13. It had become a NepTel-style impersonator that could no longer transcribe clean Nepali. **This model is a failure, not a win.** It is kept in this repo ([`benchmark/outputs/kriti-distilled-puredistill.neptel.json`](benchmark/outputs/kriti-distilled-puredistill.neptel.json)) as the reference example of what benchmark-gaming looks like.

---

## 4. The shipped recipe — distillation **+** anti-forgetting anchor

The fix for catastrophic forgetting is standard and effective: mix general-domain data with human labels back into training so the model can't collapse onto one style.

**Kriti Telephony training mix ([`train/train_kriti_telephony.py`](train/train_kriti_telephony.py)):**
- ~12k conversational clips with **teacher (style-matched) labels** — teaches telephony + reference style
- ~8k clean Nepali clips (OpenSLR-54) with **human labels** — anchors general ability
- Full fine-tune, lr 1e-5, 3 epochs, fp32, gradient clipping.

**Result, validated on both axes ([`eval/eval_both.py`](eval/eval_both.py)):**

| | NepTel WER | General-Nepali WER |
|---|---:|---:|
| Pure distillation (no anchor) | 32.96 | 16.53 |
| **Kriti Telephony (with anchor)** | **31.32** | **9.92** |

The anchor did two things: it took general Nepali from 16.5 back to **9.9** (functional again), **and** it *improved* NepTel from 32.96 to 31.32 — because the general data reduced overfitting to the teacher's own noise, improving generalization on both axes at once.

---

## 5. What we deliberately did **not** do

- We did **not** train on NepTel audio or references (canary-marked, held out).
- We did **not** ship the pure-distillation model, despite its lower-looking cost, because it fails the general-Nepali test.
- We did **not** chase sub-30 by removing the anchor — that only lowers NepTel by trading away more general-Nepali quality, which is the overfitting direction. 31.3/9.9 is the honest operating point.
- We did **not** apply test-set-specific post-processing to the shipped number. (An orthographic canonicalizer helps the *baseline* by ~1.6 pts, but *hurts* Kriti Telephony — the distilled model already outputs the correct style — so it is not used.)

---

## 6. Honest limitations

- **Kriti Telephony is a domain sibling, not an upgrade.** Base Kriti is better on clean audio (4.1 vs 9.9).
- **NepTel is one benchmark**, 3 calls / 1 vendor / 1 domain; its authors note absolute levels move ±5–8 points with call mix. Treat the *lead* as the finding, not the exact decimal.
- **Style circularity is real.** Because NepTel references and the distillation teacher share a Chirp-2 lineage, some of the NepTel lead reflects style-matching. The general-Nepali number is reported precisely to bound this, and a neutral benchmark (references independent of any model) is the planned next validation — see [`BENCHMARKS.md`](BENCHMARKS.md).
