# Reproduce / Authorized Replay

Three levels of verification, from "no GPU, five minutes" to "retrain from scratch." Every number in [`RESULTS.md`](RESULTS.md) is reachable from here.

---

## Level 1, Re-score the committed outputs (no GPU, no model, ~5 min)

This verifies our *reported numbers* against the *official scorer* using the per-segment hypotheses committed in this repo. It cannot be faked, the outputs are fixed text; the scorer is the NepTel authors'.

```bash
# one-time: get the official scorer + references
git clone https://github.com/Ampixa/nepaliconformer ../nepaliconformer

# score any released model's NepTel hypotheses
python eval/score.py --hyp benchmark/outputs/kriti-telephony.neptel.json
#   -> WER 31.32
python eval/score.py --hyp benchmark/outputs/kriti-baseline.neptel.json
#   -> WER 40.84   (NepaliConformer's published Kriti output)
python eval/score.py --hyp benchmark/outputs/kriti-baseline-redecode.neptel.json
#   -> WER 40.75   (our independent re-decode; both match Kriti's published 40.6)
python eval/score.py --hyp benchmark/outputs/nepaliconformer.neptel.json
#   -> WER 34.09
python eval/score.py --hyp benchmark/outputs/kriti-distilled-5.9k.neptel.json
#   -> WER 33.75   (an intermediate distillation checkpoint)
```

If these reproduce, the leaderboard claims are verified. The remaining levels verify that the *outputs themselves* come from the models we say they do.

---

## Level 2, Re-decode from the checkpoint (GPU)

This regenerates the hypotheses from the released checkpoint, proving the outputs aren't hand-edited.

**Environment.** Kriti is a NeMo 1.23-era model on the AI4Bharat NeMo fork. The exact pinned environment (this was the hardest part of the whole project) is in [`ENVIRONMENT.md`](ENVIRONMENT.md). Key pins: AI4Bharat NeMo fork + `torch 2.13`; numba CUDA via `NUMBA_CUDA_USE_NVIDIA_BINDING=1` with `cuda-python==12.2.1` and `numba==0.59.1` (required on Hopper/H100, see `ENVIRONMENT.md`); `precision=32-true` for the numba RNNT loss.

**Checkpoints.** Released separately (they are ~500 MB each): `kriti.nemo` (base) and `kriti_telephony.nemo`. See the Releases page / HF; SHA-256 hashes in [`evidence/results.json`](evidence/results.json).

```bash
# NepTel + general Nepali in one shot
NUMBA_CUDA_USE_NVIDIA_BINDING=1 python eval/eval_both.py kriti_telephony.nemo kriti-telephony
#   -> NepTel=31.32  GeneralNepali=9.92

# base model, same harness
NUMBA_CUDA_USE_NVIDIA_BINDING=1 python eval/eval_both.py kriti.nemo base
#   -> NepTel=40.75  GeneralNepali=4.13
```

`eval_both.py` is the honesty instrument, it reports both axes together so a model can't hide a general-Nepali regression.

---

## Level 3, Rebuild the NepTel audio and retrain from scratch

For full end-to-end reproduction.

### 3a. Reconstruct the NepTel evaluation audio

NepTel is cut from the **InfoBayAI** dual-channel Nepali call-center dataset (gated, CC-BY-4.0). We do not redistribute it.

```bash
# 1. Request access (one click, usually auto-approved):
#    https://huggingface.co/datasets/InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel
# 2. Cut the 77 segments deterministically (silence-aware ~25s windows,
#    validated against each segment's published duration):
python benchmark/cut_neptel_segments.py   # writes neptel_audio/seg_*.wav
```

The cutter self-validates: all 77 segment durations must match `references.json` to within 1 s (they do). This is what lets an independent party regenerate byte-comparable eval audio.

### 3b. Retrain Kriti Telephony

```bash
# 1. Teacher pseudo-labels conversational audio in the reference style
python train/pseudo_label_all.py     # podcast  -> distill_big_*.json
python train/pseudo_label.py         # code-switched -> combined into distill_mega_*.json

# 2. Fine-tune with the general-Nepali anchor (the shipped recipe)
NUMBA_CUDA_USE_NVIDIA_BINDING=1 python train/train_kriti_telephony.py
#   -> kriti_telephony.nemo

# 3. Validate on both axes
NUMBA_CUDA_USE_NVIDIA_BINDING=1 python eval/eval_both.py kriti_telephony.nemo kriti-telephony
```

Training data (all public, credited):
- **Conversational (teacher-labeled):** `Bijay13/nepali-podcast-code-switch-asr-dataset`, `Shyyamsh/nepali-english-codemixed-asr`
- **General anchor (human labels):** OpenSLR-54 (`openslr.org/54`)
- **Teacher:** `ampixa/nepali-conformer-offline`

---

## What "authorized replay" means here

- **The eval set stays gated.** We give you the deterministic cutter and the access link, not the audio. This preserves the benchmark's integrity (no leakage) while making replay possible for anyone who obtains the same access we did.
- **The scorer is the authors', not ours.** Level-1 verification runs *their* `nepali_normalize` on *their* `references.json`. We can't tilt the instrument.
- **Both axes, always.** Any replay that reports a NepTel number without the paired general-Nepali number is incomplete, that pairing is the whole point.

If a step doesn't reproduce, open an issue with your environment (`ENVIRONMENT.md` versions) and the command, that's exactly the kind of scrutiny this repo is built to invite.
