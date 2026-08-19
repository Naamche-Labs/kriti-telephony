# Verification

Everything in this repository is built to be checked, not trusted. This page records the independent checks we ran on the shipped model, and the exact commands anyone else can run to reproduce them.

## Independent re-verification of the shipped model

Run from scratch on the released checkpoint (`kriti_telephony.nemo`, the v2 model), against the official NepTel scorer and references:

| Check | Method | Result |
|---|---|---|
| Checkpoint identity | sha256 of the shipped file vs the HuggingFace file | matches (`d20d7aff18b7d2c8...`) |
| NepTel WER, fresh decode | re-decode all 76 scored segments and re-score, ignoring the committed output | **32.66** |
| Committed output integrity | compare the fresh decode to `benchmark/outputs/kriti-telephony.neptel.json` | byte-identical on 77 / 77 segments |
| General-Nepali WER, fresh decode | re-decode 200 held-out clean clips with human references | **6.28** |

The committed per-segment output is exactly what the model produces. No hand-editing, no stale files.

## Head-to-head on NepTel (re-score with no GPU)

Each number below comes from a committed per-segment hypothesis file scored with the benchmark's own `nepali_normalize` and `references.json`:

```bash
# one-time: get the official scorer + references
git clone https://github.com/Ampixa/nepaliconformer ../nepaliconformer

python eval/score.py --hyp benchmark/outputs/kriti-telephony.neptel.json          # 32.66  (Kriti Telephony)
python eval/score.py --hyp benchmark/outputs/nepaliconformer.neptel.json          # 34.09  (NepaliConformer offline)
python eval/score.py --hyp benchmark/outputs/kriti-baseline-redecode.neptel.json  # 40.75  (base Kriti, our re-decode)
python eval/score.py --hyp benchmark/outputs/kriti-baseline.neptel.json           # 40.84  (base Kriti, published output)
```

Base Kriti reproducing 40.75 / 40.84 against its published 40.6 is the harness sanity check: the instrument is correct before any other number is trusted.

## Checkpoint hashes

```
kriti.nemo            0144854f0cc78f4b6115b75089fad632c39207d5256e53f92da996b9bbe43582
kriti_telephony.nemo  d20d7aff18b7d2c8572cf74a9e6c3f13aa5692043d07df19d819e2523889dd2d
```

The `kriti_telephony.nemo` hash matches the file published at [huggingface.co/Aarjan/kriti-telephony](https://huggingface.co/Aarjan/kriti-telephony).

## What this does and does not prove

- It proves the shipped model scores 32.66 on NepTel, beating NepaliConformer's 34.09, using their own scorer, and that this is reproducible from the released checkpoint.
- It proves the model stays strong on general Nepali (6.28), so the telephony gain is not overfitting that breaks normal transcription.
- It does not claim the gain is free of the benchmark's known reference-style circularity (see [BENCHMARKS.md](BENCHMARKS.md)); that is why general Nepali is always reported alongside, and why a neutral benchmark is the planned next validation.
