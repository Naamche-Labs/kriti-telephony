# Kriti Telephony

**Nepali speech recognition tuned for the hardest audio there is: live, spontaneous phone calls.**

Most Nepali speech technology is built and measured on read-aloud recordings. Real Nepali on a phone line (call-center conversation, code-switching, narrowband audio) is a different problem, and models that top read-speech benchmarks tend to fall apart on it. Kriti Telephony is a domain-adapted version of [Kriti](https://github.com/Naamche-Labs/kriti) built specifically for that condition.

Every number in this repository is reproducible: per-segment outputs, the scorer, the training recipe, and a second independent metric are all included so any claim here can be checked and, if wrong, falsified.

## Results

All numbers are word error rate (WER), lower is better. Telephony is scored on the public NepTel set with its official scorer (word-level Levenshtein on normalized text). "General Nepali" is a held-out set of clean read Nepali with human references, reported next to every telephony number as a check against benchmark overfitting.

| Model | Params | NepTel WER ↓ | General Nepali WER ↓ |
|---|---|---:|---:|
| [Kriti](https://github.com/Naamche-Labs/kriti) (base) | 119 M | 40.8 | **4.1** |
| **Kriti Telephony** (this repo) | 119 M | **32.7** | 6.3 |

Kriti Telephony cuts telephony WER by a fifth (40.8 to 32.7) while keeping general Nepali strong (6.3, close to the base model's 4.1). Our reproduction of the base model reads 40.75 on NepTel against its published 40.6, a match to rounding, which confirms the evaluation harness is sound.

Read every result as a pair (telephony, then general). A telephony number on its own can hide a general-Nepali regression, so we never report one without the other.

## Two models, one for each job

|  | [**Kriti**](https://github.com/Naamche-Labs/kriti) | **Kriti Telephony** |
|---|---|---|
| Best for | clean, read, studio audio | phone calls, IVR, call-center, spontaneous speech |
| Clean-read WER | **4.1** | 6.3 |
| NepTel (telephony) WER | 40.8 | **32.7** |
| Relationship | the base model | telephony-tuned, still strong on clean audio |

Kriti Telephony is tuned for phone audio but holds up on clean audio too (6.3 vs the base model's 4.1). For pure studio or read-speech work the base Kriti still has an edge; for anything involving calls, use Kriti Telephony. Always report the WER pair.

## How it was built

Full detail lives in [METHODOLOGY.md](METHODOLOGY.md). In brief:

1. **Diagnosis.** Base Kriti's NepTel errors are dominated by style (reference orthography conventions and English-loanword transliteration) rather than by mishearing. Ordinary fine-tuning failed repeatedly because every available training set carried a different label style and pulled the model the wrong way.
2. **Style-matched supervision.** We used an existing strong Nepali model as a teacher to relabel roughly 12,000 clips of conversational Nepali audio in the target reference style, giving Kriti hard conversational audio paired with correctly-styled transcripts.
3. **General-Nepali anchor.** We mixed in a large amount of general clean Nepali with human labels so the model keeps its real transcription ability instead of collapsing into a benchmark-only system. This anchor is what keeps general-Nepali WER at 6.3, close to the base model.
4. **Full fine-tune, validated on two axes.** NepTel and held-out general Nepali, so the gain is demonstrably domain adaptation rather than leaderboard fitting.

We did not train on the NepTel evaluation audio or references. The benchmark set carries a canary marker and was held out completely. You can confirm this from the training manifests referenced in [train/](train/), which use only podcast, code-switched, and OpenSLR-54 audio, never the call recordings NepTel is cut from.

## Verify it yourself

Full instructions in [REPRODUCE.md](REPRODUCE.md). Three levels, from no-GPU to full retrain:

```bash
# Level 1: re-score the committed outputs against the official scorer (no GPU)
python eval/score.py --hyp benchmark/outputs/kriti-telephony.neptel.json
#   WER 32.66

# Level 2: re-decode from the checkpoint (GPU): NepTel and general Nepali together
python eval/eval_both.py kriti_telephony.nemo kriti-telephony
#   NepTel=32.66  GeneralNepali=6.28
```

Per-segment hypotheses for every model in the results table are committed under [benchmark/outputs/](benchmark/outputs/). You can re-score them on CPU with no model and get the exact numbers above.

## Model checkpoint

| Model | HuggingFace |
|---|---|
| Kriti Telephony | [Aarjan/kriti-telephony](https://huggingface.co/Aarjan/kriti-telephony) |
| Kriti (base) | [Naamche Labs / Kriti](https://github.com/Naamche-Labs/kriti) |

```python
from huggingface_hub import hf_hub_download
import nemo.collections.asr as nemo_asr
path = hf_hub_download("Aarjan/kriti-telephony", "kriti_telephony.nemo")
model = nemo_asr.models.ASRModel.restore_from(path, strict=False)
model.cur_decoder = "rnnt"
print(model.transcribe(["call.wav"], language_id="ne"))
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for the exact package versions. Kriti runs on the AI4Bharat NeMo fork and needs `NUMBA_CUDA_USE_NVIDIA_BINDING=1` on Hopper GPUs.

## Repository layout

```
kriti-telephony/
  README.md            you are here
  METHODOLOGY.md       full pipeline: diagnosis, distillation, anchor, validation
  RESULTS.md           every experiment on both axes, with commentary
  REPRODUCE.md         three-level authorized replay (data access and commands)
  BENCHMARKS.md        NepTel and general-Nepali definitions, neutral-set plan
  ENVIRONMENT.md       exact pins and the H100 gotchas
  LICENSE              MIT for our code; third-party assets credited
  eval/                scorer, NepTel harness, dual-axis harness, ortho diagnostic
  train/               teacher relabeling and the shipped training recipe
  benchmark/           NepTel segment cutter and per-segment outputs (evidence)
  evidence/            machine-readable results and checkpoint hashes
```

## Credits and lineage

- **Kriti (base)** by Naamche Labs, a 119 M hybrid RNNT-CTC Conformer built on [AI4Bharat IndicConformer](https://github.com/AI4Bharat/NeMo), with a Nepali head and a Devanagari-danda punctuation head. Kriti Telephony adapts this model. Repository: [Naamche-Labs/kriti](https://github.com/Naamche-Labs/kriti).
- **NepTel benchmark and scorer** by [Ampixa](https://github.com/Ampixa/nepaliconformer), used and cited per their submission terms.
- **Distillation teacher:** [ampixa/nepali-conformer-offline](https://huggingface.co/ampixa/nepali-conformer-offline), disclosed in [METHODOLOGY.md](METHODOLOGY.md).
- **Source audio:** NepTel is cut from the [InfoBayAI Nepali Call-Center dataset](https://huggingface.co/datasets/InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel) (gated, CC-BY-4.0). We do not redistribute it. See [REPRODUCE.md](REPRODUCE.md) for access and segment reconstruction.
- **Training audio:** [Bijay13 podcast](https://huggingface.co/datasets/Bijay13/nepali-podcast-code-switch-asr-dataset), [Shyyamsh code-switched](https://huggingface.co/datasets/Shyyamsh/nepali-english-codemixed-asr), and [OpenSLR-54](https://openslr.org/54).

## License

Our code and documentation are MIT (see [LICENSE](LICENSE)). Third-party datasets, benchmarks, and models retain their own licenses as credited above.

Naamche Labs. Kriti, honest ASR for Nepali. Report WER pairs (telephony, general), never telephony alone.
