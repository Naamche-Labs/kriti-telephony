# Kriti Telephony v2

**Nepali speech recognition for real phone calls that stays strong on clean speech.**

Most Nepali speech technology is built and measured on read-aloud recordings. Real Nepali on a phone line (call-center conversation, code-switching, narrowband audio) is a different problem. Kriti Telephony is a 119M-parameter model by Naamche Labs built for that condition. Version 2 also brings clean-speech accuracy back to the level of the base model, so one model now does both.

Every number here is reproducible: per-segment outputs, the scorer and the full training recipe are in this repository. v1 is preserved at the [`v1` tag](https://github.com/Naamche-Labs/kriti-telephony/tree/v1).

## Results

Word error rate (WER), lower is better.

**NepTel 0.2**: real 8 kHz Nepali phone audio (78 segments, 2,647 words), references corrected by a human against the audio, scored with the benchmark's own [`score_v02.py`](https://github.com/Ampixa/rani-asr/tree/master/eval).

| System | Decoding | NepTel 0.2 WER |
|---|---|---:|
| **Kriti Telephony v2** | beam search, width 4 | **32.49** |
| Kriti Telephony v2 | greedy | 33.09 |
| Rani (Ampixa Labs), published output | greedy, no language model | 33.51 |
| Kriti Telephony v1 | greedy | 37.89 |

**FLEURS Nepali test**: 726 clean read sentences with human references, not used in training.

| Model | WER | CER |
|---|---:|---:|
| Kriti (base) | 24.33 | 8.28 |
| Kriti Telephony v1 | 31.53 | 10.54 |
| **Kriti Telephony v2** | 24.77 | 8.81 |

v1 traded clean-speech accuracy for phone accuracy. v2 keeps the phone gain and is back to base-model level on clean speech. Read every result as a pair (phone, then clean).

**Speed**: one CPU thread, Apple M5, batch 1, greedy decoding, mainline NeMo 3.0, on the 25 NepTel clips (622.72 s) that Ampixa uses for its speed table: 40.6 times real time, first text for a 10 s clip in 0.24 s.

Details and caveats, stated plainly:

- The v2 NepTel numbers are decoded on the AI4Bharat NeMo fork, the runtime Kriti is trained in. Beam search is NeMo `maes`, width 4, no language model ([`train/v2/decode_lm.py`](train/v2/decode_lm.py)).
- The same weights reduced to a plain Nepali RNNT on mainline NeMo 3.0 give 33.62 greedy and 33.47 beam. The two runtimes differ by about half a point on this small set.
- Paired segment bootstrap, v2 (beam) minus Rani: -1.0 points, 95% interval (-3.0, +1.0). v2 has the lower error, and the interval still includes zero.
- By part of NepTel: 56 customer-support calls, v2 30.28 vs Rani 30.39; 22 voice-assistant segments, v2 36.85 vs Rani 39.66.

## Verify it yourself

Re-score the committed outputs with the NepTel scorer, no GPU and no model needed:

```bash
git clone https://github.com/Ampixa/rani-asr
python rani-asr/eval/score_v02.py --hyp benchmark/outputs/kriti-telephony-v2.beam4.neptel-0.2.json   # 32.49
python rani-asr/eval/score_v02.py --hyp benchmark/outputs/kriti-telephony-v2.greedy.neptel-0.2.json  # 33.09
```

Re-decode from the checkpoint on a GPU with the AI4Bharat fork ([ENVIRONMENT.md](ENVIRONMENT.md)):

```bash
python train/v2/decode_lm.py kriti_telephony.nemo --set neptel --alphas 0 --beam 4   # 32.49
```

## How v2 was built

Starting from Kriti Telephony v1, full fine-tune in two rounds. Recipe: [`train/v2/`](train/v2/).

1. **Round 1, real conversational Nepali with human labels.** IndicVoices Nepali: 458 h from 1,076 speakers, standard-spelling transcripts. Half of the clips pass through a phone-line simulation: 300 to 3400 Hz band, 8 kHz, one of GSM, G.711 mu-law or A-law, G.726 or narrowband Opus ([`prep_tel.py`](train/v2/prep_tel.py)). The v1 training mix is kept so the model keeps its call-center style.
2. **Round 2, more speech, cleaner schedule.** About 36 h of extra audio: Nepali parliament speech with its human sentences and Nepali YouTube (YODAS2), labelled by the round-1 model and kept only where labels agreed ([`plabel2.py`](train/v2/plabel2.py)), 70% phone-simulated. Cosine learning-rate decay from 2e-5 to 1e-6.

Checkpoints were chosen on a phone-simulated IndicVoices validation set, never on NepTel. NepTel audio and references were not used for training or selection. No Rani weights or Rani outputs were used.

What did not work, for the record: training on IndicVoices verbatim (spoken-style) transcripts made NepTel worse (35.47), and a word-level language model trained on IndicVoices text hurt on phone calls.

## Model checkpoint

| File on [Aarjan/kriti-telephony](https://huggingface.co/Aarjan/kriti-telephony) | Runtime |
|---|---|
| `kriti_telephony.nemo` (v2) | AI4Bharat NeMo fork, behind every number above |
| `kriti_telephony_v2_nemo3.nemo` | mainline NeMo 3.0, plain Nepali RNNT ([`convert_to_nemo3.py`](train/v2/convert_to_nemo3.py)) |

```python
from huggingface_hub import hf_hub_download
import nemo.collections.asr as nemo_asr
path = hf_hub_download("Aarjan/kriti-telephony", "kriti_telephony.nemo")
model = nemo_asr.models.ASRModel.restore_from(path, strict=False)
model.cur_decoder = "rnnt"
print(model.transcribe(["call.wav"], language_id="ne"))
```

On mainline NeMo 3.0: download `kriti_telephony_v2_nemo3.nemo`, then `nemo_asr.models.ASRModel.restore_from(path).transcribe(["call.wav"])`.

## Repository layout

```
kriti-telephony/
  README.md            you are here
  METHODOLOGY.md       full pipeline: diagnosis, distillation, anchor, validation
  RESULTS.md           every experiment on both axes, with commentary
  REPRODUCE.md         three-level authorized replay (data access and commands)
  VERIFICATION.md      independent re-verification of the shipped model
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
- **NepTel benchmark and scorer** by [Ampixa Labs](https://github.com/Ampixa/rani-asr), used and cited per their submission terms.
- **Distillation teacher:** [ampixa/nepali-conformer-offline](https://huggingface.co/ampixa/nepali-conformer-offline), disclosed in [METHODOLOGY.md](METHODOLOGY.md).
- **Source audio:** NepTel is cut from the [InfoBayAI Nepali Call-Center dataset](https://huggingface.co/datasets/InfoBayAI/Nepali_Call_Center_Audio_Dataset_Dual_Channel) (gated, CC-BY-4.0). We do not redistribute it. See [REPRODUCE.md](REPRODUCE.md) for access and segment reconstruction.
- **v2 training audio:** [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) Nepali (CC-BY-4.0, AI4Bharat), [YODAS2](https://huggingface.co/datasets/espnet/yodas2) `ne000` (CC-BY-3.0), [Nepali Parliament DS v2](https://huggingface.co/datasets/kiranpantha/no-filter-raw-NepaliParliamentDSv2) (CC-BY-4.0). Clean-speech evaluation: [FLEURS](https://huggingface.co/datasets/google/fleurs) Nepali test.
- **v1 training audio:** [Bijay13 podcast](https://huggingface.co/datasets/Bijay13/nepali-podcast-code-switch-asr-dataset), [Shyyamsh code-switched](https://huggingface.co/datasets/Shyyamsh/nepali-english-codemixed-asr), and [OpenSLR-54](https://openslr.org/54).

## License

Our code and documentation are MIT (see [LICENSE](LICENSE)). Third-party datasets, benchmarks, and models retain their own licenses as credited above.

Naamche Labs. Kriti, honest ASR for Nepali. Report WER pairs (telephony, general), never telephony alone.
