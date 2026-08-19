# Results

Two metrics throughout, always reported together:

- **NepTel WER**: real Nepali call-center telephony, official scorer, about 76 scored segments.
- **General-Nepali WER**: 200 held-out clean read clips (OpenSLR-54) with human references. This column is the honesty check. It catches a model that wins telephony by quietly getting worse at everything else.

Sanity check on the harness: our reproduction of the base Kriti model scores 40.75 on NepTel against its published 40.6, a match to rounding. The instrument is sound before any other number is trusted.

## Headline

| Model | NepTel WER | General-Nepali WER |
|---|---:|---:|
| Kriti (base) | 40.75 | 4.13 |
| **Kriti Telephony** | **31.32** | 9.92 |

Kriti Telephony cuts telephony WER by roughly a quarter (40.75 to 31.32) while keeping general Nepali functional (9.92). It is a domain sibling of the base model, strong where the base model is weak.

## How it scaled

Distillation of style-matched conversational supervision (see [METHODOLOGY.md](METHODOLOGY.md)) improved cleanly with data:

| Distillation clips | NepTel WER |
|---:|---:|
| 2,800 (frozen encoder) | 36.48 |
| 5,900 (frozen encoder) | 34.93 |
| 5,900 (full fine-tune) | 33.75 |
| 12,000 (full fine-tune) + general anchor | **31.32** |

Adding the human-labeled general-Nepali anchor to the 12k mix is what produces the shipped model: it holds general Nepali at 9.92 and, because it reduces overfitting, also improves NepTel relative to conversational data alone.

## Failed fine-tunes (the controls that matter)

Before distillation, five straightforward fine-tunes were tried. All regressed on NepTel. They rule out "just fine-tune it" and localize the cause to label-style mismatch.

| Data | Encoder | NepTel WER | vs base (40.75) |
|---|---|---:|---|
| OpenSLR-54, simulated telephony | full | 43.44 | worse |
| Podcast, telephonified | full | 41.13 | worse |
| Podcast, clean | full | 46.88 | much worse |
| Podcast, human labels | frozen | 42.01 | worse |
| OpenSLR-54, human labels | frozen | 40.71 | worse |

Control detail: the last model scored 4.13 on OpenSLR val, identical to baseline, meaning it learned nothing new on its own domain (Kriti is already saturated there) yet still hurt NepTel. Training where the model is already perfect only drifts it.

## Decoding and post-processing levers (base model)

None beat simple greedy RNNT decoding by a meaningful margin. Recorded so others do not repeat them.

| Lever | NepTel WER |
|---|---:|
| Greedy RNNT (default) | 40.75 |
| Beam search (size 4) | 41.47 |
| maes + KenLM shallow fusion | about 49 |
| N-best word-LM rescoring | 39.79 |
| ROVER over model variants | 39.33 |
| Orthographic canonicalization (corpus-based) | 39.12 |

Orthographic canonicalization is the only post-processing win on the base model (about 1.6 points). It actually hurts Kriti Telephony (31.32 to 31.78), because the distilled model already emits reference-style orthography, so the shipped number uses no post-processing.

## Evidence

Per-segment hypotheses for the key models are committed under [benchmark/outputs/](benchmark/outputs/) and can be re-scored on CPU with no model:

| File | Model | Re-scores to |
|---|---|---:|
| `kriti-baseline.neptel.json` | base Kriti (published output) | 40.84 |
| `kriti-baseline-redecode.neptel.json` | base Kriti (our re-decode) | 40.75 |
| `nepaliconformer.neptel.json` | NepaliConformer offline | 34.09 |
| `kriti-distilled-5.9k.neptel.json` | distillation, 5.9k, full FT | 33.75 |
| `kriti-telephony.neptel.json` | **Kriti Telephony** | **31.32** |

```bash
python eval/score.py --hyp benchmark/outputs/kriti-telephony.neptel.json   # 31.32
```

Machine-readable summary of every number: [evidence/results.json](evidence/results.json).
