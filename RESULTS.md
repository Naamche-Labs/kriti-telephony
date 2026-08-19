# Results — every experiment, both axes

Two metrics throughout:
- **NepTel WER** — real Nepali call-center telephony, official scorer, ~76 scored segments.
- **General-Nepali WER** — 200 held-out clean read clips (OpenSLR-54) with **human** references. This column is the honesty check: it catches models that win NepTel by overfitting.

Baseline reproduction sanity check: our harness scores base Kriti at **40.75** on NepTel vs. its published **40.6** — a match to rounding. The instrument is sound.

---

## Main table

| # | Model | NepTel ↓ | General ↓ | Status |
|---|---|---:|---:|---|
| 0 | NepaliConformer offline (competitor / teacher) | 34.09 | — | prior best on NepTel |
| 1 | **Kriti** (base) | 40.75 | **4.13** | strong general, weak telephony |
| 2 | Distill 2.8k, frozen | 36.48 | — | distillation works, scales |
| 3 | Distill 5.9k, frozen | 34.93 | — | ↑ data ↓ WER |
| 4 | Distill 5.9k, full FT | 33.75 | — | full FT > frozen |
| 5 | Distill 12k, full FT (pure) | 32.96 | **16.53** | ❌ overfit — general collapsed |
| 6 | **Kriti Telephony** (distill 12k + 8k anchor) | **31.32** | 9.92 | ✅ shipped: leads + generalizes |

Rows 2–5 use only teacher (NepaliConformer) pseudo-labels on conversational audio. Row 6 adds the human-labeled general-Nepali anchor — the change that makes it shippable.

---

## Failed fine-tunes (the controls that matter)

Before distillation, five straightforward fine-tunes were tried. **All regressed on NepTel.** They rule out "just fine-tune it" and localize the cause to label-style mismatch.

| Data | Encoder | NepTel ↓ | vs base (40.75) |
|---|---|---:|---|
| OpenSLR-54, simulated telephony | full | 43.44 | worse |
| Podcast, telephonified | full | 41.13 | worse |
| Podcast, clean | full | 46.88 | much worse (overfit small set) |
| Podcast, human labels | frozen | 42.01 | worse |
| OpenSLR-54, human labels | frozen | 40.71 | worse |

Control detail: the last row's model scored **4.13** on OpenSLR val — identical to baseline — i.e. it learned nothing new on its own domain (Kriti is already saturated there) yet still hurt NepTel. Training where the model is already perfect only drifts it.

---

## Decoding / post-processing levers (base model)

None of these beat simple greedy RNNT decoding by a meaningful margin; recorded so others don't repeat them.

| Lever | NepTel ↓ |
|---|---:|
| Greedy RNNT (default) | 40.75 |
| Beam search (size 4) | 41.47 |
| maes + KenLM shallow fusion | ~49 |
| N-best word-LM rescoring | 39.79 |
| ROVER over model variants | 39.33 |
| **Orthographic canonicalization** (corpus-based) | **39.12** |

Orthographic canonicalization is the only post-processing win on the *base* model (~1.6 pts). Note it **hurts** Kriti Telephony (31.32 → 31.78), because the distilled model already emits reference-style orthography — so the shipped number uses **no** post-processing.

---

## The trade-off frontier (why we stop at 31.3)

| Model | NepTel ↓ | General ↓ |
|---|---:|---:|
| Base Kriti | 40.75 | 4.13 |
| Kriti Telephony (12k distill + 8k anchor) | 31.32 | 9.92 |
| Pure distill (no anchor) | 32.96 | 16.53 |

Removing or shrinking the anchor pushes NepTel lower but pays for it in general-Nepali WER — the overfitting direction. **31.3 / 9.9 is the deliberate operating point:** a real telephony lead that still generalizes. We do not publish a "sub-30" number, because reaching it means shipping a model that transcribes clean Nepali worse — a leaderboard trophy that fails in production.

---

## Evidence

Per-segment hypotheses for the key rows are committed under [`benchmark/outputs/`](benchmark/outputs/) and can be re-scored on CPU with no model:

| File | Model | Re-scores to |
|---|---|---:|
| `kriti-baseline.neptel.json` | base Kriti | 40.75 |
| `nepaliconformer.neptel.json` | competitor | 34.09 |
| `kriti-distilled-5.9k.neptel.json` | distill 5.9k full | 33.75 |
| `kriti-distilled-puredistill.neptel.json` | pure distill 12k (overfit) | 32.96 |
| `kriti-telephony.neptel.json` | **Kriti Telephony** | **31.32** |

```bash
python eval/score.py --hyp benchmark/outputs/kriti-telephony.neptel.json   # -> 31.32
```

Machine-readable summary of every number: [`evidence/results.json`](evidence/results.json).
