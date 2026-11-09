# Kriti Telephony v2 recipe

Scripts as run for v2, on one GPU box with the AI4Bharat NeMo fork (see `../../ENVIRONMENT.md`). Paths point at that box (`/home/rani_beat`, `/home/kriti`); change them for your machine.

| Step | Script |
|---|---|
| IndicVoices Nepali to 16 kHz wav and manifests | `prep_iv.py` |
| Phone-line simulation (band, 8 kHz, GSM / G.711 / G.726 / Opus-NB) | `prep_tel.py` |
| Round 1 manifest (IndicVoices, half phone-simulated, plus the v1 mix) | `build_manifest.py` |
| Extra audio: YODAS2 `ne000` and Nepali Parliament DS v2, cut to clips | `prep_extra2.py` |
| Label the extra audio with the round-1 model, keep agreeing labels | `plabel2.py` |
| Fine-tune (TF32, batch 64, cosine LR); round 1 used lr 2e-5 constant, round 2 cosine 2e-5 to 1e-6 | `train_v4.py` |
| Per-checkpoint scoring on the dev set and NepTel 0.2 | `eval_ckpt.py` |
| Beam search (`maes`, width 4) on the multilingual fork | `decode_lm.py` with `patch.py` |
| FLEURS Nepali test | `fleurs_eval.py` |
| Reduce to a plain Nepali RNNT for mainline NeMo 3.0 | `convert_to_nemo3.py` |

`patch.py` exists because the fork's beam search does not pass the language id to the multilingual joint. It fixes the Nepali head and its blank index (256) for decoding only; weights are untouched.
