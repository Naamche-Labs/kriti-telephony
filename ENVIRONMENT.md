# Environment

Kriti is a NeMo 1.23-era model built on the **AI4Bharat NeMo fork**, not mainline NeMo. Getting a working environment on modern hardware (H100 / Hopper, driver 580) was the single hardest part of this project. These are the exact pins that work.

## Platform
- **GPU:** NVIDIA H100 (Hopper, `sm_90`), driver 580
- **Python:** 3.10.12
- **NeMo:** AI4Bharat fork, installed editable (`pip install -e '.[runtime]'`) — **not** `nvidia-nemo-toolkit` from PyPI. Mainline NeMo cannot load Kriti (its multilingual tokenizer/joint differ).

## Pinned packages
```
torch==2.13.0
pytorch-lightning==2.5.6
numba==0.59.1
cuda-python==12.2.1
numpy==1.26.4
pyarrow==16.1.0
scipy==1.15.3
soundfile==0.14.0
librosa==0.11.0
huggingface-hub==0.23.2
transformers==4.46.3
datasets==2.19.2
Cython==0.29.37
kenlm==0.2.0          # only for the LM-fusion experiments in RESULTS.md
```

## The three non-obvious requirements

1. **`NUMBA_CUDA_USE_NVIDIA_BINDING=1` — mandatory on Hopper.** Kriti's RNNT loss is `warprnnt_numba`, a numba-JIT CUDA kernel. numba's own CUDA binding **segfaults** on H100 + driver 580. Forcing NVIDIA's official CUDA bindings fixes it — but only with `cuda-python==12.2.1` (newer cuda-python reorganized its module layout and breaks numba's import; older lacks the API). Always launch training/eval with this env var set.

2. **`precision=32-true` for training.** The numba RNNT kernel raises `NumbaNotImplementedError` on bfloat16 tensors. Full fp32 avoids it; an H100 has ample memory for a 119M model at batch 16.

3. **`restore_from(..., strict=False)`.** Kriti's multi-softmax joint carries per-language head keys that mainline loaders reject. Load with `strict=False` (the model's own loader does the same) and set `cur_decoder="rnnt"`, `language_id="ne"` for Nepali inference.

## Quick setup sketch
```bash
python3.10 -m venv venv && source venv/bin/activate
git clone <ai4bharat-nemo-fork> nemo && cd nemo && git checkout <pinned-commit>
pip install "Cython<3"
pip install --only-binary=:all: numpy==1.26.4 pyarrow==16.1.0
pip install -e '.[runtime]'
pip install cuda-python==12.2.1 numba==0.59.1
# then always:  export NUMBA_CUDA_USE_NVIDIA_BINDING=1
```

If a decode segfaults immediately, you've missed requirement (1). If it dies with a `|V2` numba dtype error, you've missed requirement (2).
