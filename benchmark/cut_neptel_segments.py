import json, os, glob, re
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

DS = open("/tmp/dspath.txt").read().strip()
REF = json.load(open(os.path.expanduser("~/kriti-ft/bench/benchmark/references.json")))["segments"]
OUT = os.path.expanduser("~/kriti-ft/neptel_audio")
os.makedirs(OUT, exist_ok=True)
SR = 16000

def load16k(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR); y = resample_poly(y, SR // g, sr // g).astype("float32")
    return y

def chunk_silence(y, sr=SR, max_s=25.0, min_s=1.5, win_ms=30, thresh_rel=0.08):
    """Consecutive <=max_s chunks, cut at a silence frame near max_s."""
    hop = int(sr * win_ms / 1000)
    nf = len(y) // hop
    rms = np.array([np.sqrt(np.mean(y[i*hop:(i+1)*hop]**2) + 1e-12) for i in range(nf)])
    thr = thresh_rel * rms.max()
    silent = rms < thr
    chunks = []
    start = 0  # sample index
    maxn = int(max_s * sr); minn = int(min_s * sr)
    i = 0  # frame index of current position
    while start < len(y):
        end_cap = start + maxn
        if end_cap >= len(y):
            chunks.append((start, len(y))); break
        # candidate cut = last silent frame within [start+minn, end_cap]
        lo_f = (start + minn) // hop
        hi_f = end_cap // hop
        sil_frames = [f for f in range(lo_f, min(hi_f, nf)) if silent[f]]
        if sil_frames:
            cut = sil_frames[-1] * hop  # cut at the latest silence before cap
        else:
            cut = end_cap
        chunks.append((start, cut))
        start = cut
    return chunks

# Build per-sample global chunk lists: channel0 then channel1
def sample_global(sample_dir):
    ch0 = load16k(os.path.join(DS, sample_dir, "SPEAKER_00.wav"))
    ch1 = load16k(os.path.join(DS, sample_dir, "SPEAKER_01.wav"))
    c0 = [("SPEAKER_00", a, b, ch0) for a, b in chunk_silence(ch0)]
    c1 = [("SPEAKER_01", a, b, ch1) for a, b in chunk_silence(ch1)]
    return c0 + c1, c0, c1

# Sample-01 (batch1): seg_0000-0013 = ch0 windows 0-13, seg_0014-0027 = ch1 windows 0-13
g1, c0_1, c1_1 = sample_global("Sample - 01")
g2, _, _ = sample_global("Sample - 02")
g3, _, _ = sample_global("Sample - 03")

def dur(ch): return (ch[2] - ch[1]) / SR

def emit(seg_name, ch):
    y = ch[3][ch[1]:ch[2]]
    sf.write(os.path.join(OUT, seg_name), y, SR, subtype="PCM_16")
    return dur(ch)

report = []
for s in REF:
    name = s["seg"]; ref_dur = s["dur_s"]
    if name.startswith("seg_"):
        k = int(re.findall(r"\d+", name)[0])
        ch = c0_1[k] if k < 14 else c1_1[k - 14]
    elif name.startswith("s02_"):
        k = int(re.findall(r"\d+", name)[-1]); ch = g2[k] if k < len(g2) else None
    elif name.startswith("s03_"):
        k = int(re.findall(r"\d+", name)[-1]); ch = g3[k] if k < len(g3) else None
    if ch is None:
        report.append((name, ref_dur, -1, 999)); continue
    d = emit(name, ch)
    report.append((name, ref_dur, round(d, 1), round(abs(d - ref_dur), 1)))

# duration-match diagnostics
mism = [r for r in report if r[3] > 1.0]
print("total segs written:", sum(1 for r in report if r[2] > 0))
print("dur mismatches (>1s):", len(mism))
for r in mism[:20]: print("  MISMATCH", r)
print("len g1=%d g2=%d g3=%d" % (len(g1), len(g2), len(g3)))
print("CUT_DONE")
