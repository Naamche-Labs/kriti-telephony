#!/usr/bin/env python3
"""Score a hypothesis file against the NepTel reference set.

This mirrors the official NepTel scorer (`eval/score_reference.py` in the
NepaliConformer repo): it skips excluded segments, applies the >6 words/sec
speaking-rate gate on the raw reference text, normalizes both sides, and computes
word-level Levenshtein. It uses the official normalizer and references so numbers
match the official tool to the decimal.

Setup (one-time):
    git clone https://github.com/Ampixa/nepaliconformer  ../nepaliconformer
    # provides: asr/nepali_normalize.py  and  benchmark/references.json

Usage:
    python eval/score.py --hyp benchmark/outputs/kriti-telephony.neptel.json
    python eval/score.py --hyp <file> --refs ../nepaliconformer/benchmark/references.json

A hypothesis file is a JSON list: [{"seg": "seg_0000.wav", "text": "..."}, ...]
No GPU or model required; this scores committed per-segment outputs directly.
"""
import argparse, json, os, sys

MAX_WORDS_PER_SEC = 6.0


def load_normalizer(nc_repo):
    asr_dir = os.path.join(nc_repo, "asr")
    if not os.path.isdir(asr_dir):
        sys.exit(
            f"Could not find the official normalizer at {asr_dir}.\n"
            "Clone it once:  git clone https://github.com/Ampixa/nepaliconformer "
            f"{nc_repo}\n(see the header of this file)."
        )
    sys.path.insert(0, asr_dir)
    from nepali_normalize import normalize  # noqa: E402
    return normalize


def load_refs(manifest, normalize):
    """Exact port of the official scorer's segment selection."""
    man = json.load(open(manifest, encoding="utf-8"))
    segs = []
    for m in man["segments"]:
        if m.get("excluded"):
            continue
        if not m.get("reference") and not m.get("chirp2"):
            continue
        text = m.get("reference") or m["chirp2"]
        rate = len(text.split()) / max(float(m.get("dur_s", 0)) or 0.1, 0.1)
        if rate > MAX_WORDS_PER_SEC:
            continue
        segs.append({"seg": m["seg"], "ref": normalize(text)})
    return segs


def wer(ref_words, hyp_words):
    R, H = len(ref_words), len(hyp_words)
    d = list(range(H + 1))
    for i in range(1, R + 1):
        prev, d[0] = d[0], i
        for j in range(1, H + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref_words[i - 1] != hyp_words[j - 1]))
            prev, d[j] = d[j], cur
    return d[H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="hypothesis JSON: [{seg, text}, ...]")
    ap.add_argument("--refs", default=None, help="NepTel references.json (default: ../nepaliconformer/benchmark/references.json)")
    ap.add_argument("--nc-repo", default="../nepaliconformer", help="path to a NepaliConformer checkout (for the normalizer)")
    a = ap.parse_args()

    normalize = load_normalizer(a.nc_repo)
    refs_path = a.refs or os.path.join(a.nc_repo, "benchmark", "references.json")
    segs = load_refs(refs_path, normalize)

    hyp = {x["seg"]: x["text"] for x in json.load(open(a.hyp))}

    tot_err = tot_ref = 0
    for s in segs:
        r = s["ref"].split()
        h = normalize(" ".join(t for t in hyp.get(s["seg"], "").split() if t != "<breath>")).split()
        tot_err += wer(r, h)
        tot_ref += len(r)

    print(f"WER {100.0 * tot_err / tot_ref:.2f}   (scored {len(segs)} segments, {tot_ref} reference words)")


if __name__ == "__main__":
    main()
