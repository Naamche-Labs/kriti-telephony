#!/usr/bin/env python3
"""Score a hypothesis file against the NepTel reference set.

This reproduces the official NepTel metric: word-level Levenshtein on normalized
text, with a speaking-rate gate that drops any reference implying >6 words/sec
(transcription-engine hallucinations). It uses the *official* normalizer and
references from the NepaliConformer repo so numbers match to the decimal.

Setup (one-time):
    git clone https://github.com/Ampixa/nepaliconformer  ../nepaliconformer
    # provides: asr/nepali_normalize.py  and  benchmark/references.json

Usage:
    python eval/score.py --hyp benchmark/outputs/kriti-telephony.neptel.json
    python eval/score.py --hyp <file> --refs ../nepaliconformer/benchmark/references.json

A hypothesis file is a JSON list: [{"seg": "seg_0000.wav", "text": "..."}, ...]
No GPU or model required, this scores committed per-segment outputs directly.
"""
import argparse, json, os, sys

MAX_WORDS_PER_SEC = 6.0


def load_normalizer(nc_repo):
    """Import the official NepTel normalizer from a NepaliConformer checkout."""
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


def wer(ref_words, hyp_words):
    R, H = len(ref_words), len(hyp_words)
    d = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        d[i][0] = i
    for j in range(H + 1):
        d[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1]),
            )
    return d[R][H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="hypothesis JSON: [{seg, text}, ...]")
    ap.add_argument("--refs", default=None, help="NepTel references.json (default: ../nepaliconformer/benchmark/references.json)")
    ap.add_argument("--nc-repo", default="../nepaliconformer", help="path to a NepaliConformer checkout (for the normalizer)")
    a = ap.parse_args()

    normalize = load_normalizer(a.nc_repo)
    refs_path = a.refs or os.path.join(a.nc_repo, "benchmark", "references.json")
    segments = json.load(open(refs_path))["segments"]
    REF = {s["seg"]: s["reference"] for s in segments}
    DUR = {s["seg"]: s["dur_s"] for s in segments}

    hyp = {x["seg"]: x["text"] for x in json.load(open(a.hyp))}

    tot_err = tot_ref = scored = gated = 0
    for seg, reference in REF.items():
        r = normalize(reference).split()
        if DUR[seg] > 0 and len(r) / DUR[seg] > MAX_WORDS_PER_SEC:
            gated += 1
            continue
        h = normalize(hyp.get(seg, "")).split()
        tot_err += wer(r, h)
        tot_ref += len(r)
        scored += 1

    print(f"WER {100.0 * tot_err / tot_ref:.2f}   "
          f"(scored {scored} segments, {tot_ref} reference words, {gated} rate-gated)")


if __name__ == "__main__":
    main()
