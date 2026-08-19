import json, os, sys, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.expanduser("~/kriti-ft/bench/asr"))
from nepali_normalize import normalize
import numpy as np
BENCH=os.path.expanduser("~/kriti-ft/bench/benchmark")
DATA=os.path.expanduser("~/kriti-ft/data")
REF={s["seg"]:s["reference"] for s in json.load(open(BENCH+"/references.json"))["segments"]}
DUR={s["seg"]:s["dur_s"] for s in json.load(open(BENCH+"/references.json"))["segments"]}

def okey(w):
    w=w.replace("ँ","").replace("ं","").replace("ः","")
    w=w.replace("ी","ि").replace("ू","ु").replace("ऱ","र").replace("़","")
    return w

# build lexicon from big corpora
freq=Counter()
for fn in [DATA+"/big_nepali_text.txt", DATA+"/lm_corpus.txt", DATA+"/conv_text.txt"]:
    if os.path.exists(fn):
        for line in open(fn, encoding="utf-8"):
            for w in normalize(line).split():
                freq[w]+=1
# canonical form per ortho-key = most frequent surface form
groups=defaultdict(list)
for w,c in freq.items(): groups[okey(w)].append((c,w))
canon={}
for k,lst in groups.items():
    lst.sort(reverse=True); canon[k]=lst[0][1]
print("lexicon words:", len(freq), "ortho-keys:", len(canon), flush=True)

def canonicalize(text, min_count=3):
    out=[]
    for w in text.split():
        k=okey(w)
        cw=canon.get(k)
        # only replace if canonical exists, differs, and is well-attested & more frequent than the surface form
        if cw and cw!=w and freq[cw]>=min_count and freq[cw]>freq.get(w,0):
            out.append(cw)
        else:
            out.append(w)
    return " ".join(out)

def wer(hypmap, apply_canon):
    te=tn=0
    for seg in REF:
        r=normalize(REF[seg]).split()
        if DUR[seg]>0 and len(r)/DUR[seg]>6.0: continue
        h=normalize(hypmap.get(seg,""))
        if apply_canon: h=canonicalize(h)
        h=h.split()
        R,H=len(r),len(h); d=np.zeros((R+1,H+1),int); d[:,0]=np.arange(R+1); d[0,:]=np.arange(H+1)
        for i in range(1,R+1):
            for j in range(1,H+1):
                d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(r[i-1]!=h[j-1]))
        te+=int(d[R,H]); tn+=R
    return 100.0*te/tn

base={x["seg"]:x["text"] for x in json.load(open(BENCH+"/outputs/kriti-naamche.json"))}
print("Kriti baseline: raw=%.2f  canonicalized=%.2f" % (wer(base,False), wer(base,True)), flush=True)
# also try different min_count
for mc in [1,2,5,10]:
    def wer2(hypmap):
        te=tn=0
        for seg in REF:
            r=normalize(REF[seg]).split()
            if DUR[seg]>0 and len(r)/DUR[seg]>6.0: continue
            h=canonicalize(normalize(hypmap.get(seg,"")), mc).split()
            R,H=len(r),len(h); d=np.zeros((R+1,H+1),int); d[:,0]=np.arange(R+1); d[0,:]=np.arange(H+1)
            for i in range(1,R+1):
                for j in range(1,H+1):
                    d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(r[i-1]!=h[j-1]))
            te+=int(d[R,H]); tn+=R
        return 100.0*te/tn
    print("  min_count=%d -> WER=%.2f" % (mc, wer2(base)), flush=True)
print("ORTHO_DONE", flush=True)
