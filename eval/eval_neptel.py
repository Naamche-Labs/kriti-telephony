import json, os, sys, argparse, pickle
sys.path.insert(0, os.path.expanduser("~/kriti-ft/bench/asr"))
import numpy as np, torch
from nepali_normalize import normalize
import nemo.collections.asr as nemo_asr

BENCH=os.path.expanduser("~/kriti-ft/bench/benchmark"); DATA=os.path.expanduser("~/kriti-ft/data")
AUD=os.path.expanduser("~/kriti-ft/neptel_audio")
REF={s["seg"]:s["reference"] for s in json.load(open(BENCH+"/references.json"))["segments"]}
DUR={s["seg"]:s["dur_s"] for s in json.load(open(BENCH+"/references.json"))["segments"]}
names=[s for s in REF if os.path.exists(AUD+"/"+s)]; files=[AUD+"/"+n for n in names]
freq,canon=pickle.load(open(DATA+"/ortho.pkl","rb"))
def okey(w):
    return w.replace("ँ","").replace("ं","").replace("ः","").replace("ी","ि").replace("ू","ु").replace("ऱ","र").replace("़","")
def canonicalize(text):
    out=[]
    for w in text.split():
        cw=canon.get(okey(w))
        out.append(cw if (cw and cw!=w and freq.get(cw,0)>freq.get(w,0)) else w)
    return " ".join(out)
def wer(hypmap, canon_on):
    te=tn=0
    for seg in names:
        r=normalize(REF[seg]).split()
        if DUR[seg]>0 and len(r)/DUR[seg]>6.0: continue
        h=normalize(hypmap.get(seg,"")); h=canonicalize(h) if canon_on else h; h=h.split()
        R,H=len(r),len(h); d=np.zeros((R+1,H+1),int); d[:,0]=np.arange(R+1); d[0,:]=np.arange(H+1)
        for i in range(1,R+1):
            for j in range(1,H+1):
                d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(r[i-1]!=h[j-1]))
        te+=int(d[R,H]); tn+=R
    return 100.0*te/tn

ap=argparse.ArgumentParser(); ap.add_argument("--ckpt"); ap.add_argument("--tag",default="m")
a=ap.parse_args()
m=nemo_asr.models.ASRModel.restore_from(a.ckpt, map_location="cuda", strict=False)
m.eval(); m.cur_decoder="rnnt"
outs=m.transcribe(files, batch_size=16, verbose=False, language_id="ne")
cand=outs[0] if isinstance(outs,tuple) else outs
hyp={}
for n,o in zip(names,cand):
    t=o if isinstance(o,str) else (o.text if hasattr(o,"text") else str(o))
    hyp[n]=" ".join(w for w in t.split() if w!="<breath>")
json.dump([{"seg":k,"text":v} for k,v in hyp.items()], open(BENCH+"/outputs/%s.json"%a.tag,"w"), ensure_ascii=False)
raw=wer(hyp,False); can=wer(hyp,True)
print("RESULT tag=%s -> raw=%.2f  +ortho=%.2f  (competitor=34.09)" % (a.tag, raw, can), flush=True)
print("FINAL_DONE", flush=True)
