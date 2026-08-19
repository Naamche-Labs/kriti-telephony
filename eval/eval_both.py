import json, os, sys
sys.path.insert(0, os.path.expanduser("~/kriti-ft/bench/asr"))
from nepali_normalize import normalize
import numpy as np, torch
import nemo.collections.asr as nemo_asr
ck=sys.argv[1]; tag=sys.argv[2]
B=os.path.expanduser("~/kriti-ft/bench/benchmark"); AUD=os.path.expanduser("~/kriti-ft/neptel_audio")
REF={s["seg"]:s["reference"] for s in json.load(open(B+"/references.json"))["segments"]}
DUR={s["seg"]:s["dur_s"] for s in json.load(open(B+"/references.json"))["segments"]}
nt=[s for s in REF if os.path.exists(AUD+"/"+s)]
gen=[json.loads(l) for l in open("/home/ubuntu/kriti-ft/data/cleanslr_val.json")][:200]
def wer(refs,hyps):
    te=tn=0
    for r,h in zip(refs,hyps):
        r=normalize(r).split(); h=normalize(h).split()
        R,H=len(r),len(h); d=np.zeros((R+1,H+1),int); d[:,0]=np.arange(R+1); d[0,:]=np.arange(H+1)
        for i in range(1,R+1):
            for j in range(1,H+1): d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(r[i-1]!=h[j-1]))
        te+=int(d[R,H]); tn+=R
    return 100.0*te/tn
def nt_wer(hyp):
    rs=[];hs=[]
    for s in nt:
        r=normalize(REF[s]).split()
        if DUR[s]>0 and len(r)/DUR[s]>6: continue
        rs.append(REF[s]); hs.append(hyp[s])
    return wer(rs,hs)
m=nemo_asr.models.ASRModel.restore_from(ck,map_location="cuda",strict=False); m.eval(); m.cur_decoder="rnnt"
o=m.transcribe([AUD+"/"+s for s in nt],batch_size=16,verbose=False,language_id="ne"); c=o[0] if isinstance(o,tuple) else o
h1={s:(x if isinstance(x,str) else x.text) for s,x in zip(nt,c)}
o=m.transcribe([g["audio_filepath"] for g in gen],batch_size=16,verbose=False,language_id="ne"); c=o[0] if isinstance(o,tuple) else o
h2=[(x if isinstance(x,str) else x.text) for x in c]
print("BOTH tag=%s -> NepTel=%.2f  GeneralNepali=%.2f  (baseline: NepTel=40.75 Gen=4.13, competitor NepTel=34.09)" % (tag, nt_wer(h1), wer([g["text"] for g in gen],h2)), flush=True)
print("EVAL_BOTH_DONE", flush=True)
