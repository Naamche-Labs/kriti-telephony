# Label extra audio with our best checkpoint. Keep a clip only if labels agree:
# parliament: keep the given sentence when our RNNT transcript is within 30% WER of it; otherwise our RNNT if RNNT and CTC heads agree within 10%.
# yodas: our RNNT transcript when RNNT and CTC agree within 15%.
import os,sys,json,warnings; warnings.filterwarnings("ignore"); os.environ["HOME"]="/home"
sys.path.insert(0,"/home/kriti/neptel02/rani-asr/eval"); import score_v02 as S
import torch; torch.set_float32_matmul_precision("high")
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
rows=[json.loads(l) for l in open("/home/rani_beat/extra/extra_raw.json")]
m=EncDecHybridRNNTCTCBPEModel.restore_from(sys.argv[1],map_location="cuda",strict=False); m.eval()
P=[r["audio_filepath"] for r in rows]; tx=lambda h: h.text if hasattr(h,"text") else str(h)
m.cur_decoder="rnnt"; o=m.transcribe(P,batch_size=64,language_id="ne"); R=[tx(h) for h in (o[0] if isinstance(o,tuple) else o)]
m.cur_decoder="ctc"; o=m.transcribe(P,batch_size=64,language_id="ne"); C=[tx(h) for h in (o[0] if isinstance(o,tuple) else o)]
def w(a,b):
    a=S.normalize(a).split(); b=S.normalize(b).split()
    return S.edits(a,b)/max(1,len(a))
keep=[]; stat={}
for r,rn,ct in zip(rows,R,C):
    if len(rn.split())<2: continue
    t=None
    if r["src"]=="parliament" and r["orig"] and w(r["orig"],rn)<=0.30: t=r["orig"]; k="parl_orig"
    elif w(rn,ct)<=(0.10 if r["src"]=="parliament" else 0.15): t=rn; k=r["src"]+"_pseudo"
    if t: keep.append({"audio_filepath":r["audio_filepath"],"duration":r["duration"],"text":t,"lang":"ne","src":k}); stat[k]=stat.get(k,0)+r["duration"]/3600
with open("/home/rani_beat/extra/extra_lab.json","w") as f:
    for x in keep: f.write(json.dumps(x,ensure_ascii=False)+"\n")
print("PLABELDONE kept",len(keep),"of",len(rows),{k:round(v,1) for k,v in stat.items()},flush=True)
