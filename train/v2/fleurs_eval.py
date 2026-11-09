import os,sys,json,csv,warnings; warnings.filterwarnings("ignore"); os.environ["HOME"]="/home"
sys.path.insert(0,"/home/kriti/neptel02/rani-asr/eval"); import score_v02 as S
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
F="/home/rani_beat/fleurs"; rows=[l.rstrip("\n").split("\t") for l in open(F+"/data/ne_np/test.tsv")]
seen=set(); items=[]
for r in rows:
    if r[1] in seen or not os.path.exists(F+"/test/"+r[1]): continue
    seen.add(r[1]); items.append((F+"/test/"+r[1],r[2]))
for p in sys.argv[1:]:
    m=EncDecHybridRNNTCTCBPEModel.restore_from(p,map_location="cuda",strict=False); m.cur_decoder="rnnt"; m.eval()
    o=m.transcribe([a for a,_ in items],batch_size=16,language_id="ne"); o=o[0] if isinstance(o,tuple) else o
    we=wn=ce=cn=0; out=[]
    for (a,t),h in zip(items,o):
        h=h.text if hasattr(h,"text") else str(h); out.append({"file":os.path.basename(a),"text":h})
        r=S.normalize(t).split(); hh=S.normalize(h).split(); we+=S.edits(r,hh); wn+=len(r)
        rc=list(" ".join(r)); hc=list(" ".join(hh)); ce+=S.edits(rc,hc); cn+=len(rc)
    json.dump(out,open(p.replace(".nemo",".fleurs.json"),"w"),ensure_ascii=False,indent=0)
    print("FLEURS",os.path.basename(p),"n",len(items),"WER",round(100*we/wn,2),"CER",round(100*ce/cn,2),flush=True)
