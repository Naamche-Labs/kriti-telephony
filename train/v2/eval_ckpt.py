import os,sys,json,warnings; warnings.filterwarnings("ignore"); os.environ["HOME"]="/home"
sys.path.insert(0,"/home/kriti/neptel02/rani-asr/eval"); import score_v02 as S
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
p=sys.argv[1]; tag=os.path.basename(p)
m=EncDecHybridRNNTCTCBPEModel.restore_from(p,map_location="cuda",strict=False); m.cur_decoder="rnnt"; m.eval()
tx=lambda h: h.text if hasattr(h,"text") else str(h)
def wer(R,H):
    e=n=0
    for r,h in zip(R,H): r=S.normalize(r).split(); e+=S.edits(r,S.normalize(h).split()); n+=len(r)
    return round(100*e/n,2)
dev=[json.loads(l) for l in open("/home/rani_beat/dev_v2.json")]
o=m.transcribe([d["audio_filepath"] for d in dev],batch_size=16,language_id="ne"); o=o[0] if isinstance(o,tuple) else o
dw=wer([d["text"] for d in dev],[tx(h) for h in o])
B="/home/kriti/neptel02"; refs=json.load(open(B+"/rani-asr/benchmark/references.json"))["segments"]
o=m.transcribe([B+"/neptel/v0.2/audio/"+s["seg"] for s in refs],batch_size=8,language_id="ne"); o=o[0] if isinstance(o,tuple) else o
hy=[{"seg":s["seg"],"text":tx(h)} for s,h in zip(refs,o)]
json.dump(hy,open(p.replace(".nemo",".neptel02.json"),"w"),ensure_ascii=False,indent=0)
nw=wer([s["reference"] for s in refs],[h["text"] for h in hy])
line=f"{tag}\tdev_tel {dw}\tneptel02 {nw}"
open("/home/rani_beat/results.tsv","a").write(line+"\n"); print(line,flush=True)
