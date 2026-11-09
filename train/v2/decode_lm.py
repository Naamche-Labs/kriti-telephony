import os,sys,json,warnings,argparse; warnings.filterwarnings("ignore"); os.environ["HOME"]="/home"
sys.path.insert(0,"/home/rani_beat/lm"); sys.path.insert(0,"/home/kriti/neptel02/rani-asr/eval")
from omegaconf import open_dict
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
import score_v02 as S, patch
ap=argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--set",default="dev"); ap.add_argument("--lm",default=""); ap.add_argument("--alphas",default="0"); ap.add_argument("--beam",type=int,default=4); ap.add_argument("--n",type=int,default=400); ap.add_argument("--out",default="")
a=ap.parse_args()
m=EncDecHybridRNNTCTCBPEModel.restore_from(a.ckpt,map_location="cuda",strict=False); m.cur_decoder="rnnt"; m.eval(); patch.patch(m)
if a.set=="dev":
    rows=[json.loads(l) for l in open("/home/rani_beat/dev_v2.json")][:a.n]; paths=[r["audio_filepath"] for r in rows]; refs=[r["text"] for r in rows]; segs=None
else:
    B="/home/kriti/neptel02"; R=json.load(open(B+"/rani-asr/benchmark/references.json"))["segments"]; paths=[B+"/neptel/v0.2/audio/"+s["seg"] for s in R]; refs=[s["reference"] for s in R]; segs=[s["seg"] for s in R]
def wer(H):
    e=n=0
    for r,h in zip(refs,H): r=S.normalize(r).split(); e+=S.edits(r,S.normalize(h).split()); n+=len(r)
    return round(100*e/n,2)
tx=lambda h: h.text if hasattr(h,"text") else str(h)
for al in [float(x) for x in a.alphas.split(",")]:
    dec=m.cfg.decoding
    with open_dict(dec):
        dec.strategy="maes"; dec.beam.beam_size=a.beam; dec.beam.return_best_hypothesis=True
        dec.beam.maes_num_steps=2; dec.beam.maes_prefix_alpha=1; dec.beam.maes_expansion_gamma=2.3; dec.beam.maes_expansion_beta=2
        dec.beam.ngram_lm_model=a.lm if (a.lm and al>0) else None; dec.beam.ngram_lm_alpha=al
    m.change_decoding_strategy(dec,decoder_type="rnnt")
    lb=m.decoding.blank_id; m.decoder.blank_idx=lb
    d=m.decoding.decoding
    if hasattr(d,"blank"): d.blank=lb; d.vocab_size=lb
    o=m.transcribe(paths,batch_size=16,language_id="ne"); o=o[0] if isinstance(o,tuple) else o
    H=[tx(h) for h in o]; w=wer(H); print(f"RESULT set={a.set} beam={a.beam} alpha={al} lm={os.path.basename(a.lm) if a.lm else None} WER={w}",flush=True)
    if a.out and segs: json.dump([{"seg":s,"text":h} for s,h in zip(segs,H)],open(a.out.replace(".json",f"_a{al}.json"),"w"),ensure_ascii=False,indent=0)
