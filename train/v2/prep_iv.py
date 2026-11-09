# IndicVoices Nepali -> 16k wav + NeMo manifests. Text = standard-spelling "normalized" field, tags stripped.
import pyarrow.parquet as pq, glob, json, os, io, re, ast, soundfile as sf, numpy as np
from multiprocessing import Pool
OUT="/home/rani_beat/ivwav"; os.makedirs(OUT,exist_ok=True)
TAG=re.compile(r"<[^>]*>|\[[^\]]*\]")
def clean(t): return re.sub(r"\s+"," ",TAG.sub(" ",t or "")).strip()
def do(f):
    split="valid" if "valid" in f else "train"; rows=[]
    tb=pq.read_table(f)
    for i,r in enumerate(tb.to_pylist()):
        try:
            rep=r.get("verification_report") or "{}"
            dec=ast.literal_eval(rep).get("decision","") if rep.startswith("{") else ""
            if dec in ("reject","rejected","bad"): continue
            d=float(r["duration"] or 0)
            if d<1 or d>24: continue
            txt=clean(r.get("normalized") or r.get("text"))
            if not txt: continue
            a,sr=sf.read(io.BytesIO(r["audio_filepath"]["bytes"]),dtype="float32")
            if a.ndim>1: a=a.mean(1)
            if sr!=16000:
                import librosa; a=librosa.resample(a,orig_sr=sr,target_sr=16000)
            p=f"{OUT}/{os.path.basename(f)[:-8]}_{i}.wav"; sf.write(p,a,16000)
            rows.append({"audio_filepath":p,"duration":round(len(a)/16000,3),"text":txt,"lang":"ne","split":split,"speaker":r["speaker_id"],"scenario":r["scenario"],"decision":dec})
        except Exception as e: pass
    return rows
if __name__=="__main__":
    fs=sorted(glob.glob("/home/rani_beat/iv/nepali/*.parquet"))
    with Pool(24) as p: res=p.map(do,fs)
    allr=[x for r in res for x in r]
    for sp in ("train","valid"):
        with open(f"/home/rani_beat/iv_{sp}.json","w") as fo:
            for x in allr:
                if x["split"]==sp: fo.write(json.dumps(x,ensure_ascii=False)+"\n")
    print("PREPDONE",len(allr),sum(x["duration"] for x in allr)/3600)
