import json,glob,os,re,io,tarfile,soundfile as sf,numpy as np,librosa
from multiprocessing import Pool
O="/home/rani_beat/extra/wav"; os.makedirs(O,exist_ok=True)
def chunks(a,sr):
    iv=librosa.effects.split(a,top_db=35,frame_length=1024,hop_length=256)
    out=[];cur=None
    for s,e in iv:
        if cur is None: cur=[s,e]
        elif (e-cur[0])/sr<=16 and (s-cur[1])/sr<1.0: cur[1]=e
        else: out.append(cur); cur=[s,e]
    if cur: out.append(cur)
    res=[]
    for s,e in out:
        while (e-s)/sr>18: res.append((s,s+16*sr)); s+=16*sr
        if (e-s)/sr>=2: res.append((s,e))
    return res
def yodas(i):
    rows=[]; tf=tarfile.open(f"/home/rani_beat/extra/yodas2/data/ne000/audio/{i}.tar.gz")
    for m in tf:
        if not m.name.endswith(".wav"): continue
        vid=os.path.basename(m.name)[:-4]
        try: a,sr=sf.read(io.BytesIO(tf.extractfile(m).read()),dtype="float32")
        except Exception: continue
        if a.ndim>1: a=a.mean(1)
        if sr!=16000: a=librosa.resample(a,orig_sr=sr,target_sr=16000); sr=16000
        for k,(s,e) in enumerate(chunks(a,sr)):
            f=f"{O}/yv_{vid}_{k}.wav"; sf.write(f,a[s:e],sr); rows.append({"audio_filepath":f,"duration":round((e-s)/sr,3),"orig":"","src":"yodas"})
    return rows
def parl(f):
    import pyarrow.parquet as pq; rows=[]
    for r in pq.read_table(f).to_pylist():
        try:
            a,sr=sf.read(io.BytesIO(r["audio"]["bytes"]),dtype="float32")
            if a.ndim>1: a=a.mean(1)
            if sr!=16000: a=librosa.resample(a,orig_sr=sr,target_sr=16000); sr=16000
            d=len(a)/sr
            if not (1.0<=d<=20): continue
            uid=r["utterance_id"]; p=f"{O}/p_{uid}.wav"; sf.write(p,a,sr); rows.append({"audio_filepath":p,"duration":round(d,3),"orig":r["sentence"],"src":"parliament"})
        except Exception: pass
    return rows
if __name__=="__main__":
    for f in glob.glob(O+"/y_*.wav"): os.remove(f)
    ids=[os.path.basename(x)[:-5] for x in sorted(glob.glob("/home/rani_beat/extra/yodas2/data/ne000/text/*.json"))]
    with Pool(26) as p: a=p.map(yodas,ids); b=p.map(parl,sorted(glob.glob("/home/rani_beat/extra/parliament/data/*.parquet")))
    rows=[x for r in a+b for x in r]
    with open("/home/rani_beat/extra/extra_raw.json","w") as fo:
        for x in rows: fo.write(json.dumps(x,ensure_ascii=False)+"\n")
    import collections; c=collections.Counter(x["src"] for x in rows); h=collections.Counter()
    for x in rows: h[x["src"]]+=x["duration"]/3600
    print("PREPDONE",dict(c),{k:round(v,1) for k,v in h.items()},flush=True)
