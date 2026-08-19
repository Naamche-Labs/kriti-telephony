import json, os, glob, numpy as np, soundfile as sf, librosa
from datasets import load_dataset, Audio
import nemo.collections.asr as nemo_asr
TOK=os.environ.get("HF_TOKEN")
OUT="/home/ubuntu/kriti-ft/data/pod_clean"; os.makedirs(OUT,exist_ok=True)
# regenerate clean 16k wavs for ALL podcast clips
ds=load_dataset("Bijay13/nepali-podcast-code-switch-asr-dataset", split="train", token=TOK)
ds=ds.cast_column("audio", Audio(sampling_rate=16000))
files=[]
print("decoding", len(ds), "podcast clips", flush=True)
for i in range(len(ds)):
    try:
        a=ds[i]["audio"]; y=np.asarray(a["array"],dtype=np.float32)
        if y.ndim>1: y=y.mean(1)
        dur=len(y)/a["sampling_rate"]
        if dur<1 or dur>20: continue
        m=max(1e-9,np.max(np.abs(y))); y=(y/m*0.9).astype(np.float32)
        p=OUT+"/p%d.wav"%i; sf.write(p,y,16000); files.append((p,round(float(dur),2)))
    except Exception: continue
    if i%1500==0: print("  decoded",i,flush=True)
print("clean clips:", len(files), flush=True)
# pseudo-label ALL with nepaliconformer
ck=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ampixa--nepali-conformer-offline/snapshots/*/nepali_conformer_offline.nemo"))[0]
nc=nemo_asr.models.ASRModel.restore_from(ck, map_location="cuda"); nc.eval()
o=nc.transcribe([f for f,_ in files], batch_size=16, verbose=False)
c=o[0] if isinstance(o,tuple) else o
man=[]
for (p,dur),x in zip(files,c):
    t=(x if isinstance(x,str) else x.text); t=" ".join(w for w in t.split() if w!="<breath>")
    if len(t.split())>=2: man.append({"audio_filepath":p,"duration":dur,"text":t,"lang":"ne"})
import random; random.seed(13); random.shuffle(man)
open("/home/ubuntu/kriti-ft/data/distill_big_train.json","w").write("\n".join(json.dumps(x,ensure_ascii=False) for x in man[150:]))
open("/home/ubuntu/kriti-ft/data/distill_big_val.json","w").write("\n".join(json.dumps(x,ensure_ascii=False) for x in man[:150]))
print("DISTILL_BIG_DONE", len(man), "train", len(man)-150, flush=True)
