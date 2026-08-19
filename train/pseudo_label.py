import json, os, glob
import nemo.collections.asr as nemo_asr
ck=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ampixa--nepali-conformer-offline/snapshots/*/nepali_conformer_offline.nemo"))[0]
m=nemo_asr.models.ASRModel.restore_from(ck, map_location="cuda"); m.eval()
# use clean conversational podcast clips (NepTel is vendor-cleaned -> clean audio)
rows=[json.loads(l) for l in open("/home/ubuntu/kriti-ft/data/conv_train.json") if l.strip()]
clean=[r for r in rows if r["audio_filepath"].endswith("_c.wav")]
files=[r["audio_filepath"] for r in clean]
print("pseudo-labeling", len(files), "clean conversational clips with nepaliconformer", flush=True)
o=m.transcribe(files, batch_size=16, verbose=False)
c=o[0] if isinstance(o,tuple) else o
man=[]
for r,x in zip(clean,c):
    t=(x if isinstance(x,str) else x.text)
    t=" ".join(w for w in t.split() if w!="<breath>")
    if len(t.split())>=2:
        man.append({"audio_filepath":r["audio_filepath"],"duration":r["duration"],"text":t,"lang":"ne"})
import random; random.seed(11); random.shuffle(man)
open("/home/ubuntu/kriti-ft/data/distill_train.json","w").write("\n".join(json.dumps(x,ensure_ascii=False) for x in man[150:]))
open("/home/ubuntu/kriti-ft/data/distill_val.json","w").write("\n".join(json.dumps(x,ensure_ascii=False) for x in man[:150]))
print("DISTILL_LABELED", len(man), "train", len(man)-150, flush=True)
