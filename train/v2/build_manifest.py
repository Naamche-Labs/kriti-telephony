import json,random
random.seed(11)
iv=[json.loads(l) for l in open("/home/rani_beat/iv_train.json")]
tel={}
for l in open("/home/rani_beat/iv_train_tel.json"):
    r=json.loads(l); tel[r["audio_filepath"].split("/")[-1].rsplit("_",1)[0]]=r
out=[]
for r in iv:
    k=r["audio_filepath"].split("/")[-1][:-4]
    out.append(tel.get(k,r))                     # each clip once: phone version if made, else clean
mixed=[json.loads(l) for l in open("/home/kriti/manifests/mixed_train.json")]
for r in mixed: r["text"]=r["text"].replace("<unk>"," ").strip()
out+=mixed*2                                     # keep Kriti-Telephony conversational/telephony style
random.shuffle(out)
with open("/home/rani_beat/train_v2.json","w") as f:
    for r in out: f.write(json.dumps({"audio_filepath":r["audio_filepath"],"duration":r["duration"],"text":r["text"],"lang":"ne"},ensure_ascii=False)+"\n")
val=[json.loads(l) for l in open("/home/rani_beat/iv_valid_tel.json")]; random.shuffle(val)
with open("/home/rani_beat/dev_v2.json","w") as f:
    for r in val[:800]: f.write(json.dumps({"audio_filepath":r["audio_filepath"],"duration":r["duration"],"text":r["text"],"lang":"ne"},ensure_ascii=False)+"\n")
print("train",len(out),"hours",round(sum(r["duration"] for r in out)/3600,1),"tel",sum(1 for r in out if "ivtel" in r["audio_filepath"]))
