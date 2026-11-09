# Phone-line simulation: 300-3400 Hz band, 8 kHz, one random narrowband codec (GSM / G.711 mu/A / G.726 / Opus-NB), back to 16 kHz.
import json, random, subprocess, os, sys
from multiprocessing import Pool
random.seed(7)
OUT="/home/rani_beat/ivtel"; os.makedirs(OUT,exist_ok=True)
CODECS=[("gsm",["-c:a","libgsm","-ar","8000","-ac","1"],"gsm"),("mulaw",["-c:a","pcm_mulaw","-ar","8000"],"wav"),("alaw",["-c:a","pcm_alaw","-ar","8000"],"wav"),
        ("g726",["-c:a","g726","-ar","8000","-b:a","32k"],"wav"),("opus",["-c:a","libopus","-ar","8000","-b:a",random.choice(["8k","12k"]),"-application","voip"],"ogg")]
def tel(x):
    name,args,ext=random.choice(CODECS)
    base=os.path.basename(x["audio_filepath"])[:-4]; mid=f"/dev/shm/{base}.{ext}"; out=f"{OUT}/{base}_{name}.wav"
    gain=random.uniform(-8,4)
    try:
        subprocess.run(["ffmpeg","-v","error","-y","-i",x["audio_filepath"],"-af",f"highpass=f=300,lowpass=f=3400,volume={gain:.1f}dB",*args,mid],check=True)
        subprocess.run(["ffmpeg","-v","error","-y","-i",mid,"-ar","16000","-ac","1",out],check=True); os.remove(mid)
        y=dict(x); y["audio_filepath"]=out; y["codec"]=name; return y
    except Exception: return None
if __name__=="__main__":
    src,dst,frac=sys.argv[1],sys.argv[2],float(sys.argv[3])
    rows=[json.loads(l) for l in open(src)]
    pick=[r for r in rows if random.random()<frac]
    with Pool(26) as p: res=[r for r in p.imap_unordered(tel,pick,chunksize=64) if r]
    with open(dst,"w") as f:
        for r in res: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print("TELDONE",dst,len(res),flush=True)
