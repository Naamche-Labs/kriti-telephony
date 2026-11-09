# Convert a Kriti multilingual-fork .nemo into a plain mainline Nepali RNNT .nemo (no CTC, no other languages).
import sys,os,tarfile,tempfile,shutil,yaml,torch,warnings; warnings.filterwarnings("ignore")
from omegaconf import OmegaConf
import nemo.collections.asr as na
src,dst=sys.argv[1],sys.argv[2]
tmp=tempfile.mkdtemp(); tarfile.open(src).extractall(tmp)
c=yaml.safe_load(open(f"{tmp}/model_config.yaml")); ne=c["tokenizer"]["langs"]["ne"]
tok=f"{tmp}/ne_tok"; os.makedirs(tok)
shutil.copy(f"{tmp}/{ne['model_path'][5:]}",f"{tok}/tokenizer.model"); shutil.copy(f"{tmp}/{ne['vocab_path'][5:]}",f"{tok}/vocab.txt"); shutil.copy(f"{tmp}/{ne['spe_tokenizer_vocab'][5:]}",f"{tok}/tokenizer.vocab")
V=256
cfg={"sample_rate":16000,"labels":None,"model_defaults":c.get("model_defaults",{}),
 "tokenizer":{"dir":tok,"type":"bpe"},
 "preprocessor":c["preprocessor"],"spec_augment":c.get("spec_augment"),"encoder":c["encoder"],
 "decoder":{"_target_":"nemo.collections.asr.modules.RNNTDecoder","normalization_mode":None,"random_state_sampling":False,"blank_as_pad":True,"prednet":c["decoder"]["prednet"],"vocab_size":V},
 "joint":{"_target_":"nemo.collections.asr.modules.RNNTJoint","log_softmax":None,"preserve_memory":False,"fuse_loss_wer":False,"num_classes":V,"vocabulary":[],
          "jointnet":{k:v for k,v in c["joint"]["jointnet"].items()}},
 "decoding":{"strategy":"greedy_batch","greedy":{"max_symbols":10}},
 "loss":{"loss_name":"default"},
 "optim":{"name":"adamw","lr":1e-5},
 "train_ds":{"manifest_filepath":None,"sample_rate":16000,"batch_size":16,"shuffle":True,"use_start_end_token":False},"validation_ds":{"manifest_filepath":None,"sample_rate":16000,"batch_size":16,"shuffle":False,"use_start_end_token":False},"test_ds":None}
cfg["joint"]["jointnet"]["encoder_hidden"]=c["encoder"]["d_model"]; cfg["joint"]["jointnet"]["pred_hidden"]=c["decoder"]["prednet"]["pred_hidden"]
for k in ["multilingual","language_keys"]: cfg["joint"]["jointnet"].pop(k,None)
m=na.models.EncDecRNNTBPEModel(cfg=OmegaConf.create(cfg))
sd=torch.load(f"{tmp}/model_weights.ckpt",map_location="cpu",weights_only=False)
new={}
for k,v in sd.items():
    if k.startswith("ctc_decoder"): continue
    if k.startswith("joint.joint_net.2."):
        if k.startswith("joint.joint_net.2.ne."): new["joint.joint_net.2."+k.split(".ne.")[1]]=v
        continue
    if k=="decoder.prediction.embed.weight":
        e=v[:V+1].clone(); e[V]=0; new[k]=e; continue
    new[k]=v
miss,unexp=m.load_state_dict(new,strict=False)
print("missing",miss,"unexpected",unexp)
m.eval(); m.save_to(dst); print("SAVED",dst,"params",sum(p.numel() for p in m.parameters()))
