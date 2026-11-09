# Round 2: init from best checkpoint, IndicVoices+Kriti mix + labelled extra audio, TF32, batch 64, cosine LR 2e-5 -> 1e-6.
import os,json,warnings,sys,math; warnings.filterwarnings("ignore"); os.environ["HOME"]="/home"
import torch; torch.set_float32_matmul_precision("high"); torch.backends.cudnn.allow_tf32=True
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr
INIT,OUT,EPOCHS,TRAIN=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
BS=64
n=sum(1 for l in open(TRAIN) if l.strip()); steps=math.ceil(n/BS)*EPOCHS
print("train",n,"steps",steps,"init",INIT,flush=True)
m=nemo_asr.models.ASRModel.restore_from(INIT,map_location="cpu",strict=False)
for ds in [m.cfg.train_ds,m.cfg.validation_ds]:
    with open_dict(ds):
        ds.is_concat=False; ds.is_tarred=False; ds.batch_size=BS; ds.num_workers=24; ds.max_duration=20
        for k in ["concat_sampling_technique","concat_sampling_probabilities","concat_shuffle","concat_sampling_seed","concat_sampling_probabilities_file"]: ds.pop(k,None)
with open_dict(m.cfg.train_ds): m.cfg.train_ds.manifest_filepath=TRAIN; m.cfg.train_ds.shuffle=True
with open_dict(m.cfg.validation_ds): m.cfg.validation_ds.manifest_filepath="/home/rani_beat/dev_v2.json"
with open_dict(m.cfg.optim):
    m.cfg.optim.lr=2e-5
    m.cfg.optim.sched={"name":"CosineAnnealing","warmup_steps":300,"min_lr":1e-6,"max_steps":steps}
tr=pl.Trainer(devices=1,accelerator="gpu",max_epochs=EPOCHS,precision="32-true",enable_checkpointing=False,logger=False,gradient_clip_val=1.0,log_every_n_steps=200,val_check_interval=1.0)
class EpochSave(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        p=OUT.replace(".nemo","_epoch%d.nemo"%trainer.current_epoch)
        try: pl_module.save_to(p); print("CKPT_SAVED",p,flush=True)
        except Exception as e: print("ckpt save failed:",e,flush=True)
tr.callbacks.append(EpochSave())
m.set_trainer(tr)
m.setup_training_data(m.cfg.train_ds); m.setup_validation_data(m.cfg.validation_ds); m.setup_optimization(m.cfg.optim)
print("TRAIN_START",flush=True); tr.fit(m); m.save_to(OUT); print("TRAIN_DONE",OUT,flush=True)
