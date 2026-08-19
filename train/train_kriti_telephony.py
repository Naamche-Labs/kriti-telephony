import os, json
import pytorch_lightning as pl
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr
DATA="/home/ubuntu/kriti-ft/data"
def load(fn): return [json.loads(l) for l in open(fn) if l.strip()]
print("mixed train:", len(load(DATA+"/mixed_train.json")), flush=True)
m=nemo_asr.models.ASRModel.restore_from("/home/ubuntu/kriti-ft/kriti/kriti.nemo", map_location="cpu", strict=False)
for ds in [m.cfg.train_ds, m.cfg.validation_ds]:
    with open_dict(ds):
        ds.is_concat=False; ds.is_tarred=False; ds.batch_size=16; ds.num_workers=6
        for k in ["concat_sampling_technique","concat_sampling_probabilities","concat_shuffle","concat_sampling_seed","concat_sampling_probabilities_file"]: ds.pop(k,None)
with open_dict(m.cfg.train_ds): m.cfg.train_ds.manifest_filepath=DATA+"/mixed_train.json"; m.cfg.train_ds.shuffle=True
with open_dict(m.cfg.validation_ds): m.cfg.validation_ds.manifest_filepath=DATA+"/mixed_val.json"
with open_dict(m.cfg.optim): m.cfg.optim.lr=1e-5; m.cfg.optim.pop("sched",None)
m.setup_training_data(m.cfg.train_ds); m.setup_validation_data(m.cfg.validation_ds); m.setup_optimization(m.cfg.optim)
tr=pl.Trainer(devices=1,accelerator="gpu",max_epochs=3,precision="32-true",enable_checkpointing=False,logger=False,gradient_clip_val=1.0,log_every_n_steps=100,val_check_interval=1.0)
m.set_trainer(tr); print("TRAIN_START", flush=True); tr.fit(m)
m.save_to("/home/ubuntu/kriti-ft/kriti_mixed.nemo"); print("TRAIN_DONE", flush=True)
