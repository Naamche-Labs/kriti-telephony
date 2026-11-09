# Make beam search work on the multilingual-fork Kriti: fixed language "ne" in the joint, drop language_ids for beam infer.
import functools
def patch(m, lang="ne"):
    j=m.joint; orig=j.joint_after_projection
    def jap(f,g,language_ids=None): return orig(f,g,language_ids=language_ids or [lang])
    j.joint_after_projection=jap
    from nemo.collections.asr.parts.submodules import rnnt_beam_decoding as B
    for cls in [B.BeamRNNTInfer]:
        if getattr(cls,"_ne_patched",False): continue
        oc=cls.__call__
        def call(self,*a,language_ids=None,**k): return oc(self,*a,**k)
        cls.__call__=call; cls._ne_patched=True
    return m
