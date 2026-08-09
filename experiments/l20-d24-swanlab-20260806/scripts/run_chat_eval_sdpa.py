"""Run scripts.chat_eval with NanoChat's PyTorch SDPA fallback forced on."""

import os
import runpy

import nanochat.flash_attention as fa_module


fa_module._override_impl = "sdpa"
fa_module.USE_FA3 = fa_module._resolve_use_fa3()
assert fa_module.USE_FA3 is False
if int(os.environ.get("LOCAL_RANK", "0")) == 0:
    print("Attention backend override: PyTorch SDPA", flush=True)

runpy.run_module("scripts.chat_eval", run_name="__main__")
