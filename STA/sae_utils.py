import plotly.express as px
from typing import Any, Dict, Optional, Protocol, Tuple
import tempfile
import os, sys
import torch
from torch.utils.data import DataLoader
from sae_lens import SAE
from pathlib import Path
import json
import shutil
import numpy as np
# from sae_lens.toolkit.pretrained_sae_loaders import (
#     gemma_2_sae_loader,
#     get_gemma_2_config,
# )
from sae_lens import SAE, SAEConfig, LanguageModelSAERunnerConfig, SAETrainingRunner
from safetensors import safe_open
from functools import partial
import math
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
import random
# from baseline.caa.utils.input_format import llama3_chat_input_format

from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

sys.path.append("../")
import einops
import pdb

# fix the seed for reproducing atp selection
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

def activation_selection_contrastive_for_toxic_freq(
    model, 
    sae, 
    prompts,
    neg_prompts=None,
    prefix_prompts=None, 
    batch_size=4, 
    model_name=None, 
    desc="None",
    mean=True,
    model_name_or_path="None",
):
    print(f"Computing the activation selection for activation_selection_contrastive_for_toxic_freq")
    indice = len(prompts)

    shuffled_indices = list(range(indice))
    random.shuffle(shuffled_indices)

    ques_feature_acts_list = []
    pos_feature_acts_list = []
    neg_feature_acts_list = []

    for i in tqdm(shuffled_indices, desc=f"act for {desc}"):
        # try:
            ques = prompts[i]["ques"]
            # if "llama-3.1-8b-instruct" in model_name_or_path.lower():
            #     ques = llama3_chat_input_format(ques, model_output=None, system_prompt=None)
            pos_prompt = prompts[i]["pos"]
            neg_prompt = prompts[i]["neg"]

            ques_tokens = model.to_tokens(
                ques,
                padding_side="right" if "gemma" in model_name else "left",
            )

            ques_tokens_len = ques_tokens.shape[-1]

            pos_tokens = model.to_tokens(
                pos_prompt,
                padding_side="right" if "gemma" in model_name else "left",
            )
            pos_attention_mask = torch.ones_like(pos_tokens)
            pos_attention_mask[pos_tokens == model.tokenizer.pad_token_id] = 0

            pos_cache = get_cache_and_logits_act(model, pos_tokens, pos_attention_mask, sae)

            pos_feature_acts = pos_cache["hook_sae_acts_post"]
            pos_feature_acts = pos_feature_acts * pos_attention_mask.unsqueeze(-1)

            ques_feature_acts = pos_feature_acts[:, :ques_tokens_len, :]
            pos_feature_acts = pos_feature_acts[:, ques_tokens_len:, :]

            ques_feature_acts = einops.reduce(
                ques_feature_acts,
                "batch ques feature_nums -> batch feature_nums",
                "mean",
            )

            pos_feature_acts = einops.reduce(
                pos_feature_acts,
                "batch pos feature_nums -> batch feature_nums",
                "mean",
            )

            neg_tokens = model.to_tokens(
                neg_prompt,
                padding_side="right" if "gemma" in model_name else "left",
            )
            neg_attention_mask = torch.ones_like(neg_tokens)
            neg_attention_mask[neg_tokens == model.tokenizer.pad_token_id] = 0

            neg_cache = get_cache_and_logits_act(model, neg_tokens, neg_attention_mask, sae)

            neg_feature_acts = neg_cache["hook_sae_acts_post"]
            neg_feature_acts = neg_feature_acts * neg_attention_mask.unsqueeze(-1)
            neg_feature_acts = neg_feature_acts[:, ques_tokens_len:, :]

            neg_feature_acts = einops.reduce(
                neg_feature_acts,
                "batch neg feature_nums -> batch feature_nums",
                "mean",
            )

            ques_feature_acts_list.append(ques_feature_acts)
            pos_feature_acts_list.append(pos_feature_acts)
            neg_feature_acts_list.append(neg_feature_acts)
        # except RuntimeError as e:
        #     if "out of memory" in str(e):
        #         print(f"Skipping due to OOM at index {i}")
        #         torch.cuda.empty_cache()
        #     else:
        #         raise e
    ques_feature_acts = torch.cat(ques_feature_acts_list, dim=0)
    pos_feature_acts = torch.cat(pos_feature_acts_list, dim=0)
    neg_feature_acts = torch.cat(neg_feature_acts_list, dim=0)
    # pdb.set_trace()
    feature_score = (
        pos_feature_acts.mean(0) - neg_feature_acts.mean(0)
    )

    pos_feature_freq = (pos_feature_acts > 0).float().sum(0)
    print(pos_feature_freq.shape)
    print(pos_feature_freq)
    neg_feature_freq = (neg_feature_acts > 0).float().sum(0)
    print(neg_feature_freq.shape)
    print(neg_feature_freq)

    pos_act_mean = pos_feature_acts.mean(0)
    print(pos_act_mean.shape)
    print(pos_act_mean)
    neg_act_mean = neg_feature_acts.mean(0)
    print(neg_act_mean.shape)
    print(neg_act_mean)

    return feature_score, pos_feature_freq, neg_feature_freq, pos_act_mean, neg_act_mean

def load_sae_from_dir(sae_dir: Path | str, device: str = "cpu") -> SAE:
    """
    Due to a bug (https://github.com/jbloomAus/SAELens/issues/168) in the SAE save implementation for SAE Lens we need to make
    a specialized workaround.

    WARNING this will be creating a directory where the files are LINKED with the exception of "cfg.json" which is copied. This is NOT efficient
    and you should not be calling it many times!

    This wraps: https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py#L284.

    SPECIFICALLY fix cfg.json.
    """
    sae_dir = Path(sae_dir)
    # print(f"Loading SAE from {sae_dir}")

    if not all([x.is_file() for x in sae_dir.iterdir()]):
        raise ValueError(
            "Not all files are present in the directory! Only files allowed for loading SAE Directory."
        )

    # https://github.com/jbloomAus/SAELens/blob/9dacd4a9672c138b7c900ddd9a28d1b3b3a0870c/sae_lens/config.py#L188
    # Load ourselves instead of from_json because there are some __dir__ elements that are not in the JSON
    # They should ALL be enumerated in `derivatives`
    ##### BEGIN #####
    cfg_f = sae_dir / "cfg.json"
    with open(cfg_f, "r") as f:
        cfg = json.load(f)
    derivatives = [
        k for k in cfg.keys() if k not in LanguageModelSAERunnerConfig.__annotations__.keys()
    ]
    derivative_values = [cfg[x] for x in derivatives]
    for x in derivatives:
        del cfg[x]
            
    runner_config = LanguageModelSAERunnerConfig(**cfg)
    # print(runner_config.__dict__)
    # assert all(
    #     [
    #         d in runner_config.__dict__ and runner_config.__dict__[d] == dv
    #         for d, dv in zip(derivatives, derivative_values)
    #     ]
    # )
    del derivative_values
    del derivatives
    ##### END #####

    # Load the SAE
    sae_config = runner_config.get_training_sae_cfg_dict()
    sae = None
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Copy in the CFG
        sae_config_f = temp_dir / "cfg.json"
        with open(sae_config_f, "w") as f:
            json.dump(sae_config, f)
        # Copy all the other files
        for name_f in sae_dir.iterdir():
            if name_f.name == "cfg.json":
                continue
            else:
                shutil.copy(name_f, temp_dir / name_f.name)
        # Load SAE
        sae = SAE.load_from_pretrained(temp_dir, device=device)
    # sae = SAE.load_from_pretrained(sae_dir, device=device)
    assert sae is not None and isinstance(sae, SAE)

    with safe_open(os.path.join(sae_dir, "sae.safetensors"), framework="pt", device=device) as f:  # type: ignore
        log_sparsity = f.get_tensor("sparsity")

    return sae, log_sparsity

def get_cache_and_logits_act(model, tokens, attention_mask, sae):
    '''
    get the activation cache for all sae modules
    dict_keys(['hook_sae_acts_pre', 'hook_sae_acts_post', 'hook_sae_recons', 'hook_sae_output'])
    Return: cache: Dict[str, torch.Tensor]
    '''
    filter_not_input = lambda name: "_input" not in name
    cache = {}

    def sae_fwd_hook(act, hook):
        cache[hook.name] = act.detach()

    sae.reset_hooks()
    sae.add_hook(filter_not_input, sae_fwd_hook, "fwd")

    # add hook for model, to replace the original mlp output with the sae reconstruction.
    def reconstr_direct(activations, hook):
        cache["activations"] = activations.detach()
        output = sae(activations)
        return output

    model.reset_hooks()

    direct_output = model.run_with_hooks(
        tokens,
        attention_mask=attention_mask,
        fwd_hooks=[
            (
                sae.cfg.hook_name,
                partial(reconstr_direct),
            ),
        ],
        return_type="loss",
    )

    sae.reset_hooks()
    model.reset_hooks()
    return cache