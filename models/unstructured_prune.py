from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from colorama import Fore 

import torch
import torch.nn as nn
import json
import os
os.environ["CUDA_VISIBLE_DEVICE"] = '0'
import time
import random
import numpy as np
import scipy.sparse
from tqdm import tqdm

from .autolayer import LayerEngine


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
def prepare_calibration_input_opt(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers

    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(traindata), model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    inps.requires_grad = False
    cache = {
        "i": 0,
        "attention_mask": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    model.config.use_cache = use_cache

    position_ids = None

    return inps, outs, attention_mask, position_ids

def prepare_calibration_input(model, dataloader, device, n_samples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    #layers = model.model.layers
    #print(model)
    layers = model.base_model.model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (n_samples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids

def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

def prune_model(base_model_addr, sparsity_ratio, calibration_data, save_model_path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_addr,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model = PeftModel.from_pretrained(model, './tuned_models/cloud0510Lamp4/checkpoint-39')
    model = model.merge_and_unload()
    print(Fore.RED + "Cloud lora model merged.")

    model.seqlen = 2048
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_addr, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda:0")
    Lamda = 0.08
    
    print("target sparsity", sparsity_ratio)
    graph_string = "W:(ABSLOG)-(VAR)-(ATAN)-(7)"
    lengine = LayerEngine.from_string(graph_string)
    print(f"Current graph string: {graph_string}")
    
    ##### calucalte outlier ratio
    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader = []
    
    '''with open(calibration_data_path, 'r') as f:
        traindata = json.load(f)[0]['profile']'''
    
    
    traindata = calibration_data
    for line in tqdm(traindata):
        train_item = line['input']
        trainenc = tokenizer(train_item + line['output'], 
                             return_tensors='pt', 
                             max_length=model.seqlen, 
                             truncation=True, 
                             padding='max_length', 
                             padding_side='left')
        # assert train_questions[i]['id'] == train_outputs['golds'][i]['id']
        inp = trainenc.input_ids
        tar = inp.clone()
        query_len = tokenizer(train_item, return_tensors='pt').input_ids.shape[1]
        tar[:, :query_len] = -100
        dataloader.append((inp, tar))
    print("dataset loading complete")
    
    st = time.time()
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device, len(dataloader)
            )

    if "opt" in base_model_addr:
        layers = model.model.decoder.layers
    else:
        layers = model.base_model.model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(len(dataloader)):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        for name in subset:

            ##print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            # activation_data = torch.sqrt(
            #     wrapped_layers[name].scaler_row.reshape((1, -1))
            # )
            layer_wmetric.append(W_metric)

        for j in range(len(dataloader)):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        # This part is the key of computing the outlier 
        # for out_ratio in [args.Hyper_m]:
        #     out_ratio_layer = check_outlier_mean(
        #         layer_wmetric, out_ratio, name=f"layer_{i}"
        #     )
        #     print("layer outlier ratio", out_ratio, out_ratio_layer)
        ##print(layer_wmetric)
        out_ratio_layer = lengine.compute_importance(layer_wmetric)

        all_layer_ratio.append(out_ratio_layer)

    # 0-1 scale 
    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) / (all_layer_ratio.max() - all_layer_ratio.min())

    all_layer_ratio = [1-item for item in all_layer_ratio]
    all_layer_ratio = np.array(all_layer_ratio)

    print("before adjustment", all_layer_ratio)

    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
        1 / (all_layer_ratio.max() - all_layer_ratio.min()) * Lamda * 2
    )

    all_layer_ratio = (
        all_layer_ratio - np.mean(all_layer_ratio) + (1 - sparsity_ratio)
    )

    print(
        all_layer_ratio,
        np.mean(all_layer_ratio),
        np.max(all_layer_ratio),
        np.min(all_layer_ratio),
    )

    print("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    ############## prune

    use_cache = model.config.use_cache
    model.config.use_cache = False
    period1 = time.time() - st
    
    print("loading calibdation data")
    dataloader = []
    '''with open(calibration_data_path, 'r') as f:
        traindata = json.load(f)[0]['profile']'''
    
    traindata = calibration_data
    for line in tqdm(traindata):
        train_item = line['input']
        trainenc = tokenizer(train_item + line['output'], 
                             return_tensors='pt', 
                             max_length=model.seqlen, 
                             truncation=True, 
                             padding='max_length', 
                             padding_side='left')
        # assert train_questions[i]['id'] == train_outputs['golds'][i]['id']
        inp = trainenc.input_ids
        tar = inp.clone()
        query_len = tokenizer(train_item, return_tensors='pt').input_ids.shape[1]
        tar[:, :query_len] = -100
        dataloader.append((inp, tar))
    print("dataset loading complete")
    
    st = time.time()
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(
                model, dataloader, device
            )
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device, len(dataloader)
            )

    if "opt" in base_model_addr:
        layers = model.model.decoder.layers

    else:
        layers = model.base_model.model.model.layers

    mask = 0
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(len(dataloader)):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()

        for name in subset:

            ##print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            # W = subset[name].weight.data
            # part1 = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(
            #     W
            # ) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            # part2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # part2 = part2**0.5
            # W_metric = part1 * part2

            activation_data = torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            layer_sparsity_ratio = 1 - all_layer_ratio[i]

            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            
            # unstructured pruning
            indices = sort_res[1][
                :, : int(W_metric.shape[1] * layer_sparsity_ratio)
            ]
            W_mask.scatter_(1, indices, True)
    #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(len(dataloader)):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    period2 = time.time() - st
    print("time used:", period1 + period2)
    
    '''os.makedirs(save_model_path, exist_ok=True)
    for name, param in model.named_parameters():
        if param.dim() > 1:  # 只处理权重矩阵（忽略bias等）
            # 转换为COO稀疏格式
            sparse_param = scipy.sparse.coo_matrix(param.detach().cpu().numpy())
            
            # 保存三个核心组件
            torch.save({
                'data': torch.FloatTensor(sparse_param.data),
                'row': torch.LongTensor(sparse_param.row),
                'col': torch.LongTensor(sparse_param.col),
                'shape': sparse_param.shape
            }, os.path.join(save_model_path, f"{name}.sparse"))
        else:
            # 非矩阵参数正常保存
            torch.save(param, os.path.join(save_model_path, f"{name}.dense"))'''
    return model, tokenizer  
    
def load_sparse_model(model, load_dir):
    """加载稀疏格式模型"""
    for name, param in model.named_parameters():
        if os.path.exists(os.path.join(load_dir, f"{name}.sparse")):
            # 加载稀疏组件
            sparse_data = torch.load(os.path.join(load_dir, f"{name}.sparse"))
            
            # 重建COO矩阵
            coo_matrix = scipy.sparse.coo_matrix(
                (sparse_data['data'].numpy(),
                 (sparse_data['row'].numpy(), sparse_data['col'].numpy())),
                shape=sparse_data['shape']
            )
            
            # 转换为PyTorch稀疏张量
            values = torch.FloatTensor(coo_matrix.data)
            indices = torch.vstack([
                torch.LongTensor(coo_matrix.row),
                torch.LongTensor(coo_matrix.col)
            ])
            param.data = torch.sparse_coo_tensor(
                indices, values, coo_matrix.shape,
                device=param.device
            ).to_dense()  # 可选：保持密集格式便于计算
        elif os.path.exists(os.path.join(load_dir, f"{name}.dense")):
            param.data = torch.load(os.path.join(load_dir, f"{name}.dense"))
    return model

if __name__ == "__main__":
    base_model_addr = "../model_weights/Meta-Llama-3.1-8B-Instruct"
    sparsity_ratio = 0.5
    calibration_data_path = "./data/LaMP_4/processed/privacy_train.json"
    save_model_path = "./pruned_model/unstructured"
    
    prune_model(base_model_addr, sparsity_ratio, calibration_data_path, save_model_path)