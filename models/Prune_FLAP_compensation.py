import torch
import torch.nn as nn
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from colorama import Fore, init
import copy
import gc
init(autoreset=True)

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids
        
def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the C4 (Common Crawl) dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation dataset.
    """
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    # traindata = load_dataset('json', data_files={'train': 'datasets/c4/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('json', data_files={'validation': 'datasets/c4/c4-validation.00000-of-00008.json.gz'}, split='validation')
    
    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Define BiasGPT class
class BiasGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, metric):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.out_dim = layer.weight.data.shape[0]
        self.in_dim = layer.weight.data.shape[1]
        self.type = metric
        self.nsamples = 0

        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        if self.type == "WIFN":
            self.scaler_inp = torch.zeros((self.in_dim), device=self.dev)
        else:   
            self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (dim, seqlen)

        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.type == "WIFN":
            inp = inp.type(torch.float32)
            self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
            self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / (self.nsamples + batch_size)
        else:
            if self.nsamples == 0:
                self.fluc_inp = 0
            else:
                self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号

        self.nsamples += batch_size

        
    def free(self):
        self.baseline_inp = None
        if hasattr(self, 'fluc_inp'):
            self.fluc_inp = None
        if hasattr(self, 'scaler_inp'):
            self.scaler_inp = None
        torch.cuda.empty_cache()  

# create a dictionary to map the method name to the function
"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}


def find_layers(module, layers=[nn.Linear], name=''):
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
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.
    
    Args:
        model (nn.Module): The model to check.
        
    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            sub_count += W.numel()
            count += W.numel()
            if 'self_attn' in name:
                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size
            else:
                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()
            
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration. 
    
    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded. 
        
    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((2048, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 


def compress(layer, threshold, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    """
    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    
    # Real Pruning
    # Attention Weight Pruning
    if not bias:
        if attn_mask is not None:
            # ensure that retain_heads mod num_key_value_heads == 0
            temp_attn = (attn_mask>threshold)
        
            head_dim = layer.self_attn.head_dim
            retain_heads = torch.count_nonzero(temp_attn)   
            
            _ , indices = torch.topk(attn_mask, retain_heads)
            attn_mask = torch.zeros_like(temp_attn)
            attn_mask[indices] = 1
            attn_mask = attn_mask.bool()
            
            attn_k_mask = attn_mask.repeat_interleave(head_dim)
            attn_q_mask = attn_mask.repeat_interleave(head_dim * layer.self_attn.num_key_value_groups)
            
            # Prune the query, key and value projection weights
            # We reduce the size of the weights based on the attention mask
            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_q_mask)[0], :]
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_k_mask)[0], :]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_k_mask)[0], :]

            # Update output dimensions of q, k, v projections based on remaining heads
            layer.self_attn.q_proj.out_features = attn_q_mask.sum().item()
            layer.self_attn.k_proj.out_features = attn_k_mask.sum().item()
            layer.self_attn.v_proj.out_features = attn_k_mask.sum().item()
                
            # Prune the output projection weight
            output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_q_mask)[0]]
            
            # Update layer configurations for the new output shape after pruning
            layer.self_attn.num_key_value_heads = retain_heads
            layer.self_attn.num_heads = layer.self_attn.num_key_value_groups * layer.self_attn.num_key_value_heads
            
            # Assign the pruned weights
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Pruning
        if mlp_mask is not None:
            # Prune the up and gate projection weights
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
            
            # Update output dimensions of up and gate projections based on the mlp mask
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            layer.mlp.down_proj.in_features = mlp_mask.sum().item()
            
            output_weight = layer.mlp.down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
                
            # Prune the down projection weight
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  
                
            # Assign the pruned weights
            layer.mlp.down_proj.weight.data = output_weight
    else:
        if attn_mask is not None:
            temp_attn = (attn_mask>threshold)
            
            head_dim = layer.self_attn.head_dim
            retain_heads = torch.count_nonzero(temp_attn)   
            
            _ , indices = torch.topk(attn_mask, retain_heads)
            attn_mask = torch.zeros_like(temp_attn)
            attn_mask[indices] = 1
            attn_mask = attn_mask.bool()
            
            attn_q_mask = attn_mask.repeat_interleave(head_dim * layer.self_attn.num_key_value_groups)
            output_weight = layer.self_attn.o_proj.weight.data
            # output_bias = ((attn_mean_inp.float() * ~attn_q_mask.to(device)) @ output_weight.T)
            output_bias = ((attn_mean_inp.float()) @ output_weight.T)
            layer.self_attn.o_proj.bias.data = output_bias
        if mlp_mask is not None:
            output_weight = layer.mlp.down_proj.weight.data
            # output_bias = ((mlp_mean_inp.float() * ~mlp_mask.to(device)) @ output_weight.T)
            output_bias = ((mlp_mean_inp.float()) @ output_weight.T)
            layer.mlp.down_proj.bias.data = output_bias
    
    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()
    
    
def cal_remove_neuron(args, model):
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if args.structure == "UL-MM":
        remove_params = args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * (args.remove_heads // num_layers) * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    else:
        remove_params = num_layers * args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * args.remove_heads * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))

def check_layer_compatibility(q_proj, k_proj, v_proj, o_proj):
    assert q_proj.out_features == k_proj.out_features == v_proj.out_features
    assert o_proj.in_features == q_proj.out_features

def replace_linear(old_layer, pruned_weight, pruned_bias=None):
    # 根据剪枝后的权重创建新层
    out_features, in_features = pruned_weight.shape
    new_layer = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=old_layer.bias is not None
    )
    
    # 复制剪枝后的权重
    new_layer.weight.data = pruned_weight
    if pruned_bias is not None and new_layer.bias is not None:
        new_layer.bias.data = pruned_bias
    
    return new_layer

def prune_flap(base_model_addr, sparsity_ratio, calibration_data, save_model_path):
    model = LlamaForCausalLM.from_pretrained(base_model_addr)
    print(model.device)
    for i in range(32):
        model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(torch.zeros((4096,))).to(model.device)
        model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(torch.zeros((4096,))).to(model.device)
        torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
        torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)
    model.seqlen = 128
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_addr)
    tokenizer.pad_token = tokenizer.eos_token
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False
    # base_model = copy.deepcopy(model)
    
    prune_model(model, model, tokenizer, sparsity_ratio, calibration_data['cluster_data'], bias=False)
    print(model)
    prune_model(model, model, tokenizer, sparsity_ratio, calibration_data['self_data'], bias=True)
    
    # del model
    # del base_model
    # del pruned_model
    torch.cuda.empty_cache()
    gc.collect()
    return model, tokenizer

def prune_model(base_model, pruned_model, tokenizer, sparsity_ratio, calibration_data, bias):
    
    ### Calibration data processing.
    # calibration_data: list of data lines. [{"input": xxx, "output": xxx}]
    print("loading calibdation data")
    nsamples = len(calibration_data)
    dataloader = []
    for line in tqdm(calibration_data):
        train_item = line['input']
        trainenc = tokenizer(train_item + line['output'], 
                             return_tensors='pt', 
                             max_length=base_model.seqlen, 
                             truncation=True, 
                             padding='max_length', 
                             padding_side='left')
        # assert train_questions[i]['id'] == train_outputs['golds'][i]['id']
        inp = trainenc.input_ids
        tar = inp.clone()
        query_len = tokenizer(train_item, return_tensors='pt').input_ids.shape[1]
        tar[:, :query_len] = -100
        dataloader.append((inp, tar))
    #dataloader = process_cali(calibratio)
    print("dataset loading complete")
    
    device = torch.device("cuda:0")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(base_model, dataloader, device)
    
    ###
    layers = base_model.model.layers
    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []
        
    # Split into sub-problems, separate statistics for each module
    ## why only o_proj and down_proj?
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(base_model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = base_model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], "WIFV")            

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        # print(inps.shape)
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if name == 'self_attn.o_proj':
                W_metric = metrics["WIFV"](wrapped_layers, subset, name) ** 2
                attn_metric_list.append(W_metric.cpu())
                # print(W_metric.shape)
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics["WIFV"](wrapped_layers, subset, name)
                mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            wrapped_layers[name].free()

        inps, outs = outs, inps # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

    all_attn_metric, all_mlp_metric = [], []
    for i in range(len(layers)):
        attn_metric = attn_metric_list[i].unsqueeze(0)
        attn_metric = standarlization(attn_metric)
        attn_metric = attn_metric.reshape(-1, 512).mean(dim=1)
        all_attn_metric.append(attn_metric)
        
        mlp_metric = mlp_metric_list[i].unsqueeze(0)
        mlp_metric = standarlization(mlp_metric)
        all_mlp_metric.append(mlp_metric.squeeze(0))
    
    prune_metric = torch.cat(all_attn_metric + all_mlp_metric, dim=0)
    sorted_prune, indices = torch.sort(prune_metric, descending=True)
    compression_weight = torch.ones_like(indices)
    compression_weight[indices < attn_metric.numel()] = 1280.0 / 3
    threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - sparsity_ratio)))]
    print("Threshold:", threshold)
        
    # attn_mask = (attn_metric > threshold)
    # mlp_mask = (all_mlp_metric > threshold)
    
    
    
    for idx in range(len(layers)):
        mlp_mask = (all_mlp_metric[idx] > threshold)
        if f"model.layers.{idx}" in getattr(base_model, 'hf_device_map', {}): 
            compress(pruned_model.model.layers[idx], threshold,
                     all_attn_metric[idx], None, attn_baseline_inp_list[idx], None, model.hf_device_map[f"model.layers.{idx}"], unstr=False, bias=bias)
        else:
            compress(pruned_model.model.layers[idx], threshold, 
                     all_attn_metric[idx], None, attn_baseline_inp_list[idx], None, device, unstr=False, bias=bias)
                
        if f"model.layers.{idx}" in getattr(base_model, 'hf_device_map', {}): 
            compress(pruned_model.model.layers[idx], threshold, 
                     None, mlp_mask, None, mlp_baseline_inp_list[idx], model.hf_device_map[f"model.layers.{idx}"], unstr=False, bias=bias)
        else:
            compress(pruned_model.model.layers[idx], threshold, 
                     None, mlp_mask, None, mlp_baseline_inp_list[idx], device, unstr=False, bias=bias)
                
    pruned_model.config.use_cache = base_model.config.use_cache 
    del attn_metric_list, mlp_metric_list, attn_baseline_inp_list, mlp_baseline_inp_list
    del wrapped_layers, subset, dataloader, inps, outs
    torch.cuda.empty_cache()
    gc.collect()
    
    '''for i in range(32):
        check_layer_compatibility(
            model.model.layers[i].self_attn.q_proj,
            model.model.layers[i].self_attn.k_proj,
            model.model.layers[i].self_attn.v_proj,
            model.model.layers[i].self_attn.o_proj
        )
        print(Fore.RED + f"Layer {i} Checked.")'''
    check_sparsity(pruned_model)
    # import pdb
    # pdb.set_trace()
    return pruned_model, tokenizer  
   
