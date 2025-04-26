# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import json
from datasets import load_dataset
from instruction import build_question_instruct

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    traindata = load_dataset('../../data/c4', data_files={'train': 'c4-train.00000-of-01024.json'}, split='train')
    valdata = load_dataset('../../data/c4', data_files={'validation': 'c4-validation.00000-of-00008.json'}, split='validation')

    # Generate samples from training set
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

def get_xlam(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('json', data_files={'train':"../Multi-Teacher-Tree-Alignment-master-3/data/xlam-function-calling-60k/train_data.json"}, split='train')
    valdata = load_dataset('json', data_files={'validation':"../Multi-Teacher-Tree-Alignment-master-3/data/xlam-function-calling-60k/test_data.json"}, split='validation')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - 1)
        train_item = build_question_instruct(traindata[i]['query'], traindata[i]['tools'], "", 'xlam', train=True)
        trainenc = tokenizer(train_item + traindata[i]["answers"], return_tensors='pt', max_length=seqlen, truncation=True, padding='max_length', padding_side='left')
        inp = trainenc.input_ids
        tar = inp.clone()
        query_len = tokenizer(train_item, return_tensors='pt').input_ids.shape[1]
        tar[:, :query_len] = -100
        trainloader.append((inp, tar))

    i = random.randint(0, len(traindata) - 1)
    demo_string = "\nHere are provided examples for your reference.\n"
    demo_string += f"""\n# EXAMPLE #:\n# USER REQUEST #:\n {traindata[i]["query"]}\n# RESULT #:\n ```json\n{traindata[i]["answers"]}\n```"""
    val_data = [build_question_instruct(valdata[i]['query'], valdata[i]['tools'], demo_string, 'xlam', train=False) for i in range(len(valdata))]
    print(val_data[0])
    sys_msg = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a great planner that generates plan to complete the given problem. <|eot_id|><|start_header_id|>user<|end_header_id|>\n")
    full_prompt = [
            sys_msg
            + "\n"
            + val_item
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for val_item in val_data
    ]
    valenc = [tokenizer(val_item, return_tensors='pt').input_ids for val_item in full_prompt]
    # valenc = TokenizerWrapper(valenc)
    val_gt = [json.loads(valdata[i]['answers']) for i in range(len(valdata))]
    return trainloader, (valenc, val_gt)

def get_bfcl(nsamples, seed, seqlen, tokenizer):
    # traindata = []
    # valdata = load_dataset('json', data_files={'validation':"../Multi-Teacher-Tree-Alignment-master-3/data/bfcl/BFCL_v3_parallel_multiple.json"}, split='validation')
    # gt = load_dataset('json', data_files={'gt':"../Multi-Teacher-Tree-Alignment-master-3/data/bfcl/possible_answer/BFCL_v3_parallel_multiple.json"}, split='gt')
    valdata = []
    gt = []
    with open("../Multi-Teacher-Tree-Alignment-master-3/data/bfcl/BFCL_v3_parallel_multiple.json", 'r') as f:
        for line in f:
            valdata.append(json.loads(line))
    with open("../Multi-Teacher-Tree-Alignment-master-3/data/bfcl/possible_answer/BFCL_v3_parallel_multiple.json", 'r') as f:
        for line in f:
            gt.append(json.loads(line))
    
    random.seed(seed)
    trainloader = []
    
    demo_string = "\nHere are provided examples for your reference.\n"
    demo = {
        "user_request": "Can I find the dimensions and properties of a triangle, if I know its three sides are 5 units, 4 units and 3 units long?",
        "result": '[{"triangle_properties.get": {"side1": 5, "side2": 4, "side3": 3, "get_area": "" or true, "get_perimeter": "" or true, "get_angles": "" or true}}]'
    }
    demo_string += f"""\n# EXAMPLE #:\n# USER REQUEST #:\n {demo['user_request']}\n# RESULT #:\n ```json\n{demo['result']}\n```"""
    val_data = [build_question_instruct(valdata[i]['question'][0][0]['content'], json.dumps(valdata[i]['function']), demo_string, 'bfcl', train=False) 
                for i in range(len(valdata))]
    print(val_data[0])
    sys_msg = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a great planner that generates plan to complete the given problem. <|eot_id|><|start_header_id|>user<|end_header_id|>\n")
    full_prompt = [
            sys_msg
            + "\n"
            + val_item
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for val_item in val_data
    ]
    valenc = [tokenizer(val_item, return_tensors='pt').input_ids for val_item in full_prompt]
    # valenc = TokenizerWrapper(valenc)
    val_gt = [gt[i]['ground_truth'] for i in range(len(gt))]
    func = [valdata[i]['function'] for i in range(len(valdata))]
    return trainloader, (valenc, val_gt, func)

def get_lamp(nsamples, seed, seqlen, tokenizer):
    train_questions = load_dataset('json', data_files={'train_question':"../PublicEval/LaMP_4/train/train_questions.json"}, split='train_question')
    train_outputs = load_dataset('json', data_files={'train_output':"../PublicEval/LaMP_4/train/train_outputs.json"}, split='train_output')[0]
    dev_questions = load_dataset('json', data_files={'dev_question':"../PublicEval/LaMP_4/valid/dev_questions.json"}, split='dev_question')
    dev_outputs = load_dataset('json', data_files={'dev_output':"../PublicEval/LaMP_4/valid/dev_outputs.json"}, split='dev_output')[0]

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - 1)
        train_item = train_questions[i]['input']
        trainenc = tokenizer(train_item + train_outputs["golds"][i]["output"], return_tensors='pt', max_length=seqlen, truncation=True, padding='max_length', padding_side='left')
        assert train_questions[i]['id'] == train_outputs['golds'][i]['id']
        inp = trainenc.input_ids
        tar = inp.clone()
        query_len = tokenizer(train_item, return_tensors='pt').input_ids.shape[1]
        tar[:, :query_len] = -100
        trainloader.append((inp, tar))

    val_data = [dev_questions[i]['input'] for i in range(500)]
    print(val_data[0])
    valenc = [tokenizer(val_item, return_tensors='pt').input_ids for val_item in val_data]
    # valenc = TokenizerWrapper(valenc)
    val_gt = [dev_outputs['golds'][i] for i in range(500)]
    return trainloader, (valenc, val_gt)


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "xlam" in name:
        return get_xlam(nsamples, seed, seqlen, tokenizer)
    if 'bfcl' in name:
        return get_bfcl(nsamples, seed, seqlen, tokenizer)
    if 'lamp' in name:
        return get_lamp(nsamples, seed, seqlen, tokenizer)
    