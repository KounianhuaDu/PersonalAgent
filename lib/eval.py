# Import necessary modules
import time
import torch
import torch.nn as nn
import sys
import json
from colorama import Fore, init
import evaluate

# Import get_loaders function from data module within the same directory
from .data import get_loaders 
from ast_checker import ast_checker

init(autoreset=True)

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, eval_dataset, device=torch.device("cuda:0")):
    # Set dataset 
    accs = []
    times = []
    for dataset in eval_dataset:
        print(f"Evaluating {dataset}")
        _, testdata = get_loaders(dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
        with torch.no_grad():
            st = time.time()
            if dataset == 'xlam':
                acc = eval_acc_xlam(model, tokenizer, xlamdata, device)
            elif dataset == 'bfcl':
                acc = eval_acc_bfcl(model, tokenizer, xlamdata, device)
            elif dataset == 'lamp':
                acc = eval_acc_lamp(model, tokenizer, xlamdata, device)
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            accs.append(acc)
            times.append(time.time() - st)
                
    return accs, times

def eval_acc_lamp(model, tokenizer, testdata, device=None):
    testenc, gts = testdata
    preds = []
    labels = []
    for query, gt in zip(testenc, gts):
        print('=' * 100)
        query = query.to(device)
        print(query.shape)
        attention_mask = torch.ones(
                query.shape, dtype=torch.long, device=device
            )
        lm_logits = model.generate(
            query, 
            max_new_tokens=128,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,)
        s = tokenizer.decode(lm_logits[0][query.shape[1]:], skip_special_tokens=True)
        print(Fore.YELLOW + s)
        
        preds.append(s.strip())
        labels.append([gt.strip()])
    rouge_metric = evaluate.load('rouge')
    result_rouge = rouge_metric.compute(predictions=preds, references=labels)
    result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
    print(result)
    return result["rouge"]


def eval_acc_xlam(model, tokenizer, testdata, device=None):
    testenc, gts = testdata
    res = []
    for query, gt in zip(testenc, gts):
        print('=' * 100)
        query = query.to(device)
        print(query.shape)
        attention_mask = torch.ones(
                query.shape, dtype=torch.long, device=device
            )
        lm_logits = model.generate(
            query, 
            max_new_tokens=256,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,)
        # lm_logits = model(query).logits
        print(lm_logits.shape)
        s = tokenizer.decode(lm_logits[0][query.shape[1]:], skip_special_tokens=True)
        # s = tokenizer.batch_decode(lm_logits.argmax(-1), skip_special_tokens=True)[0]
        print(Fore.YELLOW + s)
        try:
            if '```json' in s:
                s = s.split('```json')[1].split('```')[0]
            elif '```' in s:
                s = s.split('```')[1]
            s = s.replace("\n", "")
            s = s.replace("\_", "_")
            s = s.replace("\\", "") 
            s = s.replace("/n", "")
            if not s.startswith('['):
                s = '[' + s
            if not s.endswith(']'):
                s += ']'
            content = json.loads(s)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            content = []
        print(gt)
        res.append(f1_score_with_order(content, gt))
        print(res[-1])
    
    return sum(res) / len(res)

def eval_acc_bfcl(model, tokenizer, testdata, device=None):
    testenc, gts, funcs = testdata
    res = []
    for query, gt, func in zip(testenc, gts, funcs):
        print('=' * 100)
        query = query.to(device)
        print(query.shape)
        attention_mask = torch.ones(
                query.shape, dtype=torch.long, device=device
            )
        lm_logits = model.generate(
            query, 
            max_new_tokens=256,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,)
        # lm_logits = model(query).logits
        print(lm_logits.shape)
        s = tokenizer.decode(lm_logits[0][query.shape[1]:], skip_special_tokens=True)
        # s = tokenizer.batch_decode(lm_logits.argmax(-1), skip_special_tokens=True)[0]
        print(Fore.YELLOW + s)
        try:
            if '```json' in s:
                s = s.split('```json')[1].split('```')[0]
            elif '```' in s:
                s = s.split('```')[1]
            s = s.replace("\n", "")
            s = s.replace("\_", "_")
            s = s.replace("\\", "") 
            s = s.replace("/n", "")
            if not s.startswith('['):
                s = '[' + s
            if not s.endswith(']'):
                s += ']'
            content = json.loads(s)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            content = []

        score = ast_checker(func, content, gt, "Python", "BFCL_v3_parallel_multiple", "llama3")
        res.append(score['valid'])
    
    return sum(res) / len(res)
def f1_score_with_order(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return 0

    min_len = min(len(pred), len(gt))
    matches = sum(1 for p, g in zip(pred, gt) if p == g)

    precision = matches / len(pred)
    recall = matches / len(gt)
    f = 2 * precision * recall / (precision + recall + 1e-9)
    return f        

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)


        # print ("nlls",nlls)
        sys.stdout.flush()

    
    print ('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

