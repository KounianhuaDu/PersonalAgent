from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
import torch
import os
import json
import random
import time
import numpy as np
import csv
from tqdm import tqdm

from .pruning_utils import get_block_pruned_network, get_model, count_params

def prune_model(base_model_addr, sparsity_ratio, calibration_data, save_model_path):
    config = AutoConfig.from_pretrained(base_model_addr)
    model_orig, tokenizer, _ = get_model(base_model=base_model_addr)
    # tokenizer.pad_token = tokenizer.eos_token
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    output_dir = './output_block_sensitivity'
    unsorted_output_path = os.path.join(output_dir, "all_ppl_unsorted.csv")
    sorted_output_path = os.path.join(output_dir, "all_ppl_sorted.csv")
    block_order_path = os.path.join(output_dir, "block_order.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    # data loading
    '''with open(calibration_data_path, 'r') as f:
        traindata = json.load(f)[0]['profile']'''
    traindata = calibration_data
    tokenized_samples, history = [], []
    for line in tqdm(traindata):
        tokenized_sample = tokenizer(
            line['input'] + line['output'],
            return_tensors="pt",
            max_length=512, 
            truncation=True, 
            padding='max_length', 
            padding_side='left'
        )
        tmp_ids = tokenized_sample.input_ids
        tokenized_samples.append(tmp_ids)
    print('Calibration data processed.')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    example_prompts = torch.cat(tokenized_samples, dim=0).to(device)
    
    for block_idx in tqdm(range(model_orig.config.__getattribute__("num_hidden_layers"))):
        csv_log_path = os.path.join(
            './output_block_sensitivity', f"ppl_block{block_idx}_removed.csv"
        )
        '''if os.path.exists(csv_log_path):
            print(f"already computed - {csv_log_path}")
            continue'''

        model = get_block_pruned_network(
            model_orig,
            unimportance_order=[block_idx],
            num_pruned_blocks=1,
            device=device,
        )
        
        # Measure PPL
        t0 = time.perf_counter()
        nlls = []
        with torch.no_grad():
            for j in range(len(traindata)):
                x = example_prompts[j].unsqueeze(0)
                output = model(x)
                lm_logits = output.logits
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = x[:, 1:].contiguous()
                loss = loss_fct(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                nlls.append(loss)

        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())

            # Save
        with open(csv_log_path, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(
                ["removed_block", "ppl_bookcorpus", "num_calib_data", "params"]
            )
            logwriter.writerow(
                [block_idx, ppl, example_prompts.shape[0], count_params(model)]
            )

        print(f"PPL over Bookcorpus {example_prompts.shape[0]} samples: {ppl}")
        print(f"  * time in sec: {time.perf_counter()-t0}")
        del model
        
    unsorted_results = []
    with open(unsorted_output_path, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            ["removed_block", "ppl_bookcorpus", "num_calib_data", "params"]
        )
        for block_idx in range(
            model_orig.config.__getattribute__("num_hidden_layers")
        ):
            csv_log_path = os.path.join(
                output_dir, f"ppl_block{block_idx}_removed.csv"
            )
            with open(csv_log_path, "r") as file:
                next(file)  # pass the header line
                data = [float(i) for i in str(next(file).strip()).split(",")]
                logwriter.writerow(data)
                unsorted_results.append(data)

    sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse=False)

    block_order = []
    with open(sorted_output_path, "w") as logfile, open(
        block_order_path, "w"
    ) as logfile_order:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            ["removed_block", "ppl_bookcorpus", "num_calib_data", "params"]
        )
        logwriter.writerows(sorted_results)
        for data in sorted_results:
            block_order.append(int(data[0]))
        logwriter_order = csv.writer(logfile_order, delimiter=",")
        logwriter_order.writerow(block_order)

    print(f"=== block order removed: {block_order_path}")
    print(block_order)
    print(f"len: {len(block_order)}")
    
    num_pruned_blocks = int(len(block_order) * sparsity_ratio)
    # Load the precomputed block unimportance order
    unimportance_order = []
    with open(block_order_path, "r") as file:
        unimportance_order = [int(i) for i in str(next(file).strip()).split(",")]
    last_block_index = model_orig.config.num_hidden_layers - 1
    keep_block_info = [
        0,
        1,
        2,
        3,
        last_block_index - 1,
        last_block_index,
    ]  # to keep first and last few blocks unpruned
    unimportance_order = [
        idx for idx in unimportance_order if idx not in keep_block_info
    ]
    
    model = get_block_pruned_network(
        model_orig,
        unimportance_order=unimportance_order,
        num_pruned_blocks=num_pruned_blocks,
        device=device,
    )

    '''# Save
    os.makedirs(save_model_path, exist_ok=True)
    model.save_pretrained(save_model_path, max_shard_size="10GB")
    tokenizer.save_pretrained(save_model_path)'''
    return model, tokenizer
        
if __name__ == "__main__":
    base_model_addr = "../model_weights/Meta-Llama-3.1-8B-Instruct"
    sparsity_ratio = 0.2
    calibration_data_path = "./data/LaMP_4/processed/privacy_train.json"
    save_model_path = "./pruned_model/structured"
    
    prune_model(base_model_addr, sparsity_ratio, calibration_data_path, save_model_path)