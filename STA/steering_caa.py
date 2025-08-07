import sys
import pdb

sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import json
from model_wrapper import (
    LlamaWrapper,
    GemmaWrapper,
)
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from utils.helpers import get_a_b_probs
from utils.tokenize import E_INST
import torch
from dataloader import GenerationDataset
from transformers import StoppingCriteria
import pickle
from colorama import Fore, init
init(autoreset=True)
import re

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def get_his(p_id, test_ranked, k):
    ranked_profiles = test_ranked[p_id][:k]
    
    q_a_history = []
    for idx, sample in enumerate(ranked_profiles):
        line = f"Historical sample {idx}:\n Q: {sample['text']}. \n A: {sample['title']}."
        q_a_history.append(line)
    q_a_history = '\n'.join(q_a_history)
    return q_a_history

def build_rag_instruction(prompt, his, form='python', use_example=False):
    inp = f"Generate a headline for the following article: {prompt}" if not prompt.startswith('Generate') else prompt
    inp += f"For your reference, here are the user's past QA pairs:\n {his}\n"
    if form == 'raw':
        inp += "Please only generate the most suitable one headline, except which no extra text is needed."
    elif form == 'json':
        inp += """\nFormat the output in json format like this:  
```json
{"headline": Your generated headline here}  
```  """
    elif form == 'python':
        inp += """\nFormat the output in python code like this:
```python
print("Your generated headline here")
```"""
    if use_example:
        inp += f"""\nFor your reference, here is an example of a QA pair:\n
            Q: {example['input']}
            A: {example['output']}
        """
    # inp += "\nYour answer here:"

    return inp

def build_zeroshot_instruction(prompt, form='python', use_example=False):
    inp = f"Generate a headline for the following article: {prompt}" if not prompt.startswith('Generate') else prompt
    if form == 'raw':
        inp += "Please only generate the most suitable one headline, except which no extra text is needed."
    elif form == 'json':
        inp += """\nFormat the output in json format like this:  
```json
{"headline": Your generated headline here}  
```  """
    elif form == 'python':
        inp += """\nFormat the output in python code like this:
```python
print("Your generated headline here")
```"""
    if use_example:
        inp += f"""\nFor your reference, here is an example of a QA pair:\n
            Q: {example['input']}
            A: {example['output']}
        """
    # inp += "\nYour answer here:"

    return inp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--mode", type=str, default="toxic")
    parser.add_argument("--model_name", type=str, default="llama-3.1")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--trim", type=float, default=-1)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--system_prompt_back", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="llama-3.1")
    parser.add_argument("--data_name", type=str, default="power-seeking")
    parser.add_argument("--eval_data_name", type=str, default="power-seeking")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/root/paddlejob/workspace/msy/shae/data")
    parser.add_argument("--AB", action="store_true", default=False)
    parser.add_argument("--qa", action="store_true", default=False)
    parser.add_argument("--form", default='raw', type=str)
    
    args = parser.parse_args()
    print(args)

    model = LlamaWrapper(args.model_name_or_path) if args.model_name == "llama-3.1" else GemmaWrapper(args.model_name_or_path)
    tokenizer = model.tokenizer

    dataset = GenerationDataset()
    get_data = dataset.get_data_for_caa_eval
        
    all_vector_dataset, test_ranked = get_data(
        data_path=args.data_path,
        data_name=args.eval_data_name,
    )

    device = model.device
    all_results = {}
    uids = []

    for uid, vector_dataset in all_vector_dataset.items():
        # if (len(uids) >= 2):
        #     break
        uids.append(uid)
        prompt_tokens_list = []
        gt_list = []
        ids = []
        for i in range(len(vector_dataset)):
            ques = vector_dataset[i]["input"]
            gt_list.append(vector_dataset[i]["output"])
            ids.append(vector_dataset[i]['id'])
            if not ques: continue
            if args.model_name=="gemma-2-9b" or args.model_name=="llama-3.1":
                if args.system_prompt != "":
                    if ques is not None:
                        ques = args.system_prompt + " " + ques
                    else:
                        ques = args.system_prompt
                        
                p_id = vector_dataset[i]['id']
                ranked_his = get_his(p_id, test_ranked, 5)
                raw_prompt = build_rag_instruction(ques, ranked_his, form=args.form)
                sys_msg = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>/n{args.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>/n"
                )

                # Prepare the prompt by combining system_message and user prompt
                full_prompt = (
                    sys_msg
                    + "\n"
                    + raw_prompt
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                )
                if i == 0:
                    print(Fore.GREEN + full_prompt)
                prompt_tokens_list.append(tokenizer.encode(full_prompt, return_tensors="pt").to(device))
            else:
                raise NotImplementedError
            
        if args.trim == -1:
            if args.model_name == "gemma-2-9b" or args.model_name=="llama-3.1":
                vector_root = os.path.join(args.data_name, "caa_vector_pt", f"{args.model_name}")
            else:
                vector_root = os.path.join(args.data_path, args.data_name, "caa_vector", f"{args.model_name}_{args.mode}")
        else:
            if args.model_name == "gemma-2-9b-it":
                vector_root = os.path.join(args.data_path, args.data_name, "caa_vector_it", f"{args.model_name}_{args.mode}_trim")
            elif args.model_name == "gemma-2-9b" or args.model_name=="llama-3.1":
                vector_root = os.path.join(args.data_path, args.data_name, "caa_vector_pt", f"{args.model_name}_{args.mode}_trim")
            else:
                vector_root = os.path.join(args.data_path, args.data_name, "caa_vector", f"{args.model_name}_{args.mode}_trim")

        directory = os.path.dirname(args.output_file)  
        if not os.path.exists(directory):  
            os.makedirs(directory)  
        for layer in args.layers:
            print(f"Layer {layer}")
            if args.trim == -1:
                vector_path = os.path.join(
                    vector_root, f"{uid}_{layer}.pt"
                )
            else:
                vector_path = os.path.join(
                    vector_root, f"{layer}_trim{args.trim}.pt"
                )
                print("vector_path: ", vector_path)

            steering_vector = torch.load(vector_path).to(device)
            print("Steering vector path: ",vector_path)
            print("Steering vector: ",steering_vector)
            print("Steering vector == 0: ", (steering_vector == 0).sum())
            print("Steering vector norm: ",steering_vector.norm())
            for multiplier in args.multipliers:
                
                preds = []
                preds_all = []
                model.reset_all()
                print(f"Multiplier {multiplier}")
                # print(f"##########steering_vector\n{steering_vector}##########")

                # model.set_add_activations(
                #     layer, multiplier * steering_vector
                # )
                max_new_tokens=args.max_new_tokens
                print("max_new_tokens: ", max_new_tokens)
                for prompt_tokens in tqdm(prompt_tokens_list, desc=f"Generating... layer-{layer} multiplied by {multiplier}"):
                    prompt_tokens = prompt_tokens.to(device)  
                    # output = model.model.generate(prompt_tokens, max_new_tokens=max_new_tokens)
                    # print(f'##############output:\n{tokenizer.batch_decode(output)[0]}\n##############')
                    # pdb.set_trace()
                    output = model.model.generate(prompt_tokens, max_new_tokens=max_new_tokens)
                    preds_all.append(tokenizer.batch_decode(output)[0]) 
                    output = output[:,prompt_tokens.shape[-1]:]
                    output = tokenizer.batch_decode(output)[0]
                    output = output.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
                    if args.form == 'python':
                        try:
                            output = re.search(r'print\(["\'](.*?)["\']\)', output).group(1)
                        except Exception as e:
                            print(Fore.RED + str(e))
                            output = ""
                    print(Fore.YELLOW + output)
                    preds.append(output)
                print("Without clean_preds!!!")
                # if "exaggerated-safety" not in args.eval_data_name:
                #     preds = clean_preds(preds)
                # preds = clean_preds(preds)
                results = [
                    {"id": ids[idx], "generation": preds[idx], "output": gt_list[idx]} for idx in range(len(preds))
                ]
                all_results[uid] = results
    output_file = args.output_file.replace(".result.json", f"_multiplier{multiplier}.result.json")

    json.dump(all_results, open(output_file, 'w'), indent=4)
    print(f"Output file: {output_file}")
                