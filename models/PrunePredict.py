from colorama import Fore, init
init(autoreset=True)
import os
import pickle as pkl
import json

#from .structured_prune import prune_model as prune_model_structured
from .Prune_FLAP import prune_flap as prune_model_structured
from .unstructured_prune import prune_model as prune_model_unstructured

from tqdm import tqdm
import random

import torch

class PrunePredict:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        if args.structured:
            self.prune_func = prune_model_structured
        else:
            self.prune_func = prune_model_unstructured
        self.u_ids = []
        with open(os.path.join(args.data_path, args.dataset, 'processed', 'train.pkl'), 'rb') as f:
             self.calibration_data = pkl.load(f)
             self.u_ids = list(self.calibration_data.keys())
        with open(os.path.join(args.data_path, args.dataset, 'processed', 'train_ranked.json'), 'r') as f:
             self.calibration_ranked = json.load(f)
             
        with open(os.path.join(args.data_path, args.dataset, 'processed', 'seen_test.pkl'), 'rb') as f:
            self.test_data = pkl.load(f)
        with open(os.path.join(args.data_path, args.dataset, 'processed', 'seen_test_ranked.json'), 'r') as f:
            self.test_ranked = json.load(f)
        
    def prune_for_one_user(self, u_id):
        data = self.calibration_data[u_id]
        qa_lines = []
        profile_lines = []
        for line in data:
            profile_lines += line['profile']
            ranked_his = self.get_calibration_his(line['id'], self.args.k)
            raw_prompt = self.build_rag_instruction(line['input'], ranked_his)
            line['rag_input'] = raw_prompt
            qa_lines.append(line)
            
        qa_lines = [{'input': line['rag_input'], 'output': line['output']} for line in qa_lines] 
        profile_lines = [{'input': line['text'], 'output': line['title']} for line in profile_lines]
        print(Fore.GREEN + f'Start to prune for user {u_id}.')
        print(f"QA lines: {len(qa_lines)}")
        print(f"Profile lines: {len(profile_lines)}")
        
        if len(qa_lines)>128:
            calibration_lines = random.sample(qa_lines, 128)
        else:
            calibration_lines = qa_lines + random.sample(profile_lines, min(128 - len(qa_lines), len(profile_lines)))
        
        #calibration_lines = random.sample(profile_lines, min(128, len(profile_lines)))
        print(f"Sampled {len(calibration_lines)} for calibration.")
        
        self.model, self.tokenizer = self.prune_func(self.args.base_model_addr, self.args.sparsity, calibration_lines, '')
        print(self.model)
        exit()
        total_params = sum(p.numel() for p in self.model.parameters())
        print(Fore.GREEN + 'Pruning ends.')
        print(Fore.GREEN + f'#Parameters of pruned models: {total_params / 1e9:.2f}B.')
        
        test_lines = self.test_data[u_id]
        outs = []
        for test_line in tqdm(test_lines):
            output_dict = self.generate(test_line)
        outs.append(output_dict)
        return outs
            
        
    def generate_response_api(
        self,
        prompt: str,
        top_k: int,
        max_length: int = 64,
        system_message: str = None,
        temperature: float = 0,
    ):
        
        full_prompt = prompt
        
        model_inputs = self.tokenizer([full_prompt], return_tensors="pt").to(
            self.model.device
        )
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=self.model.device
        )
        # Generate the response
        #print(model_inputs.input_ids)
        #print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        '''print(Fore.GREEN + f'#Parameters of pruned models: {total_params / 1e9:.2f}B.')
        exit()'''
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the response
        message = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return message
        
        
        
    def generate(self, problem_instance):
        p_id = problem_instance['id']
        
        if self.args.algo == 'zeroshot':
            raw_prompt = self.build_zeroshot_instruction(problem_instance['input'])
        elif self.args.algo == 'rag':
            ranked_his = self.get_his(p_id, self.args.k)
            raw_prompt = self.build_rag_instruction(problem_instance['input'], ranked_his)
        
        #print(raw_prompt)
        output = self.generate_response_api(raw_prompt, top_k=1)
        print(output)
        print(output.split('\n')[0])
        #exit()
        output_dict = {
            'id': p_id,
            'generation': output,
            'output': problem_instance['output']
        }

        return output_dict

    def get_his(self, p_id, k):
        ranked_profiles = self.test_ranked[p_id][:k]
        
        q_a_history = []
        for idx, sample in enumerate(ranked_profiles):
            line = f"Historical sample {idx}:\n Q: {sample['text']}. \n A: {sample['title']}."
            q_a_history.append(line)
        q_a_history = '\n'.join(q_a_history)
        return q_a_history
    
    def get_calibration_his(self, p_id, k):
        ranked_profiles = self.calibration_ranked[p_id][:k]
        
        q_a_history = []
        for idx, sample in enumerate(ranked_profiles):
            line = f"Historical sample {idx}:\n Q: {sample['text']}. \n A: {sample['title']}."
            q_a_history.append(line)
        q_a_history = '\n'.join(q_a_history)
        return q_a_history
    
    def extract(self, response_text):
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        try:
            if response_text.strip()[0] != '[':
                response_text = '[' + response_text + ']'
            
            response_text = json.loads(response_text)
            top_lines_text = []
            for sample in response_text:
                ele_key = list(sample.keys())[0]
                top_lines_text.append(sample[ele_key])
        except Exception as e:
            top_lines_text = ['\n' for i in range(3)]

        return top_lines_text
    
    def build_zeroshot_instruction(self, prompt):
        if self.args.dataset == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.dataset == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.dataset == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.dataset == "LaMP_4":
            inp = f"Generate a headline for the following article: {prompt}\n"
            inp += "Please only generate the most suitable one headline, except which no extra text is needed."
        elif self.args.dataset == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.args.dataset == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        return inp
    
    def build_rag_instruction(self, prompt, his):
        if self.args.dataset == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.dataset == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.dataset == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.dataset == "LaMP_4":  
            inp = f"Generate a headline for the following article: {prompt}"
            inp += f"For your reference, here are the user's past QA pairs:\n {his}\n"
            inp += "Please ONLY generate the most suitable one headline, except which NO EXTRA TEXT is needed."
        elif self.args.dataset == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.args.dataset == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        return inp
    
    