from colorama import Fore, init
init(autoreset=True)
import os
import pickle as pkl
import json

#from .structured_prune import prune_model as prune_model_structured
from .Prune_FLAP_group import prune_flap as prune_model_structured
from .unstructured_prune import prune_model as prune_model_unstructured

from tqdm import tqdm
import random
import torch
from transformers import DebertaV2Tokenizer, DebertaV2Model
from pathlib import Path
import argparse
from sklearn.cluster import KMeans
import numpy as np
import sys
import re
sys.path.append('..')
random.seed(42)

class PrunePredict:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.format = 'python'
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
            
        # exit()
        self.cluster_dir = "./cluster_data/prune_python_data_rag.json"
        if not os.path.exists(self.cluster_dir):
            self.cluster(self.calibration_data)
        with open(self.cluster_dir, 'r') as f:
            self.prune_data = json.load(f)
        example_user = random.choice(self.u_ids)
        self.example = random.choice(self.calibration_data[example_user])
        self.example['output'] = self.build_answer(self.example['output'])
        # print(self.example)
        # import pdb
        # pdb.set_trace()
        
        
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
        print(qa_lines[0]['input'])
        profile_lines = [{'input': self.build_zeroshot_instruction(line['text']), 'output': line['title']} for line in profile_lines]
        print(profile_lines[0]['input'])
        print(Fore.GREEN + f'Start to prune for user {u_id}.')
        print(f"QA lines: {len(qa_lines)}")
        print(f"Profile lines: {len(profile_lines)}")
        
        if len(qa_lines)>128:
            calibration_lines = random.sample(qa_lines, 128)
        else:
            calibration_lines = qa_lines + random.sample(profile_lines, min(128 - len(qa_lines), len(profile_lines)))
        
        #calibration_lines = random.sample(profile_lines, min(128, len(profile_lines)))
        print(f"Sampled {len(calibration_lines)} for calibration.")
        # self.model, self.tokenizer = self.prune_func(self.args.base_model_addr, self.args.sparsity, calibration_lines, '')
        self.model, self.tokenizer = self.prune_func(self.args.base_model_addr, self.args.sparsity, self.prune_data[str(u_id)], '')
        # print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(Fore.GREEN + 'Pruning ends.')
        print(Fore.GREEN + f'#Parameters of pruned models: {total_params / 1e9:.2f}B.')
        # exit()
        
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
        max_length: int = 128,
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
        use_example = True if self.format == 'json' else False
        if self.args.algo == 'zeroshot':
            raw_prompt = self.build_zeroshot_instruction(problem_instance['input'], use_example=use_example)
        elif self.args.algo == 'rag':
            ranked_his = self.get_his(p_id, self.args.k)
            raw_prompt = self.build_rag_instruction(problem_instance['input'], ranked_his)
        
        print(Fore.GREEN + raw_prompt)
        output = self.generate_response_api(raw_prompt, top_k=1)
        print(Fore.YELLOW + output)
        # print(output.split('\n')[0])
        #exit()
        if self.format == 'json':
            if '```json' in output:
                output = output.split('```json')[1].split('```')[0]
            try:
                output = json.loads(output)
                output = output['headline'] if 'headline' in output else output['Headline']
            except Exception as e:
                print(e)
                output = ""
        elif self.format == 'python':
            try:
                output = re.search(r'print\(["\'](.*?)["\']\)', output).group(1)
            except Exception as e:
                print(e)
                output = ""
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
    
    def build_zeroshot_instruction(self, prompt, use_example=False):
        if self.args.dataset == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.dataset == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.dataset == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.dataset == "LaMP_4":
            inp = f"Generate a headline for the following article: {prompt}" if not prompt.startswith('Generate') else prompt
            if self.format == 'raw':
                inp += "Please only generate the most suitable one headline, except which no extra text is needed."
            elif self.format == 'json':
                inp += """\nFormat the output in json format like this:  
```json
{"headline": Your generated headline here}  
```  """
            elif self.format == 'python':
                inp += """\nFormat the output in python code like this:
```python
print("Your generated headline here")
```"""

            if use_example:
                inp += f"""\nFor your reference, here is an example of a QA pair:\n
                 Q: {self.example['input']}
                 A: {self.example['output']}
                """
            inp += "\nYour answer here:"
        elif self.args.dataset == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.args.dataset == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        return inp
    
    def build_rag_instruction(self, prompt, his, use_example=False):
        if self.args.dataset == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.dataset == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.dataset == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.dataset == "LaMP_4":  
            inp = f"Generate a headline for the following article: {prompt}" if not prompt.startswith('Generate') else prompt
            inp += f"For your reference, here are the user's past QA pairs:\n {his}\n"
            if self.format == 'raw':
                inp += "Please only generate the most suitable one headline, except which no extra text is needed."
            elif self.format == 'json':
                inp += """\nFormat the output in json format like this:  
```json
{"headline": Your generated headline here}  
```  """
            elif self.format == 'python':
                inp += """\nFormat the output in python code like this:
```python
print("Your generated headline here")
```"""
            if use_example:
                inp += f"""\nFor your reference, here is an example of a QA pair:\n
                 Q: {self.example['input']}
                 A: {self.example['output']}
                """
            inp += "\nYour answer here:"
        elif self.args.dataset == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.args.dataset == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        return inp
    
    def build_answer(self, answer):
        if self.format == 'raw':
            return answer
        elif self.format == 'json':
            return f"```json\n{answer}\n```"
        elif self.format == 'python':
            return f"```python\nprint(\"{answer}\")\n```"
    
    def cluster(self, calibration_data):
        # Step 1: Load the DeBERTa-v3-base tokenizer and model
        tokenizer = DebertaV2Tokenizer.from_pretrained('../model_weights/deberta-v3-large')
        model = DebertaV2Model.from_pretrained('../model_weights/deberta-v3-large').cuda()
        task_name = 'cluster_data'

        all_user_emb = []
        processed_data = dict()
        for user in self.u_ids:
            history_embeddings_list = []
            visible_history_list = []
            qa_lines = []
            ids = []
            for samples in calibration_data[user]:
                ids.append(samples['id'])
                # profile_lines += line['profile']
                ranked_his = self.get_calibration_his(samples['id'], self.args.k)
                raw_prompt = self.build_rag_instruction(samples['input'], ranked_his)
                # raw_prompt = self.build_zeroshot_instruction(samples['input'])
                samples['rag_input'] = raw_prompt
                qa_lines.append(samples)
                for s in samples['profile']:
                    if s['id'] not in ids:
                        ids.append(s['id'])
                        visible_history_list.append(s)
            qa_lines = [{'input': line['rag_input'], 'output': self.build_answer(line['output'])} 
                        for line in qa_lines] 
            profile_lines = [{'input': self.build_zeroshot_instruction(line['text']), 'output': self.build_answer(line['title'])} 
                             for line in visible_history_list]
            if len(qa_lines)>128:
                calibration_lines = random.sample(qa_lines, 128)
            else:
                calibration_lines = qa_lines + random.sample(profile_lines, min(128 - len(qa_lines), len(profile_lines)))
            processed_data[user] = calibration_lines
            # user_nl_history_list_batched = split_batch(user_nl_history_list, batch_size)

            for batch in tqdm(calibration_lines):

                with torch.no_grad():
                    inputs = tokenizer(batch['input'] + batch['output'], return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
                    outputs = model(**inputs)

                    last_hidden_states = outputs.last_hidden_state
                    # Compute attention mask
                    attention_mask = inputs['attention_mask']

                    # Expand attention mask so it has same size as last_hidden_states, for broadcasting purposes
                    attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

                    # Multiply last hidden states by attention mask, then sum and divide by number of tokens
                    masked_hidden_states = last_hidden_states * attention_mask
                    summed = torch.sum(masked_hidden_states, 1)
                    count = torch.clamp(attention_mask.sum(1), min=1e-9)
                    mean_pooled = summed / count

                history_embeddings_list.append(mean_pooled.cpu())

            history_embedding_concat = torch.cat(history_embeddings_list, dim=0).cpu().mean(dim=0, keepdim=True)
            all_user_emb.append(history_embedding_concat)

        all_user_emb = torch.cat(all_user_emb, dim=0)
        # print(all_user_emb.size())

        Path(f'./{task_name}/').mkdir(parents=True, exist_ok=True)

        torch.save(all_user_emb, f'./{task_name}/user_history_emb.pt')


        emb = all_user_emb.numpy()

        k=50
        kmeans = KMeans(n_clusters=k, random_state=0, max_iter=3000).fit(emb)
        labels = kmeans.labels_

        prune_data = dict()

        for i in range(k):
            cluster_indices = np.where(labels == i)[0]
            all_data = []
            for idx in cluster_indices:
                uid = self.u_ids[idx]
                all_data += processed_data[uid]
            for idx in cluster_indices:
                uid = self.u_ids[idx]
                prune_data[uid] = {'self_data': processed_data[uid], 'cluster_data': all_data}
            print(len(all_data), len(cluster_indices))
            
        with open(self.cluster_dir, 'w') as f:
            json.dump(prune_data, f, indent=4)

        print('Clustering done!')
        
        