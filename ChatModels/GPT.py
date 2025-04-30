from .instruction import *
import torch
import openai
from openai import OpenAI
import os
import tiktoken
import time
import json
from time import sleep
from colorama import Fore, init

init(autoreset=True)

class GPTChat:
    def __init__(self, model_name, args, save_mid_json=[]):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.time_stamps = []
        self.ts_mode = args.ts_mode

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        API_KEY = "xxx" 
        openai.api_key = API_KEY
        os.environ["OPENAI_API_KEY"] = API_KEY
        os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"

        self.client = OpenAI()
        self.client.base_url = "xxx"
        self.width = args.width
        
        self.save_mid_json = save_mid_json

    
    def generate_response_api(self, prompt, top_k, max_length=1024, system_message=None, temperature=0.0):
        sys_msg = "You are a professional Python engineer."
        if system_message:
            sys_msg = system_message
        for ti in range(3):
            sleep_interval = 7
            try:
                response = self.client.chat.completions.create(
                    model='gpt-4o-mini',
                    #model='gpt-3.5-turbo',
                    messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                    max_tokens=max_length,  # 调整生成文本的长度
                    temperature=temperature,
                    # top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    #logprobs=True,
                    #top_logprobs=top_k
                )
                message = response.choices[0].message.content
                #log_prob = response.choices[0].logprobs.content  # 是一个length等于top k的list，每个位置是一个list{token: .., logprob:.., bytes:..}

                #log_prob = [[1.0]]
                '''input_token_num = len(self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'}))
                output_token_num = len(self.tokenizer.encode(message, allowed_special={'<|endoftext|>'}))
                self.args.total_input_token_num += input_token_num
                self.args.total_output_token_num += output_token_num'''
            except Exception as e:
                print("GPT Error:", str(e))
                sleep_t = sleep_interval * (ti + 1)
                print(f"get {ti +1}, error: {e}, sleep {sleep_t} seconds")
                sleep(sleep_t)
                continue
            return message
        
    def extract_thoughts(self, response_text, depth):
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
            '''print(Fore.RED + "Extracted lines:")
            print(response_text)
            print(Fore.RED + "Extracted lines end.")'''
        try:
            if response_text.strip()[0] != '[':
                response_text = '[' + response_text + ']'
            
            '''print(Fore.RED + "Extracted lines:")
            print(response_text)
            print(Fore.RED + "Extracted lines end.")'''

            response_text = json.loads(response_text)
            '''print(Fore.RED + "Extracted json lines:")
            print(response_text)
            print(Fore.RED + "Extracted json lines end.")'''
            top_scores = []
            #top_lines = []
            top_lines_text = []
            for i, ele in enumerate(response_text):
                ele_key = ele.keys()
                for key in ele_key:
                    if key == 'Reasonableness':
                        top_scores.append(ele['Reasonableness'])
                    else:
                        clue_content = ele[key]
                        clue_name = key.find('C')
                        clue_name = key[clue_name:]
                        clue_content = clue_name + ': ' + clue_content
                        top_lines_text.append(clue_content + '\n')
                        #print(Fore.RED + clue_content)
                #top_lines.append(self.tokenizer.encode(clue_content + '\n', allowed_special={'<|endoftext|>'}))
        except Exception as e:
            #self.args.failed_json_num += 1
            #top_lines = [self.tokenizer.encode('\n', allowed_special={'<|endoftext|>'}) for i in range(self.width)]
            top_scores = [1.0 for i in range(self.width)]
            top_lines_text = ['\n' for i in range(self.width)]

        '''print(Fore.RED + "Extracted lines:")
        print(top_lines)
        print(Fore.RED + "Extracted lines end.")'''
        return top_lines_text, top_scores


    def get_top_k_rationale_predict(self, state, depth, with_verbal=False):
        with torch.no_grad():
            input_prompt = state

            with_instru_input_prompt = input_prompt + build_intermediate_instruct(depth, self.args.width)

            print('\n-----------------Input (Generate Thought)-----------------')
            print(Fore.GREEN + with_instru_input_prompt)

            response_text = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024, temperature=0.0)
            print('\n-----------------Output (Thought)-----------------')
            print(Fore.YELLOW + response_text)

            top_lines, top_scores = self.extract_thoughts(response_text, depth)

            return top_lines, top_scores
    
    def get_rationale_predicted_sequence(self, state, problem, horizon=None, renewchild_count=0):
        with torch.no_grad():

            input_prompt = state
            previous_thoughts = input_prompt.split('-----Clues-----')[-1]
            
            with_instru_input_prompt = get_reward_instruct(previous_thoughts, problem)

            print('\n-----------------Input with Thought (Generate Code)-----------------')
            print(Fore.GREEN + with_instru_input_prompt)
            
            response_text = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024)

            print('\n-----------------Output (Code)-----------------')
            print(Fore.YELLOW + response_text)
            
            return response_text



class WithProbReturn:
    def __init__(self, sequences, scores, attentions, hidden_states, beam_indices=None, top_tokens=None):
        self.sequences = sequences
        self.scores = scores
        self.attentions = attentions
        self.hidden_states = hidden_states
        self.beam_indices = beam_indices
        self.top_tokens = top_tokens
