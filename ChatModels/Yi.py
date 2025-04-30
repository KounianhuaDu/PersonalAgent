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

class YiChat:
    def __init__(self, model_name, args):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode

        self.API_BASE = "https://api.lingyiwanwu.com/v1"
        self.API_KEY = "599f8416a9e048a3b9306bbbf47e857b"
        self.client = OpenAI(
            api_key=self.API_KEY,
            base_url=self.API_BASE
        )
        self.args = args

        self.width = args.width
        
        self.save_mid_json = []
    
    def generate_response_api(self, prompt, system_message=None):
        sys_msg = "You are a professional Python engineer."
        if system_message:
            sys_msg = system_message
        message = self.client.chat.completions.create(
            model="yi-34b-chat-0205",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
                ]
        )
        return message.choices[0].message.content.strip()
    
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

            response_text = self.generate_response_api(with_instru_input_prompt)
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
            
            response_text = self.generate_response_api(with_instru_input_prompt)

            print('\n-----------------Output (Code)-----------------')
            print(Fore.YELLOW + response_text)
            
            return response_text
