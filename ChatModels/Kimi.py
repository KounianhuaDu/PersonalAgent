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


class KimiChat:
    def __init__(self, model_name, args):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode

        self.client = OpenAI(
            api_key="sk-6znt5pgAJMQQqruGaYNZ4hE8FhH88z8hiLdzT1AmklcgebsW",
            base_url="https://api.moonshot.cn/v1",
        )
        
        self.width = args.width
        
        self.one_min_request_count = 0

    def generate_response_api(self, prompt, top_k, max_length=1024, system_message=None):
        sys_msg = "You are a helpful code generator that generate code to complete the given problem."

        if system_message:
            sys_msg = system_message

        try:
            response = self.client.chat.completions.create(
                model='moonshot-v1-8k',
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                max_tokens=max_length,  # 调整生成文本的长度
                temperature=0.0,
                # top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                logprobs=True,
                top_logprobs=top_k
            )
            message = response.choices[0].message.content.strip()
            self.one_min_request_count += 1
            print('-------------')
            print(f"kimi requestion one minute nums: {self.one_min_request_count}")
            if self.one_min_request_count == 2:
                time.sleep(60)
                self.one_min_request_count = 0
        except Exception as e:
            print("GPT Error:", str(e))
            assert False, "GPT API error: " + str(e)
        return message, None

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
            #prompt, top_k, max_length=1024, system_message=None
            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, 1)
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
            
            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, 1)

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
