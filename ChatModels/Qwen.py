# -*- coding:utf-8 _*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from .instruction import *
from colorama import Fore, init
from vllm import LLM, SamplingParams
import os


class QwenChat:
    def __init__(self, model_name, args):
        use_vllm = args.vllm
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.max_length = args.max_length
        self.model_path = os.path.join(args.modelweight, "Qwen2.5-7B-Instruct")
        
        if use_vllm:
            self.model = LLM(
                model=self.model_path, trust_remote_code=True
            )
            self.tokenizer = self.model.get_tokenizer()
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=False, padding_side="left"
            )
            self.tokenizer.pad_token_id = 0  # unk. to be different from the eos token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(
                self.device
            )

        self.terminal_token = self.tokenizer.eos_token
        

        self.width = args.width
        

    def generate_response_api(
        self,
        prompt: str,
        top_k: int,
        max_length: int = 2048,
        system_message: str = None,
        temperature: float = 0.5,
    ):
        if self.args.vllm:
            # Generate the response
            output = self.model.generate(
                prompt,
                # vllm get logits/ log_probs
                sampling_params=SamplingParams(
                    temperature=temperature,
                    top_k=top_k,
                    max_tokens=max_length,
                ),
            )
            log_probs_for_generated_tokens = (
                None  # Initialize to handle cases where it's not needed
            )
            message = output[0].outputs[0].text
        else:
            raise NotImplementedError

        
        return message, (
            log_probs_for_generated_tokens.tolist()
            if log_probs_for_generated_tokens is not None
            else None
        )

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
            '''encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # Decode input tokens to get the input prompt
            input_prompt = self.tokenizer.decode(input_ids[0].tolist())'''
            input_prompt = state

            with_instru_input_prompt = input_prompt + build_intermediate_instruct(
                depth, self.args.width
            )

            print("\n-----------------Input (Generate Thought)-----------------")
            print(Fore.GREEN + with_instru_input_prompt)

            response_text, log_probs = self.generate_response_api(
                with_instru_input_prompt, top_k=1
            )
            print('\n-----------------Output (Thought)-----------------')
            print(Fore.YELLOW + response_text)

            top_lines, top_scores = self.extract_thoughts(response_text, depth)

            return top_lines, top_scores

    def get_rationale_predicted_sequence(
        self, state, problem, max_length=None, renewchild_count=0
    ):
        with torch.no_grad():
            '''
            input_ids = state  # as a list
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            input_prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            '''
            input_prompt = state
            previous_thoughts = input_prompt.split("-----Clues-----")[-1]

            with_instru_input_prompt = get_reward_instruct(previous_thoughts, problem)

            print(
                "\n-----------------Input with Thought (Generate Code)-----------------"
            )
            print(Fore.GREEN + with_instru_input_prompt)

            response_text, log_probs = self.generate_response_api(
                with_instru_input_prompt, top_k=1
            )

            print("\n-----------------Output (Code)-----------------")
            print(Fore.YELLOW + response_text)

            return response_text
