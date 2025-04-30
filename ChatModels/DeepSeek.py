# -*- coding:utf-8 _*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
from .instruction import *
from typing import List, Literal, TypedDict
from colorama import Fore, Back, Style, init
from vllm import LLM, SamplingParams
import os
from peft import PeftModel

init(autoreset=True)

class DeepSeekChat:
    def __init__(self, model_name, args):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.model_path = os.path.join(args.modelweight, "deepseek-coder-6.7b-instruct")
        
        if args.vllm:
            self.model = LLM(
                model=self.model_path, 
                tensor_parallel_size=1, 
                trust_remote_code=True, 
                max_model_len=21552, 
                gpu_memory_utilization=0.9
                )
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=False, padding_side="left"
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map='auto')
            
            if args.lora:
                from peft import LoraConfig, TaskType, get_peft_model

                if args.lora_target_modules == "all":
                    lora_target_modules = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
                elif args.lora_target_modules.lower() == "default":
                    lora_target_modules = None
                else:
                    lora_target_modules = args.lora_target_modules.split(",")

                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False,
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=lora_target_modules
                )
                self.model = get_peft_model(self.model, peft_config)
                print(Fore.RED + "Set Lora model.")
            if args.resume:
                print(Fore.RED + "RESUME")
                #self.model.load_state_dict(
                #    torch.load(
                #        os.path.join(args.model_archive, "policy.pt"),
                #        map_location="cpu", weights_only=False)["state"])
                #if args.lora: self.model = self.model.merge_and_unload()
                #print(f"loaded pre-trained weights from {args.model_archive}.")
                resume_path = os.path.join(args.tuned_path, args.model_file)
                check_point_files = os.listdir(resume_path)
                check_points = []
                for check in check_point_files:
                    if check.startswith('check'):
                        check_points.append(check)
                check_point = os.path.join(resume_path, check_points[-1])

                self.model = PeftModel.from_pretrained(self.model, check_point)
                self.model = self.model.merge_and_unload()
                print(Fore.RED + f"Tuned model loaded from {check_point}.")
                time.sleep(3)

        self.terminal_token = self.tokenizer.eos_token
        self.width = args.width
        # Cache initialization (if needed)
        
        self.args = args

    def generate_response_api(
            self, 
            prompt, 
            top_k, 
            max_length=1024, 
            system_message=None, 
            temperature=0.01
        ):
        sys_msg = "You are a professional Python engineer."
        print(Fore.GREEN + sys_msg + prompt)

        if system_message:
            sys_msg = system_message

        chat = f"<｜begin▁of▁sentence｜>{sys_msg}\nUser: {prompt}\n\nAssistant: "

        if self.args.vllm:
            output = self.model.generate(
                prompts=chat,
                sampling_params=SamplingParams(
                    temperature=temperature,
                    top_k=top_k,
                    max_tokens=max_length
                )
            )

            log_probs_for_generated_tokens = (
                None  # Initialize to handle cases where it's not needed
            )
            message = output[0].outputs[0].text
        else:
            model_inputs = self.tokenizer([chat], return_tensors="pt").to(self.model.device)
            input_ids = self.tokenizer.encode(chat,return_tensors='pt')
            attention_mask = torch.ones(input_ids.shape,dtype=torch.long, device=self.model.device)
            # Generate the response
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                pad_token_id=self.tokenizer.eos_token_id   # Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # Decode the response
            message = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            log_probs_for_generated_tokens = (
                None  # Initialize to handle cases where it's not needed
            )

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

            response_text = self.generate_response_api(
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

            response_text = self.generate_response_api(
                with_instru_input_prompt, top_k=1
            )

            print("\n-----------------Output (Code)-----------------")
            print(Fore.YELLOW + response_text)

            return response_text
