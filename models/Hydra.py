from ChatModels import *
from transformers import LongformerPreTrainedModel, LongformerModel, DebertaModel
from colorama import Fore, init
init(autoreset=True)
from .personalized_trainer import *

import os

import torch
from torch import nn

class Hydra:
    def __init__(self, args):
        self.args = args
        
        '''# Base Generator
        print(args.arch)
        if args.arch == "llama3":
            self.generator = LlamaChat(args.arch, args)
        elif args.arch == "deepseek":
            self.generator = DeepSeekChat(args.arch, args)
        elif args.arch == "gemma":
            self.generator = GemmaChat(args.arch, args)
        elif args.arch == "gpt":
            self.generator = GPTChat(args.arch, args)
        elif args.arch == "merge":
            self.generator = MergeChat(args.arch, args)
        elif args.arch == "claude":
            self.generator = ClaudeChat(args.arch, args)
        else:
            raise NotImplementedError
        
        total_params = sum(p.numel() for p in self.generator.model.parameters())
        print(Fore.GREEN + f"#Parameters of ChatModel: {total_params / 1e9:.2f}B")'''
        
        
        # Adapter: Kind of "reward model".
        # Consists of base longformer and #users personalized heads. -> Cannot even save too much heads.
        self.adapter_model = personalization_orm_cls_trainer(args.config)
        print(Fore.GREEN + f"Adapter model loaded.")
        
        # Train adapter: Train the base and the personalized heads using the training data.
        self.adapter_full_train()
        
        # Heads tuning: Train the personalized heads using the privacy data.
        #self.heads_train()
        
    def adapter_full_train(self,):
        self.adapter_model.train_and_eval_seen_test()
        #self.adapter_model.guided_inference() # freeze base, tune heads using the new user history.
    
    def heads_train(self,):
        self.adapter_model.guided_inference() 
        
    