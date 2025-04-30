import os
import sys

sys.path.append("..")

import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from accelerate import Accelerator

import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict
import json


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## Backbone LLM
    parser.add_argument("--arch", default='llama3')

    ## Seed
    parser.add_argument("--seed", type=int, default=42)

    ## Lora config
    parser.add_argument("--lora_r", type=int, default=8, help="Lora R.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Lora alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Lora dropout.")

    ## Training setup
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--total_batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=65536)
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model 8 bit.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)

    parser.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint name. Enable only if resume is true.")

    ## Path
    parser.add_argument("--modelweight", default="../model_weights", help="Path of the original model weights.")
    parser.add_argument("--model_name", default="Meta-Llama-3.1-8B-Instruct", help="Path of the original model weights.")
    parser.add_argument("--model_out_root", default="../tuned_models", help="Path to save the tuned model weights.")
    parser.add_argument("--train_data_root", default="../train_data/final", help="Path of the training data.")

    parser.add_argument("--data_file", help="Path of the training data file.", required=True)
    parser.add_argument("--data_name", help="Name of the training data file.", required=True)
    
    parser.add_argument("--tuned_model_file", help="Name of the newly-saved checkpoint.", required=True)

    args = parser.parse_args()

    print(args)

    # Seed Config
    set_seed(42)

    # Model loading
    if args.arch in ["llama3", "deepseek", "gemma"]:
        args.modelpath = os.path.join(args.modelweight, args.model_name)
    else:
        raise NotImplementedError

    # args.load_in_8bit
    model = AutoModelForCausalLM.from_pretrained(
        args.modelpath,
        low_cpu_mem_usage=True,
        device_map={"": Accelerator().local_process_index},
        # device_map="auto",
        trust_remote_code=True,
        #quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    # Resume from a checkpoint if needed.
    if args.resume:
        if not args.checkpoint:
            checks = os.listdir(args.model_out_root)
            check_points = []
            for check in checks:
                if check.startswith("check"):
                    check_points.append(check)
            args.checkpoint = check_points[-1]

        new_model = os.path.join(args.model_out_root, args.checkpoint)
        merge_model = PeftModel.from_pretrained(model, new_model)
        model = merge_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.modelpath)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare dataset
    data_path = {
        "train": os.path.join(args.train_data_root, args.data_name)
    }
    dataset = load_dataset("json", data_files=data_path)

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        if "deepseek" in args.arch:
            text = f"""<|begin▁of▁sentence|>User: {samples['input']}
Assistant: {samples['output']}<|end▁of▁sentence|>"""
        elif "gemma" in args.arch:
            text = f"""<bos><start_of_turn>user
{samples["input"]}<end_of_turn>
<start_of_turn>model{samples['output']}"""
        elif args.arch == "llama3":
            if args.data_file == 'xlam':
                text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>/nYou are a helpful planning assistant, and you should help to generate plans correctly and efficiently.<|eot_id|><|start_header_id|>user<|end_header_id|>/n{samples['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>/n{samples['output']}<|eot_id|>
                """
            else:
                text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>/nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>/n{samples['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>/n{samples['title']}<|eot_id|>
                """
        return {"text": text}

    train_dataset = dataset["train"][4]['profile']
    train_dataset = Dataset.from_list(train_dataset).map(return_prompt_and_responses)

    # Training config
    BATCH_SIZE = min(args.total_batch_size, args.train_size)
    EPOCHS = args.epochs
    MAX_STEPS = max((len(train_dataset)) // BATCH_SIZE * EPOCHS, EPOCHS)
    MICRO_BATCH_SIZE = args.per_device_eval_batch_size

    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE

    training_args = SFTConfig(
        per_device_train_batch_size=args.per_device_eval_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_steps=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=False,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        output_dir=os.path.join(args.model_out_root, args.tuned_model_file),
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch",
        bf16=True,
        #fp16=True,
        remove_unused_columns=True,
        seed=args.seed,
        report_to="none",
        dataset_text_field="text",
    )

    #model = prepare_model_for_kbit_training(model)

    # peft_config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     target_modules=[
    #         "q_proj",
    #         # "o_proj",
    #         # "k_proj",
    #         "v_proj",
    #         # "gate_proj",
    #         # "up_proj",
    #         # "down_proj",
    #     ],
    #     # target_modules=["q_proj", "v_proj"],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # Trainer
    print_trainable_parameters(model)
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        # peft_config=peft_config,
    )
    trainer.train()
    os.makedirs(os.path.join(args.model_out_root, args.tuned_model_file), exist_ok=True)
    trainer.save_model(os.path.join(args.model_out_root, args.tuned_model_file))
