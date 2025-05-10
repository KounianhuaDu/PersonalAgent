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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from accelerate import Accelerator

import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict
import json
import pickle as pkl
from colorama import Fore, Back, Style, init

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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=256)
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model 8 bit.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)

    parser.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint name. Enable only if resume is true.")

    ## Path
    parser.add_argument("--modelweight", default="../model_weights", help="Path of the original model weights.")
    parser.add_argument("--model_name", default="Meta-Llama-3.1-8B-Instruct", help="Path of the original model weights.")
    parser.add_argument("--model_out_root", default="./tuned_models", help="Path to save the tuned model weights.")

    parser.add_argument("--data_path", default="./data", help="Root path to save the train data.")
    parser.add_argument("--dataset", default="LaMP_4", help="Dataset name.")

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
    '''with open(os.path.join(args.data_path, args.dataset, "processed", "remain_train.pkl"), 'rb') as f:
        data = pkl.load(f)
    
    train_lines = []
    for user_id, lines in data.items():
        train_lines += lines

    with open(os.path.join(args.data_path, args.dataset, "processed", "remain_train.json"), "w") as f:
        for line in train_lines:
            json.dump(line, f)
            f.write("\n")  # 每行一个 JSON 对象'''

    data_path = {"train": os.path.join(args.data_path, args.dataset, "processed", "remain_train.json")}
    dataset = load_dataset("json", data_files=data_path)
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        if args.arch == "llama3":
            text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>/nYou are a helpful personal assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>/n{samples['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>/n{samples['output']}<|eot_id|
            """
        return {"text": text}
    train_dataset = dataset["train"].map(return_prompt_and_responses)

    # Training config
    BATCH_SIZE = args.total_batch_size
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

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            # "o_proj",
            # "k_proj",
            "v_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ],
        # target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Trainer
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.model_out_root, args.tuned_model_file))
    print(Fore.RED + "Cloud model tuned and save.")
    
