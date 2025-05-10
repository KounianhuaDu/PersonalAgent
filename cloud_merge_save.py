from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from colorama import Fore 

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

modelpath = "../model_weights/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    modelpath, device_map="auto"
)
param_dtype = next(model.parameters()).dtype

model = get_peft_model(model, peft_config)
model = PeftModel.from_pretrained(model, './tuned_models/cloud0510Lamp4/checkpoint-39')
model = model.merge_and_unload()
print(Fore.RED + "Lora model merged.")