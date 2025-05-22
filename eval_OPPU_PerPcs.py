from evaluators.eval_lamp import evaluate_task
import os
import json
import sys
  
task_name = "news_headline"
algo = 'OPPU' # PerPcs or OPPU
model_name = 'Llama-3.2-3B-Instruct'
if algo == 'PerPcs':
    output_path = f"../Per-Pcs-main/output/{task_name}/LoRA-Composition"
elif algo == 'OPPU':
    output_path = f"../OPPU-main/output/{task_name}/k0-{model_name}"
    
# output_path = "./output/generation/llama3-3b_True/zeroshot_0/LaMP_4/generation.json"
evaluation_res = f"./output/res/{model_name}_True/{algo}/lamp"
os.makedirs(evaluation_res, exist_ok=True)
results = evaluate_task(output_path)  
print(results)
with open(evaluation_res + '/res.json', "w") as f:
    json.dump(results, f)