import json
import argparse
from colorama import Fore, Back, Style, init
from tqdm import tqdm
import yaml
import pickle as pkl
from evaluators.eval_lamp import evaluate_task
init(autoreset=True)
import os


def main(problems, model, k, output_path, task_name):
    outs = []
    for problem_instance in tqdm(problems):
        
        # Generate Code & Trace
        res = model.generate(problem_instance, k)
        if res:
            output_dict = res
        else:
            print(f"Generation Error for problem {problem_instance['id']}.")
            continue
        
        outs.append(output_dict)
    

    with open(os.path.join(output_path), "w") as f:
        json.dump(
            {
                "task": task_name,
                "golds": outs,
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## dataset related
    parser.add_argument(
        "--dataset", default="LaMP_4", help="Dataset to use, default: APPS"
    )
    parser.add_argument("--data_path", default="../pa_back/data", help="Path to save the data")
    
    ## output & log
    parser.add_argument(
        "--out_path", default="../pa_back/output/generation", help="Path to save the output"
    )
    parser.add_argument(
        "--res_path", default="../pa_back/output/res", help="Path to save the output"
    )

    ## backbone LLM
    parser.add_argument("--arch", default="llama3")
    parser.add_argument(
        "--modelweight",
        default="../model_weights",
        help="Path to save the model weights.",
    )
    
    ## algo
    parser.add_argument(
        "--algo", default="zeroshot", help="algorithm"
    )
    parser.add_argument(
        "--k", type=int, default=5
    )
    
    ## vllm
    parser.add_argument("--vllm", action="store_true", help="If True, use vllm.")
    
    ## resume checkpoint path
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="If True, load a tuned model.",
    )
    parser.add_argument(
        "--tuned_path",
        default="../tuned_models",
        help="Root path to save the checkpoints.",
    )
    parser.add_argument(
        "--model_file",
        default="",
        help="Checkpoint name. Valid only if resume is enabled.",
    )
    parser.add_argument(
        "--check_point",
        default="",
        help="Checkpoint name. Valid only if resume is enabled.",
    )

    ## LORA related
    parser.add_argument("--lora", action="store_true")
    parser.add_argument(
        "--lora_rank", type=int, default=8, help="LoRA rank for lora/qlora"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha for lora/qlora"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout for lora/qlora"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="all",
        help="If 'default', uses peft defaults. Use 'all' for our best guess for Llama models",
    )
    
    # Generate or eval
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="If True, enable eval mode.",
    )

    args = parser.parse_args()

    print(args)

    # Dataset loading
    with open(os.path.join(args.data_path, args.dataset, 'processed', 'seen_test.pkl'), 'rb') as f:
        data = pkl.load(f)
        problems = []
        for u_id, samples in data.items():
            problems += samples
    print(f"Got {len(problems)} problems.")
    with open(os.path.join(args.data_path, args.dataset, 'processed', 'seen_test_ranked.json'), 'r') as f:
        ranking_dict = json.load(f)
    
    # Path info
    os.makedirs(os.path.join(args.out_path, f"{args.arch}_{args.lora}", f"{args.algo}_{args.k}", args.dataset), exist_ok=True)
    output_path = os.path.join(args.out_path, f"{args.arch}_{args.lora}", f"{args.algo}_{args.k}", args.dataset, 'generation_less.json')
    os.makedirs(os.path.join(args.res_path, f"{args.arch}_{args.lora}", f"{args.algo}_{args.k}", args.dataset), exist_ok=True)
    evaluation_res = os.path.join(args.res_path, f"{args.arch}_{args.lora}", f"{args.algo}_{args.k}", args.dataset, 'res.json')
    
    if args.eval:
        # Evaluation
        results = evaluate_task(output_path)
        print(results)
        with open(evaluation_res, "w") as f:
            json.dump(results, f)
    else:
        # Model
        if args.algo == 'zeroshot':
            from models.ZeroShot import ZeroShot
            model = ZeroShot(args)
        elif args.algo == 'rag':
            from models.RAG import RAG
            model = RAG(args, ranking_dict)
    
        main(problems, model, args.k, output_path, args.dataset)
        
        
    