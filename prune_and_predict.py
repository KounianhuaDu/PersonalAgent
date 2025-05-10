import json
import argparse
from colorama import Fore, Back, Style, init
from tqdm import tqdm
import yaml
import pickle as pkl
from evaluators.eval_lamp import evaluate_task
init(autoreset=True)
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## dataset related
    parser.add_argument(
        "--dataset", default="LaMP_4", help="Dataset to use, default: APPS"
    )
    parser.add_argument("--data_path", default="./data", help="Path to save the data")
    
    ## output & log
    parser.add_argument(
        "--out_path", default="./output/generation", help="Path to save the output"
    )
    parser.add_argument(
        "--res_path", default="./output/res", help="Path to save the output"
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
    
    ## Pruning arguments
    parser.add_argument(
        "--sparsity", default = 0.3, type = float
    )
    parser.add_argument(
        "--structured", action="store_true", default=False
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

    
    args.base_model_addr = os.path.join(args.modelweight, 'Meta-Llama-3.1-8B-Instruct')
    
    if not args.eval:
        # Pruning
        from models.PrunePredict import PrunePredict
        model = PrunePredict(args)
        u_ids = model.u_ids
        
        output_dict = dict()
        total_outputs = []
        for idx, u_id in enumerate(u_ids):
            print(Fore.RED + f"Trial {idx}.")
            outputs = model.prune_for_one_user(u_id)
            output_dict[u_id] = outputs
            total_outputs += outputs
            os.makedirs(os.path.join(args.out_path, "llama-prune", args.dataset, f'Prune_res_{args.sparsity}_{args.structured}', f'{u_id}'), exist_ok=True)
            with open(os.path.join(args.out_path, "llama-prune", args.dataset, f'Prune_res_{args.sparsity}_{args.structured}', f'{u_id}', 'gen.json'), 'w') as f:
                json.dump(outputs, f)
        
        os.makedirs(os.path.join(args.out_path, "llama-prune", args.dataset, f'Prune_total_{args.sparsity}_{args.structured}'), exist_ok=True)
        with open(os.path.join(args.out_path, "llama-prune", args.dataset, f'Prune_total_{args.sparsity}_{args.structured}', 'gen.pkl'), 'wb') as f:
                pkl.dump(output_dict, f)
        
        print(Fore.RED + "All pruning finishes.")
    else:
        # Evaluate
        import evaluate
        from rouge import Rouge
        
        with open(os.path.join(args.out_path, "llama-prune", args.dataset, f'Prune_total_{args.sparsity}_{args.structured}', 'gen.pkl'), 'rb') as f:
            output_dict = pkl.load(f)
        
        total_outputs = []
        for key, values in output_dict.items():
            total_outputs += values

        def postprocess_text_generation(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]
            return preds, labels

        rouge_metric = evaluate.load("rouge", cache_dir='./evaluate_metrics/rouge')
        print('Metric Loaded.')
        def compute_metrics(decoded_preds, decoded_labels):
            decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
            result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
            return result
        
        def regularize(text):
            if '\n' in text:
                lines = text.split('\n')
                for line in lines:
                    if len(line) > 1:
                        return line
            else:
                return text

        preds = []
        gts = []
        for line in total_outputs:
            preds.append(regularize(line['generation']))
            gts.append(line['output'])
        res = compute_metrics(preds, gts)
        print(res)