from models.RAG import RAG
import json
import argparse
from colorama import Fore, Back, Style, init
from tqdm import tqdm
from evaluators.eval_lamp import evaluate_task
init(autoreset=True)
import os
import pickle as pkl
from collections import defaultdict


def main(problems, model, k):
    for problem_instance in tqdm(problems):
        idx = problem_instance['id']
        
        # Generate Code & Trace
        res = model.generate(problem_instance, k)
        if res:
            output_dict = res
        else:
            print(f"Generation Error for problem {problem_instance['id']}.")
            continue
        
        return output_dict

def prepare_freeze_train_data(privacy_path, test_path):
    u_ids = []
    with open(test_path, 'r') as f:
        data = json.load(f)
    for line in data:
        u_ids.append(line['user_id'])
    print(len(u_ids))
    print(u_ids[:5])
    
    train_samples=[]
    train_u_id_set = []
    with open(privacy_path, 'r') as f:
        data = json.load(f)
    for line in data:
        train_u_id_set.append(line['user_id'])
        if line['user_id'] in u_ids:
            train_samples.append(line)
    print(len(train_u_id_set))
    print(train_u_id_set[:5])
    return train_samples
            

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
    parser.add_argument("--out_path", default="./output", help="Path to save the data")
    

    ## backbone LLM
    parser.add_argument("--arch", default="llama3")
    parser.add_argument(
        "--modelweight",
        default="../model_weights",
        help="Path to save the model weights.",
    )
    
    ## algo
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

    args = parser.parse_args()

    print(args)

    # Dataset loading
    to_be_enhanced = ['unseen_test.pkl', 'seen_test.pkl']
    rank_files =  ['unseen_test_ranked.json', 'seen_test_ranked.json']
    names = ['unseen_test_hydra.pkl', 'seen_test_hydra.pkl']
    os.makedirs(os.path.join(args.out_path, args.dataset), exist_ok=True)
        
    for target, ranked, name in zip(to_be_enhanced, rank_files, names):
        with open(os.path.join(args.data_path, args.dataset, 'processed', target), 'rb') as f:
            target_data = pkl.load(f)
        with open(os.path.join(args.data_path, args.dataset, 'processed', ranked), 'r') as f:
            ranking_dict = json.load(f)
        
        model = RAG(args, ranking_dict, enhanced=True)
        print("Model set.")

        enhanced_dict=defaultdict(list)
        total_problems = []
        corresponding_user_ids = []
        for user_id, problems in target_data.items():
            total_problems += problems
            corresponding_user_ids += [user_id] * len(problems)
        print(len(total_problems))
        print(len(corresponding_user_ids))
        
        for problem in tqdm(total_problems):   
            user_id = problem['user_id'] 
            try:
                # Generate Code & Trace
                output_dict = model.generate(problem, args.k)
                enhanced_dict[user_id].append(output_dict)
            except Exception as e:
                print(e)
                with open(os.path.join(args.out_path, args.dataset, name.split('.')[0] + '_interupted.pkl'), 'wb') as f:
                    pkl.dump(enhanced_dict, f)
            
        with open(os.path.join(args.out_path, args.dataset, name), 'wb') as f:
            pkl.dump(enhanced_dict, f)

    from models.Hydra import Hydra
    with open('./configs/Hydra.yaml', "r") as f:
        args.config = yaml.load(f, Loader=yaml.FullLoader)
    #args.config = load_config()
    model = Hydra(args)
    