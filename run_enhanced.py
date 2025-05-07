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
    parser.add_argument("--train", action="store_true")
    

    ## backbone LLM
    parser.add_argument("--arch", default="deepseek")
    parser.add_argument(
        "--modelweight",
        default="../model_weights",
        help="Path to save the model weights.",
    )
    
    ## algo
    parser.add_argument(
        "--k", type=int, default=1
    )

    args = parser.parse_args()

    print(args)

    # Dataset loading
    if args.train:
        to_be_enhanced = ['train.pkl']
        rank_files =  ['train_ranked.json']
        names = ['train_enhanced.pkl']
    else:
        to_be_enhanced = ['unseen_test.pkl']
        rank_files =  ['unseen_test_ranked.json']
        names = ['unseen_test_enhanced.pkl']
        
    
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
                with open(os.path.join(args.data_path, args.dataset, 'processed', name.split('.')[0] + '_interupted.pkl'), 'wb') as f:
                    pkl.dump(enhanced_dict, f)
            
        with open(os.path.join(args.data_path, args.dataset, 'processed', name), 'wb') as f:
            pkl.dump(enhanced_dict, f)

    