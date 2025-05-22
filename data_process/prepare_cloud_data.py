import torch
from utils import batchify
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
from utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_paper, add_string_after_title, extract_after_colon, extract_after_abstract, extract_after_description
import pickle as pkl
import argparse
import os

def retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, k, batch_size = 16):
    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
    output_query = contriver(**query_tokens)
    output_query = mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
    scores = []
    batched_corpus = batchify(corpus, batch_size)
    for batch in batched_corpus:
        tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        outputs_batch = contriver(**tokens_batch)
        outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
        temp_scores = output_query.squeeze() @ outputs_batch.T
        scores.extend(temp_scores.tolist())
    topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
    return [profile[m] for m in topk_indices.tolist()]

def get_his(p_id, k):
    ranked_profiles = rank_dict[p_id][:k]
    
    q_a_history = []
    for idx, sample in enumerate(ranked_profiles):
        line = f"Historical sample {idx}:\n Q: {sample['text']}. \n A: {sample['title']}."
        q_a_history.append(line)
    q_a_history = '\n'.join(q_a_history)
    return q_a_history

def build_instruction(prompt, his):
    inp = f"Generate a headline for the following article: {prompt}"
    inp += f"For your reference, here are the user's past QA pairs:\n {his}"
    inp += "Please only generate the most suitable one headline, except which no extra text is needed."
    return inp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## Seed
    parser.add_argument("--seed", type=int, default=42)

    ## Path
    parser.add_argument("--data_path", default="../data", help="Root path to save the train data.")
    parser.add_argument("--dataset", default="LaMP_4", help="Dataset name.")

    args = parser.parse_args()

    print(args)

    with open(os.path.join(args.data_path, args.dataset, "processed", "remain_train.pkl"), 'rb') as f:
        data = pkl.load(f)
    
    with open(os.path.join(args.data_path, args.dataset, "processed", "remain_train_ranked.json"), 'rb') as f:
        rank_dict = json.load(f)
    
    train_lines = []
    for user_id, lines in data.items():
        train_lines += lines
    
    total_train_lines = []
    for line in tqdm(train_lines):
        p_id = line['id']
        ranked_his = get_his(p_id, k=5)
        '''rag_line = {
            'input': build_instruction(line['input'], ranked_his),
            'output': line['output']
        }'''
        raw_line = {
            'input': line['input'],
            'output': line['output']
        }
        #total_train_lines.append(rag_line)
        total_train_lines.append(raw_line)
    

    total_lines = 

    print(len(total_train_lines))
    with open(os.path.join(args.data_path, args.dataset, "processed", "remain_train.json"), "w") as f:
        json.dump(total_train_lines, f)
    
    #corpus = corpus = [f'{x["title"]} {x["text"]}' for x in profile]