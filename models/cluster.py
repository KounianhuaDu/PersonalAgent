from transformers import DebertaV2Tokenizer, DebertaV2Model
import torch
import json
from tqdm import tqdm
from pathlib import Path
import argparse
from sklearn.cluster import KMeans
import numpy as np
import pickle
import sys
import random
sys.path.append('..')
batch_size = 32

def get_first_k_tokens(text, k):
    """
    Extracts the first k tokens from a text string.

    :param text: The input text string.
    :param k: The number of tokens to extract.
    :return: The first k tokens of the text string.
    """
    # Split the text into tokens based on whitespace
    tokens = text.split()
    output = " ".join(tokens[:k])

    # Return the first k tokens
    return output

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def extract_article(text):
    marker = "] description: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

def cluster(calibration_data):
    # Step 1: Load the DeBERTa-v3-base tokenizer and model
    tokenizer = DebertaV2Tokenizer.from_pretrained('../model_weights/deberta-v3-large')
    model = DebertaV2Model.from_pretrained('../model_weights/deberta-v3-large').cuda()
    task_name = 'cluster_data'

    all_user_emb = []
    # all_len = []
    u_ids = list(calibration_data.keys())
    processed_data = dict()
    for user in u_ids:

        history_embeddings_list = []
        visible_history_list = []
        qa_lines = []
        ids = []
        for samples in calibration_data[user]:
            ids.append(samples['id'])
            qa_lines.append(samples)
            for s in samples['profile']:
                if s['id'] not in ids:
                    ids.append(s['id'])
                    visible_history_list.append(s)
        prompt = "Generate a headline for the following article: "
        profile_lines = [{'input': prompt + line['text'], 'output': line['title']} for line in visible_history_list]
        # all_len.append(len(ids))
        # for p in visible_history_list:
        #     for key, value in p.items():
        #         p[key] = get_first_k_tokens(p[key], 368)
        if len(qa_lines)>128:
            calibration_lines = random.sample(qa_lines, 128)
        else:
            calibration_lines = qa_lines + random.sample(profile_lines, min(128 - len(qa_lines), len(profile_lines)))
        processed_data[user] = calibration_lines
        # user_nl_history_list_batched = split_batch(user_nl_history_list, batch_size)

        for batch in tqdm(calibration_lines):

            with torch.no_grad():
                inputs = tokenizer(batch['input'] + batch['output'], return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
                outputs = model(**inputs)

                last_hidden_states = outputs.last_hidden_state
                # Compute attention mask
                attention_mask = inputs['attention_mask']

                # Expand attention mask so it has same size as last_hidden_states, for broadcasting purposes
                attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

                # Multiply last hidden states by attention mask, then sum and divide by number of tokens
                masked_hidden_states = last_hidden_states * attention_mask
                summed = torch.sum(masked_hidden_states, 1)
                count = torch.clamp(attention_mask.sum(1), min=1e-9)
                mean_pooled = summed / count

            history_embeddings_list.append(mean_pooled.cpu())

        history_embedding_concat = torch.cat(history_embeddings_list, dim=0).cpu().mean(dim=0, keepdim=True)
        all_user_emb.append(history_embedding_concat)

    all_user_emb = torch.cat(all_user_emb, dim=0)
    print(all_user_emb.size())

    Path(f'./{task_name}/').mkdir(parents=True, exist_ok=True)

    torch.save(all_user_emb, f'./{task_name}/user_history_emb.pt')


    emb = all_user_emb.numpy()

    k=50
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=3000).fit(emb)
    labels = kmeans.labels_

    prune_data = dict()

    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        all_data = []
        for idx in cluster_indices:
            uid = u_ids[idx]
            all_data += processed_data[uid]
        for idx in cluster_indices:
            uid = u_ids[idx]
            prune_data[uid] = {'self_data': processed_data[uid], 'cluster_data': all_data}
        print(len(all_data), len(cluster_indices))
        
    with open(f"./{task_name}/prune_data.json", 'w') as f:
        json.dump(prune_data, f, indent=4)

    print('Done!')
    return prune_data
