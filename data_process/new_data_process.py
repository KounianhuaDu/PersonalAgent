import matplotlib.pyplot as plt
from collections import Counter
import pickle as pkl
import json
import os
import random
from colorama import Fore, init
from collections import defaultdict
init(autoreset=True)

def plot_frequency_bar_chart(data):

    counter = Counter(data)
    
    # 提取元素和对应的频率
    elements = list(counter.keys())
    frequencies = list(counter.values())
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(elements, frequencies)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}',
                 ha='center', va='bottom')
    
    # 设置图表标题和标签
    plt.title('Element Frequency in List', fontsize=15)
    plt.xlabel('Elements', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # 旋转x轴标签以防重叠
    plt.xticks(rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图表
    plt.savefig('./output.png')



def process_data(root_dir, dataset, threshold):
    with open(os.path.join(root_dir, dataset, 'train/train_questions.json'), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(root_dir, dataset, 'train/train_outputs.json'), 'r') as f:
        train_output = json.load(f)
        
    with open(os.path.join(root_dir, dataset, 'valid/dev_questions.json'), 'r') as f:
        test_data = json.load(f)  
    with open(os.path.join(root_dir, dataset, 'valid/dev_outputs.json'), 'r') as f:
        test_output = json.load(f)

    # key: user id; value: total history length.
    train_uid_set = defaultdict(int)
    test_uid_set = defaultdict(int)
    
    # samples indexed by user id.
    train_samples = defaultdict(list)
    test_samples = defaultdict(list)
    
    for sample, out in zip(train_data, train_output['golds']):
        assert sample['id'] == out['id'] 
        line = {}
        line['id'] = sample['id']
        line['input'] = sample['input']
        line['profile'] = sample['profile']
        line['user_id'] = sample['user_id']
        line['output'] = out['output']
        train_uid_set[sample['user_id']] += len(sample['profile'])
        train_samples[sample['user_id']].append(line)
    
    for sample, out in zip(test_data, test_output['golds']):
        assert sample['id'] == out['id'] 
        line = {}
        line['id'] = sample['id']
        line['input'] = sample['input']
        line['profile'] = sample['profile']
        line['user_id'] = sample['user_id']
        line['output'] = out['output']
        test_uid_set[sample['user_id']] += len(sample['profile'])
        test_samples[sample['user_id']].append(line)
        
    interaction_users_set = set(train_uid_set.keys()) & set(test_uid_set.keys())
    filtered_interaction_users_set = random.sample(interaction_users_set, 200)
    
    unseen_users_test = set(test_uid_set.keys()).difference(filtered_interaction_users_set)
    filtered_unseen_users_test = random.sample(unseen_users_test, 50)
    
    remaining_train = defaultdict(list)
    train = defaultdict(list)
    seen_test = defaultdict(list)
    unseen_test = defaultdict(list)
    
    for uid, samples in train_samples.items():
        if uid in filtered_interaction_users_set:
            train[uid] = samples
        else:
            remaining_train[uid] = samples
    
    for uid, samples in test_samples.items():
        if uid in filtered_interaction_users_set:
            seen_test[uid] = samples
        elif uid in filtered_unseen_users_test:
            unseen_test[uid] = samples
    
    os.makedirs(os.path.join(root_dir, dataset, 'processed'), exist_ok=True)
    
    with open(os.path.join(root_dir, dataset, 'processed', 'train.pkl'), 'wb') as f:
        pkl.dump(train, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'remain_train.pkl'), 'wb') as f:
        pkl.dump(remaining_train, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'seen_test.pkl'), 'wb') as f:
        pkl.dump(seen_test, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'unseen_test.pkl'), 'wb') as f:
        pkl.dump(unseen_test, f)
        
    train_sample_num = [train_uid_set[uid] for uid in list(train.keys())]
    remain_train_sample_num = [train_uid_set[uid] for uid in list(remaining_train.keys())]
    seen_test_sample_num = [test_uid_set[uid] for uid in list(seen_test.keys())]
    unseen_test_sample_num = [test_uid_set[uid] for uid in list(unseen_test.keys())]
    
    qualified_train = [int(num>200) for num in train_sample_num]
    qualified_train_remain = [int(num>200) for num in remain_train_sample_num]
    qualified_seen_test = [int(num>200) for num in seen_test_sample_num]
    qualified_unseen_test = [int(num>200) for num in unseen_test_sample_num]
    
    print(Fore.GREEN + f"# User of Train: {len(list(train.keys()))}, Qualified: {sum(qualified_train)}")
    print(Fore.GREEN + f"# User of Remain Train: {len(list(remaining_train.keys()))}, Qualified: {sum(qualified_train_remain)}")
    print(Fore.GREEN + f"# User of Seen Test: {len(list(seen_test.keys()))}, Qualified: {sum(qualified_seen_test)}")
    print(Fore.GREEN + f"# User of Unseen Test: {len(list(unseen_test.keys()))}, Qualified: {sum(qualified_unseen_test)}")
    
    print(Fore.GREEN + f"# Historical Samples of Train: {sum(train_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Remain Train: {sum(remain_train_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Seen Test: {sum(seen_test_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Unseen Test: {sum(unseen_test_sample_num)}")

process_data('./data', 'LaMP_7', 1000)

