import matplotlib.pyplot as plt
from collections import Counter
import json
import os
from colorama import Fore, init

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
        test_in = json.load(f)  
    with open(os.path.join(root_dir, dataset, 'valid/dev_outputs.json'), 'r') as f:
        test_out = json.load(f)


    qualified_users = 0
    training_cnt = 0
    
    qualified_users_in_out = []
    qualified_users_ids = []
    training_in_out = []
    uid_id_profile_len = {}
    
    for sample, out in zip(train_data, train_output['golds']):
        assert sample['id'] == out['id'] 

        if len(sample['profile']) >= threshold:
            qualified_users += 1
            
            line = {}
            line['id'] = sample['id']
            line['input'] = sample['input']
            line['profile'] = sample['profile']
            line['user_id'] = sample['user_id']
            line['output'] = out['output']
            qualified_users_in_out.append(line)
            qualified_users_ids.append(sample['user_id'])
        else:
            training_cnt += 1
            line = {}
            line['id'] = sample['id']
            line['input'] = sample['input']
            line['profile'] = sample['profile']
            line['user_id'] = sample['user_id']
            line['output'] = out['output']
            training_in_out.append(line)
    
    test_in_out = []
    for t_in, t_out in zip(test_in, test_out['golds']):
        assert t_in['id'] == t_out['id']
        if t_in['user_id'] in qualified_users_ids:
            test_line = {}
            test_line['id'] = t_in['id']
            test_line['input'] = t_in['input']
            test_line['profile'] = t_in['profile']
            test_line['user_id'] = t_in['user_id']
            test_line['output'] = t_out['output']
            test_in_out.append(test_line)
    
    os.makedirs(os.path.join(root_dir, dataset, f'processed_{threshold}'), exist_ok=True)
    
    with open(os.path.join(root_dir, dataset, f'processed_{threshold}', 'shared_train.json'), 'w') as f:
        json.dump(training_in_out, f)
    with open(os.path.join(root_dir, dataset, f'processed_{threshold}', 'privacy_train.json'), 'w') as f:
        json.dump(qualified_users_in_out, f)
    with open(os.path.join(root_dir, dataset, f'processed_{threshold}', 'test.json'), 'w') as f:
        json.dump(test_in_out, f)
    
    print(Fore.GREEN + f"Threshold: {threshold}")
    print(Fore.GREEN + f"Avg. profile length: {sum(cnt)/len(cnt)}")
    print(Fore.GREEN + f"Total Users: {len(train_data)}")
    print(Fore.GREEN + f"Qualified Users: {len(qualified_users_in_out)}")
    print(Fore.GREEN + f"Total training samples: {len(training_in_out)}")
    
    #plot_frequency_bar_chart(cnt)

process_data('./', 'LaMP_4', 1000)

