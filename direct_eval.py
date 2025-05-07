import evaluate
import json
from rouge import Rouge

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


datapath = './output/generation/Hydra/LaMP_4/0_selected_wo_train.json'
with open(datapath, 'r') as f:
        data = json.load(f)
        
preds = []
gts = []
for line in data['golds']:
        preds.append(line['output'])
        gts.append(line['target'])

res = compute_metrics(preds, gts)
print(res)