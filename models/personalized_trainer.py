import os
import json
import math
import numpy as np
import random
import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import LongformerPreTrainedModel, LongformerModel
from transformers.utils import ModelOutput
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn import init
import pickle as pkl
from colorama import Fore, init
init(autoreset=True) 

@dataclass
class LongformerSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class LongformerPersonalizedClsHead(nn.Module):
    def __init__(self, config, num_users):
        super().__init__()
        self.num_users = num_users
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.dense_W = nn.Parameter(torch.empty(self.num_users, config.hidden_size, config.hidden_size))
        self.dense_b = nn.Parameter(torch.empty(self.num_users, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_W = nn.Parameter(torch.empty(self.num_users, config.hidden_size, config.num_labels))
        self.out_proj_b = nn.Parameter(torch.empty(self.num_users, config.num_labels))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.dense_W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dense_W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.dense_b, -bound, bound)
        nn.init.kaiming_uniform_(self.out_proj_W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out_proj_W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.out_proj_b, -bound, bound)

        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, hidden_states, user_mask, **kwargs):
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.dense(hidden_states)
        user_dense_W = torch.bmm(user_mask.unsqueeze(0).expand(self.hidden_size,-1,-1), self.dense_W.permute(1,0,2)).transpose(0,1)
        user_dense_b = torch.matmul(user_mask, self.dense_b)
        hidden_states = torch.bmm(hidden_states.unsqueeze(1), user_dense_W).squeeze() + user_dense_b
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # output = self.out_proj(hidden_states)
        user_out_proj_W = torch.bmm(user_mask.unsqueeze(0).expand(self.hidden_size,-1,-1), self.out_proj_W.permute(1,0,2)).transpose(0,1)
        user_out_proj_b = torch.matmul(user_mask, self.out_proj_b)
        output = torch.bmm(hidden_states.unsqueeze(1), user_out_proj_W).squeeze() + user_out_proj_b
        return output

class LongformerForPersonalizedCls(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 实现逻辑：包含longformer作为base generator，
        # 重写generate函数，
        # 最后的states拿来再做一次classification
        
        self.num_labels = config.num_labels
        self.config = config
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = LongformerPersonalizedClsHead(self.config, 531)
    
    def update_num_user(self, num_users):
        self.num_users = num_users
        self.classifier = LongformerPersonalizedClsHead(self.config, num_users)
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        user_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerSequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, user_mask)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

class personalization_orm_cls_trainer():
    def __init__(self, config):
        # super().__init__(self, config=config)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["trainer"]["tokenizer_name"]) #longformer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Since customized, only supports single device now.
        self.config = config
        
    def build_instruction(self, prompt, his):
        if self.config['task'] == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.config['task'] == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.config['task'] == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.config['task'] == "LaMP_4":  
            inp = f"Generate a headline for the following article: {prompt}"
            inp += f"For your reference, here are the user's past QA pairs:\n {his}"
            inp += "Please only generate the most suitable one headline, except which no extra text is needed."
        elif self.config['task'] == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.config['task'] == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        return inp
    
    def prepare_query_data(self, data, ranked_history, id_mapping, num_users):
        sample_ids = []
        samples = []
        sample_targets = []
        sample_gen_orig= []
        sample_targets_orig = []
        user_masks = []
        if self.config['task'] in ['LaMP_4', 'LaMP_6']:
            for user_id, lines in data.items():
                for line in lines:
                    inputs = line['input']
                    outputs = line['output']
                    generation = line['generation']
                    idx = line['id']
                    if self.config['use_rag']:
                        history = ranked_history[idx][:self.config['k']]
                        inputs = self.build_instruction(inputs, history)
                    
                    samples += [inputs+gen for gen in generation]
                    sample_ids += [idx]*len(generation)
                    sample_targets += [inputs+outputs]*len(generation)
                    sample_gen_orig += generation
                    sample_targets_orig += [outputs]*len(generation)
                    
                    user_mask = [0.0]*self.num_users
                    user_mask[self.id_mapping[user_id]] = 1.0
                    for i in range(len(generation)):
                        user_masks.append(user_mask)
                
        # convert samples into huggingface dataset with Dataset.from_dict
        dataset = Dataset.from_dict({"id": sample_ids, "generation": samples, "target": sample_targets, "orig_generation": sample_gen_orig, "orig_target": sample_targets_orig, "user_mask": user_masks}).with_format("torch")
        return dataset
    
    def prepare_full_train_data(self, config):
        # load data
        data_path = self.config['full_train_dir'][0]        
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
        
        if self.config['use_rag']:
            with open(self.config['full_train_dir'][1], 'r') as f:
                ranked_history = json.load(f)
        
        # create user mapping
        user_ids = list(data.keys())
        self.id_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.num_users = len(user_ids)  
        
        # prepare data
        samples = []
        labels = []
        user_masks = []
        if self.config['task'] in ['LaMP_4', 'LaMP_6']:
            for user_id, lines in data.items():
                for line in lines:
                    inputs = line['input']
                    outputs = line['output']
                    generation = line['generation']
                    if self.config['use_rag']:
                        idx = line['id']
                        history = ranked_history[idx][:self.config['k']]
                        inputs = self.build_instruction(inputs, history)
                    
                    #idx = line['id']
                    # For target
                    samples.append(inputs + outputs)
                    labels.append(1)
                    
                    # For psudo generation
                    samples += [inputs+gen for gen in generation]
                    labels += [0]*len(generation)
                    
                    user_mask = [0.0]*self.num_users
                    user_mask[self.id_mapping[user_id]] = 1.0
                    for i in range(len(generation)+1):
                        user_masks.append(user_mask)
                
        # convert samples into huggingface dataset with Dataset.from_dict
        dataset = Dataset.from_dict({"label": labels, "text": samples, "user_mask": user_masks}).with_format("torch").shuffle(seed=config['seed'])
        return dataset

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)
    
    def train_and_eval_seen_test(self):
        self.train_dataset = self.prepare_full_train_data(self.config)
        self.train_dataset = self.train_dataset.map(self.preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir=self.config["trainer"]["output_dir"],
            learning_rate=self.config["trainer"]["learning_rate"],
            per_device_train_batch_size=self.config["trainer"]["per_device_train_batch_size"],
            num_train_epochs=self.config["trainer"]["num_train_epochs"],
            weight_decay=self.config["trainer"]["weight_decay"],
            save_strategy=self.config["trainer"]["save_strategy"],
            push_to_hub=self.config["trainer"]["push_to_hub"],
        )
        
        self.model = LongformerForPersonalizedCls.from_pretrained(self.config["trainer"]["model_name"], num_labels=2)
        self.model.update_num_user(num_users=self.num_users)
        self.model.config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        self.model.config.eos_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        print(self.tokenizer.pad_token, self.tokenizer.encode(self.tokenizer.pad_token), self.model.config.pad_token_id, self.model.config.eos_token_id)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        print(Fore.GREEN + 'Start to train the full adapter.')
        trainer.train()
        if not os.path.exists(self.config["trainer"]["output_dir"]):
            os.makedirs(self.config["trainer"]["output_dir"])
        
        #torch.save(trainer.model.state_dict(), os.path.join(self.config["trainer"]["output_dir"], "model_weights.pth"))
        trainer.model.longformer.save_pretrained(self.config["trainer"]["output_dir"])
        print(Fore.GREEN + f'Trained full adapter saved to {self.config["trainer"]["output_dir"]}.')
        
        self.model = trainer.model
        self.model.eval()
        with torch.no_grad():
            data_path = self.config['seen_test_dir'][0]        
            with open(data_path, 'rb') as f:
                data = pkl.load(f)
            if self.config['use_rag']:
                with open(self.config['seen_test_dir'][1], 'r') as f:
                    ranked_history = json.load(f)
            else:
                ranked_history=None
            
            self.eval_dataset = self.prepare_query_data(data, ranked_history, self.id_mapping, self.num_users)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            def tokenized_dataset(data):
                return self.tokenizer(data["generation"], truncation=True)
            self.tokenized_dataset = self.eval_dataset.map(tokenized_dataset, batched=True)
            self.tokenized_dataset = self.tokenized_dataset.remove_columns(["id", "generation", "target", "orig_generation", "orig_target"])
            self.dataloader = DataLoader(self.tokenized_dataset, batch_size=self.config["trainer"]["per_device_eval_batch_size"], collate_fn=data_collator, shuffle=False)
            
            self.get_reward_score()
            self.select_and_save()
    
        
        
        
    def get_reward_score(self):
        # pass the dataset through the reward model
        self.solution_scores = []
        num = 0
        for batch in tqdm(self.dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits.detach().cpu() # B, 2
            # apply softmax on the logits
            logits = torch.nn.functional.softmax(logits, dim=-1)
            scores = logits[:,1]
            self.solution_scores.append(scores)
        self.solution_scores = torch.cat(self.solution_scores, dim=0)
        print(len(self.solution_scores), len(self.eval_dataset))
    
    def select_and_save(self):
        # select the solution with the highest score every num_return_sequences solutions
        selected_solutions = []
        for idx in range(0, len(self.eval_dataset), self.config["num_return_sequences"]):
            if idx + self.config["num_return_sequences"] < len(self.eval_dataset):
                max_idx = np.argmax(self.solution_scores[idx:idx+self.config["num_return_sequences"]])
                selected_solutions.append(self.eval_dataset[idx+int(max_idx)])
            else:
                max_idx = np.argmax(self.solution_scores[idx:])
                selected_solutions.append(self.eval_dataset[idx+int(max_idx)])
        if not os.path.exists(self.config["solution_output"]):
            os.mkdir(self.config["solution_output"])
        
        solution_file = f'{self.config["solution_output"]}/{self.config["seed"]}_selected.json'
        solutions = []
        for sol in selected_solutions:
            sol['generation'] = sol['orig_generation']
            sol['target'] = sol['orig_target']
            temp = {}
            temp['id'] = sol['id']
            temp['output'] = sol['orig_generation']
            temp['target'] = sol['orig_target']
            solutions.append(temp)
        
        
        with open(solution_file, 'w') as f:
            json.dump(
                {
                    'task': self.config['task'],
                    'golds': solutions
                }
            , f)
    
    def get_metric(self, ):
        if self.config['task'] in ["LaMP_1", "LaMP_2"]:
            metric = create_metric_f1_accuracy(self._get_labels(self.config['task']))
        elif self.config['task'] == "LaMP_3":
            metric = create_metric_mae_rmse()
        else:
            rouge_metric = evaluate.load("rouge", cache_dir='../evaluate_metrics/rouge')
            def compute_metrics(decoded_preds, decoded_labels):
                decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
                result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
                result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
                return result
            return compute_metrics
    
    def prepare_eval_data(self, config, history_path, query_path):
        dataset = self.prepare_data(self.config, history_path)
        queries = []
        idx = 0
        with open(query_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                queries.append(data)
        query_samples = []
        query_ids = []
        query_targets = []
        query_samples_orig = []
        query_targets_orig = []
        query_user_masks = []
        for idx in tqdm(range(len(queries))):
            sample = queries[idx]
            query_samples.append(sample['source']+sample['generation'])
            query_ids.append(sample['id'])
            query_targets.append(sample['source']+sample['target'])
            query_samples_orig.append(sample['generation'])
            query_targets_orig.append(sample['target'])
            user_mask = [0.0]*len(self.id_dict)
            user_mask[self.id_dict[sample['id']]] = 1.0
            query_user_masks.append(user_mask)
        # convert samples into huggingface dataset with Dataset.from_dict
        queries = Dataset.from_dict({"id": query_ids, "generation": query_samples, "target": query_targets, "orig_generation": query_samples_orig, "orig_target": query_targets_orig, "user_mask": query_user_masks}).with_format("torch")
        return dataset, queries
