from ChatModels import *
from colorama import Fore, init
init(autoreset=True)
import json

class RAG:
    def __init__(self, args, ranking_dict, enhanced=False):
        self.args = args
        
        print(args.arch)
        if args.arch in ["llama3-8b", "llama3-3b"]:
            self.generator = LlamaChat(args.arch, args)
        elif args.arch == "deepseek":
            self.generator = DeepSeekChat(args.arch, args)
        elif args.arch == "gemma":
            self.generator = GemmaChat(args.arch, args)
        elif args.arch == "gpt":
            self.generator = GPTChat(args.arch, args)
        elif args.arch == "qwen":
            self.generator = QwenChat(args.arch, args)
        elif args.arch == "claude":
            self.generator = ClaudeChat(args.arch, args)
        else:
            raise NotImplementedError
        
        if args.arch in ['gpt','deepseek']:
            print('API model.')
        else:
            total_params = sum(p.numel() for p in self.generator.model.parameters())
            print(Fore.GREEN + f"#Parameters: {total_params / 1e9:.2f}B")
        
        self.ranking_dict = ranking_dict
        self.enhanced = enhanced

    
    def get_his(self, p_id, k):
        ranked_profiles = self.ranking_dict[p_id][:k]
        
        q_a_history = []
        for idx, sample in enumerate(ranked_profiles):
            line = f"Historical sample {idx}:\n Q: {sample['text']}. \n A: {sample['title']}."
            q_a_history.append(line)
        q_a_history = '\n'.join(q_a_history)
        return q_a_history
        
    def generate(self, problem_instance, k):
        p_id = problem_instance['id']
        
        ranked_his = self.get_his(p_id, k)
        
        if not self.enhanced:
            raw_prompt = self.build_instruction(problem_instance['input'], ranked_his)
            output = self.generator.generate_response_api(raw_prompt, top_k=1)
            output_dict = {
                'id': p_id,
                'generation': output,
                'output': problem_instance['output']
            }
        else:
            raw_prompt = self.build_enhanced_instruction(problem_instance['input'], ranked_his)
            output = self.generator.generate_response_api(raw_prompt, top_k=1)
            if output:
                output = self.extract(output)
            else:
                print(f'error for {p_id}')
                output = ['\n','\n','\n']
            output_dict = {
                'id': p_id,
                'user_id': problem_instance['user_id'],
                'input': problem_instance['input'],
                'generation': output,
                'output': problem_instance['output']
            }
        
        return output_dict
    

    def build_instruction(self, prompt, his):
        if self.args.dataset == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.dataset == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.dataset == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.dataset == "LaMP_4":  
            inp = f"Generate a headline for the following article: {prompt}"
            inp += f"For your reference, here are the user's past QA pairs:\n {his}"
            inp += "Please only generate the most suitable one headline, except which no extra text is needed."
        elif self.args.dataset == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.args.dataset == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        return inp
    
    def build_enhanced_instruction(self, prompt, his):
        if self.args.dataset == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.dataset == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.dataset == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.dataset == "LaMP_4":  
            inp = f"Generate a headline for the following article: {prompt}"
            inp += f"For your reference, here are the user's past QA pairs:\n {his}"
            inp += "Please generate 3 suitable headlines and wrap them in the below format. \n"
            inp += """
```json
[
{"Headline 1": "Are You Ready to Open Up to Trust, Happiness and Joy?"},
{"Headline 2": "Fear or Freedom? Choose Trust, Happiness, Joy"},
{"Headline 3": "The Journey to Trust, Happiness, and Joy Starts With a ‘Yes’"}
]
```
"""
        elif self.args.dataset == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.args.dataset == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        return inp
    
    def extract(self, response_text):
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        try:
            if response_text.strip()[0] != '[':
                response_text = '[' + response_text + ']'
            
            response_text = json.loads(response_text)
            top_lines_text = []
            for sample in response_text:
                ele_key = list(sample.keys())[0]
                top_lines_text.append(sample[ele_key])
        except Exception as e:
            top_lines_text = ['\n' for i in range(3)]

        return top_lines_text


        
