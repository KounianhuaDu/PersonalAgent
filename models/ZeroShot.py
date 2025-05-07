from ChatModels import *
from colorama import Fore, init
init(autoreset=True)

class ZeroShot:
    def __init__(self, args):
        self.args = args
        
        print(args.arch)
        if args.arch == "llama3":
            self.generator = LlamaChat(args.arch, args)
        elif args.arch == "deepseek":
            self.generator = DeepSeekChat(args.arch, args)
        elif args.arch == "gemma":
            self.generator = GemmaChat(args.arch, args)
        elif args.arch == "gpt":
            self.generator = GPTChat(args.arch, args)
        elif args.arch == "merge":
            self.generator = MergeChat(args.arch, args)
        elif args.arch == "claude":
            self.generator = ClaudeChat(args.arch, args)
        
        else:
            raise NotImplementedError
        
        
    def generate(self, problem_instance, k=None):
        p_id = problem_instance['id']
        
        raw_prompt = self.build_instruction(problem_instance['input'])

        output = self.generator.generate_response_api(raw_prompt, top_k=1)

        output_dict = {
            'id': p_id,
            'generation': output,
            'output': problem_instance['output']
        }

        return output_dict

    def build_instruction(self, prompt):
        if self.args.dataset == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.dataset == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.dataset == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.dataset == "LaMP_4":
            inp = f"Generate a headline for the following article: {prompt}"
            inp += "Please only generate the most suitable one headline, except which no extra text is needed."
        elif self.args.dataset == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.args.dataset == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        return inp


        
