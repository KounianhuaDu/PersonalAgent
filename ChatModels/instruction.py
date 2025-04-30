def build_rationale_instruct(problem):
    rationale_instruct = """
You will be given a problem desciption.\n
Please analyze the problem and generate step-by-step clues of how to solve the problem with executable python code.\n
The problem:\n
{}\n""".format(problem)
    rationale_instruct += """
An example of step-by-step clues for finding the longest palindrome substring is displayed below:\n
```json
[
{"Clue of Step 1": "For a substring, if it is a palindrome and the length is greater than 2, it is still a palindrome after the first and last letters are removed. According to this idea, we can use dynamic programming to solve this problem."},
{"Clue of Step 2": "Then we should consider the state transition function. We use P(i,j) to indicate whether the string composed of the i-th to j-th letters of the string s (denoted as as s[i:j]) is a palindrome string. s[i:j] is a palindrome only if s[i+1:j−1] is a palindrome and the i and j letters of s are the same."},
{"Clue of Step 3": "We also need to consider the boundary conditions in dynamic programming. For a substring of length 1, it is obviously a palindrome string; For a substring of length 2, it is a palindrome as long as its two letters are the same."},
{"Clue of Step 4": "We also need to consider the the cyclic order of dynamic programming. In the state transition equation, we are moving from a shorter string to a longer string."},
{"Clue of Step 5": "The final answer is the maximum length of the valid sub-strings."}
]
```
You can refer to the following example and generate step-by-step clues for the given problem.\n

-----Clues-----
"""
    return rationale_instruct

def build_intermediate_instruct(h, k):
    ins = "\n-----Instruction-----\n"
    if h==0:
        ins += 'Now, please firstly generate {} different clues for step 1.\n'.format(k)
    else: 
        ins += 'Now that we have generated the clues for previous steps. Please follow the clues and generate {} different clues for the next Step {}.\n'.format(k, h+1)
    ins += """
Please wrap your response into a JSON object that contains keys `i Clue of Step {}` with i as the number of your clue, and key `Reasonableness` with the Reasonableness of each clue.\n
""".format(h+1)
    ins += """An example is given below.\n
```json
[
{"1 Clue of Step x": "We could set an addition equation and declare the variables to get the result.", "Reasonableness": 0.7},
{"2 Clue of Step x": "We could use the print function to finish the task in one line: print(2 + 3).", "Reasonableness": 0.6},
{"3 Clue of Step x": "We should calculate the problem by setting a=2+3, and then print(a).", "Reasonableness": 0.5}
]
```
"""
    return ins

def build_action_instruct_new(h, k):
    ins = "\n-----Instruction-----\n"
    if h==0:
        ins += 'Now, please firstly generate {} different clues for step 1.\n'.format(k)
        ins += """
Please wrap your response into a JSON object that contains keys `Clue of Step 1`, and key `Reasonableness` with the Reasonableness of each clue.\n
"""
        ins += """An example is given below.\n
```json
[
{"1 Clue of Step x": "We could set an addition equation and declare the variables to get the result.", "Reasonableness": 0.7},
{"2 Clue of Step x": "We could use the print function to finish the task in one line: print(2 + 3).", "Reasonableness": 0.6},
{"3 Clue of Step x": "We should calculate the problem by setting a=2+3, and then print(a).", "Reasonableness": 0.5}
]
```
"""
    else: 
        ins += 'Now that we have the clues for previous steps. You should analyze the existing clues and decide whether the logic of clues is consistent and informative for solving the code problem. If the previous steps are consistent and informative, output DETAIL to proceed to generate the next step of clue. If not, output REFINE to proceed to refine the last step of existing clues.'
        ins += "Please make the decision and only output 'DETAIL' or 'REFINE'."
    return ins

def build_intermediate_instruct_refine(h, k):
    ins = "\n-----Instruction-----\n"
    if h==0:
        ins += 'Now, please firstly generate {} different clues for step 1.\n'.format(k)
    else: 
        ins += 'Now that we have generated the clues for previous steps. Please refine the last clue of Step {} and generate {} better clues to replace it.\n'.format(h, k)
    
    ins += """
Please wrap your response into a JSON object that contains keys `Clue of Step {}`, and key `Reasonableness` with the Reasonableness of each clue.\n
""".format(h)
    ins += """An example is given below.\n
```json
[
{"1 Clue of Step x": "We could set an addition equation and declare the variables to get the result.", "Reasonableness": 0.7},
{"2 Clue of Step x": "We could use the print function to finish the task in one line: print(2 + 3).", "Reasonableness": 0.6},
{"3 Clue of Step x": "We should calculate the problem by setting a=2+3, and then print(a).", "Reasonableness": 0.5}
]
```
"""
    return ins

def build_intermediate_instruct_refine_VF(h, k, vf):
    ins = "\n-----Instruction-----\n"
    if h==0:
        ins += 'Now, please firstly generate {} different clues for step 1.\n'.format(k)
    else: 
        ins += 'Now we have the clues for previous steps. The execution feedback of the generated codes following the existing clues is as below. The code generated following the existing clues cannot pass some test cases:\n{} \n Please refer to the feedback and refine the last clue of Step {} and generate {} better clues to replace it.\n '.format(h, vf, k)
    
    ins += """
Please wrap your response into a JSON object that contains keys `Clue of Step {}`, and key `Reasonableness` with the Reasonableness of each clue.\n
""".format(h)
    ins += """An example is given below.\n
```json
[
{"1 Clue of Step x": "We could set an addition equation and declare the variables to get the result.", "Reasonableness": 0.7},
{"2 Clue of Step x": "We could use the print function to finish the task in one line: print(2 + 3).", "Reasonableness": 0.6},
{"3 Clue of Step x": "We should calculate the problem by setting a=2+3, and then print(a).", "Reasonableness": 0.5}
]
```
"""
    return ins

def build_intermediate_instruct_detail(h, k):
    ins = "\n-----Instruction-----\n"
    if h==0:
        ins += 'Now, please firstly generate {} different clues for step 1.\n'.format(k)
    else: 
        ins += 'Now that we have generated the clues for previous steps. Please follow the clues and generate {} different clues for the next Step {}.\n'.format(k, h+1)
    ins += """
Please wrap your response into a JSON object that contains keys `i Clue of Step {}` with i as the number of your clue, and key `Reasonableness` with the Reasonableness of each clue.\n
""".format(h+1)
    ins += """An example is given below.\n
```json
[
{"1 Clue of Step x": "We could set an addition equation and declare the variables to get the result.", "Reasonableness": 0.7},
{"2 Clue of Step x": "We could use the print function to finish the task in one line: print(2 + 3).", "Reasonableness": 0.6},
{"3 Clue of Step x": "We should calculate the problem by setting a=2+3, and then print(a).", "Reasonableness": 0.5}
]
```
"""
    return ins

def build_intermediate_instruct_merge(h, k, clues):
    ins = "\n-----Instruction-----\n"
    if h==0:
        ins += 'Now, please firstly generate {} different clues for step 1.\n'.format(k)
    else: 
        ins += 'Now we have the clues for previous steps as above. The following clue candidates for the next Step {} are: {}. \n Please refer to the clue candidates and generate a better clue from them for the next Step {}.\n'.format(h+1, clues, h+1)
    ins += """
Please wrap your response into a JSON object that contains keys `Clue of Step {}`, and key `Reasonableness` with the Reasonableness of the clue.\n
""".format(h+1)
    ins += """An example is given below.\n
```json
[
{"Clue of Step x": "We could set an addition equation and declare the variables to get the result.", "Reasonableness": 0.7}
]
```
"""
    return ins

def get_reward_instruct(previous_thoughts, problem):
    instruction = """
You are a professional Python engineer. You will be given a problem desciption.\n
Please analyze the problem and generate executable python code to solve the problem.\n
The problem:\n
{}\n
-----Clues-----\n
{}\n
-----Instruction-----\n
Now we have the clues above. Please refer to the clues and solve the problem. Please only output the code.\n
""".format(problem, previous_thoughts)
    return instruction
