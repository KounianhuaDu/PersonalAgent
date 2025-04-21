from colorama import Fore

def build_question_instruct(problem, tool_string, demo_string, dataset, train=False):
    if dataset == 'dailylife':
        rationale_instruct = """
        \n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. 
        The format must in a strict JSON format, like: {"task_steps": [one or more concrete steps, format as Step x: step description], "task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [{"name": name of the argument for the tool, "value": content of the argument}]}], "task_links": [{"source": "task name i", "target": "task name j"}]} 
        """
        rationale_instruct += """
        # REQUIREMENTS #: 
        1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #;\n
        2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes;\n
        3. the dependencies among task steps should align with the argument dependencies of the task nodes;\n
        4. the name of tool arguments should be align with the name field of # TASK LIST #;\n
        5. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;\n
        """
    elif dataset == 'xlam':
        rationale_instruct = """
        \n# GOAL #: Based on the above tools, I want you generate tool callings to solve the # USER REQUEST #. 
        The format must in a list in strict JSON format, like: [{"name": "tool name must be from # TASK LIST #", "arguements":{a dict of argument name and value pairs}}] 
        """
        if not train:
            rationale_instruct += """
            # REQUIREMENTS #: 
            1. the generated tool callings can resolve the given user request # USER REQUEST # perfectly. Tool name must be selected from # TASK LIST #;\n
            2. the tool arguments should be align with the input field of # TASK LIST #;\n
            3. You may generate more than one tool callings in some order;\n
            4. ONLY output the tool calls with no other information attached;\n
            """
    elif "bfcl" in dataset:
        rationale_instruct = """
        \n# GOAL #: Based on the above tools, I want you generate tool callings to solve the # USER REQUEST #. 
        The format must in a list in strict JSON format, like: [{<tool name>: <arguments>}] 
        """
        if not train:
            rationale_instruct += """
            # REQUIREMENTS #: 
            1. the generated tool callings can resolve the given user request # USER REQUEST # perfectly. Tool name must be selected from # TASK LIST #;\n
            2. the tool arguments should be align with the input type field of # TASK LIST #;\n
            3. You may generate more than one tool callings in some order;\n
            4. ONLY output the tool calls with no other information attached;\n
            """
    else:
        rationale_instruct = """
        \n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. 
        The format must in a strict JSON format, like: {"task_steps": [one or more concrete steps, format as Step x: step description], "task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}], "task_links": [{"source": "task name i", "target": "task name j"}]} 
        """
        rationale_instruct += """
        # REQUIREMENTS #: 
        1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #;\n
        2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes;\n
        3. the dependencies among task steps should align with the argument dependencies of the task nodes;\n
        4. the tool arguments should be align with the input-type field of # TASK LIST #;\n
        5. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;\n
        """
    if not train:
        rationale_instruct += demo_string

    tool_string = "# TASK LIST #:\n" + tool_string

    rationale_instruct += """
    \n\n
    # USER REQUEST #: {{user_request}}
    now please generate your result in a strict JSON format:
    # RESULT #:"""

    rationale_instruct = tool_string + rationale_instruct.replace("{{user_request}}", problem)

    return rationale_instruct


def build_rationale_instruct(problem, train=False):
    rationale_instruct = """
    You are tasked with breaking down a complex user request into solvable sub-tasks by creating a task plan.\n
    Problem Description:\n
    {}\n""".format(problem)

    if not train:
        rationale_instruct += """
        Below is an example of step-by-step clues for planning a task to decompose a complex request into sub-tasks:\n
        ```json
        [
        {"Clue of Step 1": "Identify the high-level goal or main request and determine the general sub-tasks required to achieve it."},
        {"Clue of Step 2": "Analyze the dependencies between sub-tasks and construct a graph where nodes represent sub-tasks and edges represent dependencies."},
        {"Clue of Step 3": "Select an optimal path within the graph, ensuring all dependencies are respected and that the path solves the problem efficiently."},
        {"Clue of Step 4": "Finish"}
        ]
        ```
        Based on this example, generate step-by-step clues to solve the given problem by breaking it down into sub-tasks and forming a connected path.

        -----Clues-----
        """
    return rationale_instruct


def build_intermediate_instruct(h=None, k=3):
    ins = "\n-----Instruction-----\n"
    if h is None:
        ins += (
            f"I need you to analyze and provide new thoughts that can lead to the correct solution plan. \n"
            f"I need you to output {k} possible thoughts and strategies. Remember each only contain one possible strategy of the problem.\n"
            f"Please wrap your response into a JSON object that contains keys `Thought-i` with i as the number of your thought, and key `Reasonableness` with the Reasonableness of each thought, which should between 0~1 and the sum should be 1.\n"
            f"The JSON should be a **list of dicts**, the dicts are splited with comma ','. \n"
            f"Example Answers:\n")
        ins += """
    ```json
    [
    {"Thought-1": "Divide the user request into smaller, manageable sub-tasks.", "Reasonableness": 0.8},
    {"Thought-2": "Organize sub-tasks in a way that ensures dependencies between them are maintained.", "Reasonableness": 0.7},
    {"Thought-3": "Finish", "Reasonableness": 0.9}
    ]
    ```\n
    If you think the previous thoughts are enough to solve the task, you can answer with "Finish".
            """
        return ins

    if h == 0:
        ins += 'Now, please generate {} different clues for the first sub-task (Step 1).\n'.format(k)
    else:
        ins += 'Now that we have generated clues for the previous sub-tasks, follow the dependencies and generate {} different clues for the next sub-task (Step {}).\n'.format(
            k, h + 1)

    ins += """
    Please wrap your response into a JSON object that contains keys `i Clue of Step {}` with i as the number of your clue, and key `Reasonableness` with the Reasonableness score of each clue.\n
    """.format(h + 1)
    print(ins)
    ins += """An example is given below.\n
    ```json
    [
    {"1 Clue of Step x": "Divide the user request into smaller, manageable sub-tasks.", "Reasonableness": 0.8},
    {"2 Clue of Step x": "Organize sub-tasks in a way that ensures dependencies between them are maintained.", "Reasonableness": 0.7},
    {"3 Clue of Step x": "Finish", "Reasonableness": 0.9}
    ]
    ```\n
    If you think the previous clues are enough to solve the task, you can answer with "Finish".
    ONLY output the clues. No other information attached. The actual plan is not needed.
    """

    return ins


def get_reward_instruct(previous_clues, problem):
    instruction = """
    You are tasked with breaking down a complex user request into sub-tasks and selecting the best sequence of steps to fulfill the original request.\n
    Problem Description:\n
    {}\n
    -----Clues-----\n
    The following clues were generated as steps for solving the problem:\n
    {}\n
    -----Instruction-----\n
    Using the clues provided, plan a connected path through the sub-tasks. Each sub-task represents a node in a graph, and the edges represent the dependencies between them. Your objective is to select a valid, connected path that solves the problem efficiently. Please return the sub-task sequence in JSON format, ensuring that all dependencies are respected.
    """.format(problem, previous_clues)
    return instruction

