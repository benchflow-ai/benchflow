import logging
import os

import json5
from jinja2 import Template
from openai import OpenAI

from benchflow import BaseAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class HistoryMemory:
    def __init__(self, previous_trace: list = [], reflection: str = "") -> None:
        self.previous_trace = previous_trace
        self.reflection = reflection

    def stringfy_thought_and_action(self) -> str:
        input_list = None
        str_output = ""
        try:
            input_list = json5.loads(self.previous_trace, encoding="utf-8")
        except:
            input_list = self.previous_trace
        if len(input_list) > 2:
            str_output = "["
            for idx in range(len(input_list)-1):
                str_output += f'Step{idx+1}:\"Thought: {input_list[idx]["thought"]}, Action: {input_list[idx]["action"]}, Reflection: {input_list[idx+1]["reflection"]}\";\n'
            str_output += "]"
            current_trace = input_list[-1]
            str_output += f'Specifically in the last step, you gave the following Thought: {current_trace["thought"]}\n You performed the following Action: {current_trace["action"]}\n You had the following Reflection: {self.reflection}\";\n'
        else:
            current_trace = input_list[-1]
            str_output += f'Specifically in the last step, you gave the following Thought: {current_trace["thought"]}\n You performed the following Action: {current_trace["action"]}\n You had the following Reflection: {self.reflection}\";\n'
        return str_output

    def construct_previous_trace_prompt(self) -> str:
        stringfy_thought_and_action_output = self.stringfy_thought_and_action()
        previous_trace_prompt = f"The previous thoughts, actions and reflections are as follows: \
            {stringfy_thought_and_action_output}.\n\nYou have done the things above.\n\n"
        return previous_trace_prompt

    @staticmethod
    def construct_cache(cache_info: list):
        pass

class BasePrompts:

    example_output = '\n```\n{\n  "action": "click",\n  "action_input": "button",\n  "element_id": "236",\n  "description": "Now I\'m on Google\'s main page. I\'m now clicking the button with element_id [236] to see more information."\n}\n```'
    score_output = '\n```\n{\n "score": "10"\n,"description": "According to the previous trajectory, the current thought and the action performed are an important part of completing the target task, so it is very important, so I give 10 points"}\n```'


    # - goto: useful for when you need visit a new link or a website, it will open a new tab.
    # - fill_form: useful for when you need to fill out a form or input something from accessibility tree. Input should be a string.
    # - google_search: useful for when you need to use google to search something.
    # - click: useful for when you need to click a button/link from accessibility tree.
    # - select_option: useful for when you need to select a drop-down box value. When you get (select and option) tags from the accessibility tree, you need to select the serial number(element_id) corresponding to the select tag, not the option, and select the most likely content corresponding to the option as Input.
    # - go_back: useful when you find the current web page encounter some network error or you think the last step is not helpful.

    planning_prompt_system = '''You are an assistant who not only helps to browse and operate web pages to achieve certain goals, but also needs to explore the information on the page to answer the questions raised by the target task. Please answer the following questions as much as possible.
        There are key information you will get:
        **Key Information**:
            - Previous trace: all thoughts, actions and reflections you have made historically.
            - Accessibility tree: characteristic expression of the current web page.
            
        **Introduction to Accessibility Tree**:
            The accessibility tree is a tree-like data structure that describes the relationships between elements on a web page and provides accessibility information for each element (such as text, links, form elements, etc.).
            - **Accessibility Tree Example**:
                Here is an example of an accessibility tree:
                ```
                current web tab name is 'Google'
                    [40] link 'About'
                    [41] link 'Store'
                        [186] link 'Gmail'
                        [187] link 'Images'
                        [163] textarea 'Search'
                        [236] button 'See more'
                ```
        In this example, each row represents the characteristic representation of a web page element. It has three attributes: '[40]' for the element's element_id, 'link' indicates the element is a link, and 'About' for the content of the element.
        Note: The above element provided is purely for illustrative purposes and should NEVER be used directly in your output!         

        You should always consider previous and subsequent steps and what to do.
        **Thought Space**:
            - What action do you think is needed now to complete the task?
            - What's the reason of taking that action?
        
        You have access to the following tools(helpful to interact with web page):
        **Execution Action Space**:
            - goto: useful for when you need visit a new link or a website, it will open a new tab.
            - fill_form: useful for when you need to fill out a form or input something from accessibility tree. Input should be a string.
            - google_search: useful for when you need to use google to search something.
            - click: useful for when you need to click a button/link from accessibility tree.
            - select_option: useful for when you need to select a drop-down box value. When you get (select and option) tags from the accessibility tree, you need to select the serial number(element_id) corresponding to the select tag, not the option, and select the most likely content corresponding to the option as Input.
            - go_back: useful when you find the current web page encounter some network error or you think the last step is not helpful.
            - cache_data: useful when you need to extract information from the page that you think is extremely valuable for completing the target task. It is not a direct answer to the target task, but it is extremely relevant to the target task. Subsequent actions may refer to this part of the information and return this information as input
            - get_final_answer: useful for when you think it is the answer to the target task and no other operations are required, Input should be a answer content.
        
        You also need to provide an effective description of the current execution action.
        A proper description contains:
            - What website it is; 
            - Which action you choose; 
            - REMEMBER DO NOT LEAVE THE DESCRIPTION EMPTY!

        You have to follow the instructions or notes:
        **Important Notes**:
            - Under the following conditions, you are restricted to using the `google_search` or `goto` tools exclusively: 
                1. In the initial step of a process or when there's no preceding interaction history (i.e., the previous trace is empty). 
                2. In situations where the accessibility tree is absent or not provided.
            - Your action should not be the same as last step's action.
            - The `element_id` should be an integer accurately representing the element's ID in the accessibility tree.
            - AVOID using the provided example's element_id as your output.
            - The output JSON blob must be valid; otherwise, it cannot be recognized.
        
        **Special Circumstances Guidelines**:
            - When performing a search on a website, if you find the search results do not display sufficient content, consider simplifying or modifying your search query. Reducing the complexity of your search query or altering keywords may yield more comprehensive results.
        
        Please ensure the accuracy of your output, as we will execute subsequent steps based on the `action`, `action_input` and `element_id` you provide.
        
        **Output Requirements**:
        - Ensure your output strictly adheres to the JSON blob format outlined below:
            
            ```
            {
                "thought": ACTUAL_THOUGHT
                "action": ACTUAL_TOOLS,
                "action_input": ACTUAL_INPUT,
                "element_id": ACTUAL_ELEMENT_ID,
                "description": ACTUAL_DESCRIPTION
            }
            ```
          
        - A VALID JSON BLOB EXAMPLE AS FELLOWS:
            ```
            {
                "thought": "In order to complete this task,I need to go to the Google home page",
                "action": "click", 
                "action_input": "button",
                "element_id": "236",
                "description": "Now I\'m on Google\'s main page. I\'m now clicking the button with element_id [236] to see more information."
            }
            ```
        '''

    planning_prompt_user = "The question here is described as \"{{user_request}}\".\n\n"

    global_reward_prompt_system = ('''\
        You are an assistant to help navigate and operate the web page to achieve certain task.
        Your goal is to evaluate the previous series of traces(thoughts and actions) and think about what key steps are needed to complete the task in the future.
        There are key information you will get:
        **Key Information**:
            - Previous trace: all thoughts, actions and reflections you have made historically.
            - Accessibility tree: characteristic expression of the current web page.
            - Screenshot: visual information of the current web page (may include).
        
        You also need to combine the previous trace to give the completion status of the current task.
        **Status Of Task Completion**
            - doing: You have completed the intermediate steps of the target task but not entirely finish the target task.
            - finished: You are entirely certain about completing the target task.
            - loop: You find that the the last two steps of previous actions are the same, it is determined that the process is stuck in a local optimum solution.
        
        You will judge and score the task completion and reasonableness of previous actions. The score ranges from 1-10, but the score you give can only be selected from [1, 3, 7, 9, 10].
        **Judging and Scoring Criteria**:
            - score = 1: You find that the status of the task is stuck in a loop by analyzing the previous trace.
            - score = 3: You find that performing the previous trajectories(thoughts and actions) is not likely helpful in completing target task and you need to adjust the direction of your planning and action or start over from beginning.
            - score = 7: You find that performing the previous trajectories(thoughts and actions) are helpful in completing the target task.
            - score = 9: You find that performing the previous trajectories(thoughts and actions) are a very critical intermediate step to complete this task.
            - score = 10: You find that performing the previous trajectories(thoughts and actions) have completed the task perfectly.
        You need to provide an effective evidence of scoring for the series of the previous trace.
            - Why do you give this score? 
            - What is the reason?

        You also need to provide an effective description or summary of the above requirements through key information and characteristics of the current web page.
        **A proper description contains**:
            - What is the current completion status of the task? (IMPORTNAT)
            - What is your overall plan for completing your goal and target task in the future? (IMPORTNAT)
            - REMEMBER DO NOT LEAVE THE DESCRIPTION EMPTY!

        **Output Requirements**:
        - Ensure your output strictly follows this format:
            ```json
            {
                "status": "ACTUAL_STATUS",
                "score": "ACTUAL_SCORE",
                "reason": "ACTUAL_REASON",
                "description": "ACTUAL_DESCRIPTION"
            }
            ```
        - A VALID JSON BLOB EXAMPLE AS FELLOWS:
            ```
            {
                "status": "doing",
                "score": "3",
                "reason": "You need to complete a search for camping tents that can accommodate 2 people and sort the results in rei by price from low to high. According to your previous trajectory, you navigated to the rei official website and clicked the 2-person button, which are correct actions. But when you complete the final step of sorting prices, you actually click on a link to a tent product. This is a completely unreasonable action. So I give it 3 points. Maybe you need to return to the previous interface to re-plan and select the 'sort by' button"
                "description": "According to the current web page information, you can know that this is the homepage of a tent product, which is not very consistent with the purpose of the target task. The next overall plan to complete this task is to return to the previous page and select the sort by button."
            }
            ```
    ''')

    global_reward_with_GroundTruth_prompt_system = ('''\
        You are an assistant to help navigate and operate the web page to achieve certain task.
        Your goal is to evaluate the previous series of traces(thoughts and actions) and think about what key steps are needed to complete the task in the future.
        There are key information you will get:
        **Key Information**:
            - Previous trace: all thoughts, actions and reflections you have made historically.
            - Current Webpage Information:
                - Accessibility tree: characteristic expression of the current web page.
                - Screenshot: visual information of the current web page. (may include)
            - Reference Guide: detailed and step-by-step reference guide for completing the target task, serving as a benchmark for evaluating progress and strategizing the necessary actions.

        **Notes to Reference Guide**:
            - The Reference Guide plays a crucial role in aiding the evaluation of the current Status of Task Completion. The 'Completion Verification' section within the Reference Guide is instrumental in determining whether a task can be classified as 'finished.'
            - Furthermore, for a task to be considered fully completed, all **key conditions** must be met as specified.


        You also need to combine the previous trace to give the completion status of the current task.
        **Status of Task Completion**
            - doing: You have completed the intermediate steps of the target task but not entirely finish the target task.
            - finished: You are entirely certain about completing the target task.
            - loop: You find that the the last two steps of previous actions are the same, it is determined that the process is stuck in a local optimum solution.

        You will judge and score the task completion and reasonableness of previous actions. The score ranges from 1-10, but the score you give can only be selected from [1, 3, 7, 9, 10].
        **Judging and Scoring Criteria**:
            - score = 1: You find that the status of the task is stuck in a loop by analyzing the previous trace.
            - score = 3: You find that performing the previous trajectories(thoughts and actions) is not likely helpful in completing target task and you need to adjust the direction of your planning and action or start over from beginning.
            - score = 7: You find that performing the previous trajectories(thoughts and actions) are helpful in completing the target task.
            - score = 9: You find that performing the previous trajectories(thoughts and actions) are a very critical intermediate step to complete this task.
            - score = 10: You find that performing the previous trajectories(thoughts and actions) have completed the task perfectly.
        You need to provide an effective evidence of scoring for the series of the previous trace.
            - Why do you give this score? 
            - What is the reason?

        You also need to provide an effective description or summary of the above requirements through key information and characteristics of the current web page.
        **A proper description contains**:
            - What is the current completion status of the task? (IMPORTNAT)
            - What is your overall plan for completing your goal and target task in the future? (IMPORTNAT)
            - REMEMBER DO NOT LEAVE THE DESCRIPTION EMPTY!

        **Output Requirements**:
        - Ensure your output strictly follows this format:
            ```json
            {
                "status": "ACTUAL_STATUS",
                "score": "ACTUAL_SCORE",
                "reason": "ACTUAL_REASON",
                "description": "ACTUAL_DESCRIPTION"
            }
            ```
        - A VALID JSON BLOB EXAMPLE AS FELLOWS:
            ```
            {
                "status": "doing",
                "score": "3",
                "reason": "You need to complete a search for camping tents that can accommodate 2 people and sort the results in rei by price from low to high. According to your previous trajectory, you navigated to the rei official website and clicked the 2-person button, which are correct actions. But when you complete the final step of sorting prices, you actually click on a link to a tent product. This is a completely unreasonable action. So I give it 3 points. Maybe you need to return to the previous interface to re-plan and select the 'sort by' button"
                "description": "According to the current web page information, you can know that this is the homepage of a tent product, which is not very consistent with the purpose of the target task. The next overall plan to complete this task is to return to the previous page and select the sort by button."
            }
            ```
    ''')

    global_reward_prompt_user = "The target task here is described as \"{{user_request}}\".\n\n"\
        "The previous trajectories(thoughts, actions and reflections) are: {{stringfy_thought_and_action_output}}.\n\nYou have done the things above.\n\n"

    current_reward_prompt_system = '''You are an assistant to help navigate and operate the web page to achieve certain task.
        Your goal is to make an assessment of the action you are currently performing.
        There are key information you will get:
        **Key Information**:
            - previous trace: all thoughts and actions to complete this task step by step
            - current trace: current thought and action performed 
            - accessibility tree: characteristic expression of the current web page
        
        You will judge and score the currently performed action. The score ranges from 1-10, but the score you give can only be selected from [1, 3, 7, 9, 10]
        **Judging and Scoring Criteria**:
            - score = 1: You may not have obtained accessibility tree information(IMPORTANT).You may have encountered the issues such as Network connection issues,Human-computer verification issues,Encountered a blank page.
            - score = 3: The action you performed (such as clicking on an element) does not help at all to complete the task when accessibility tree is provided
            - score = 7: The action you performed (such as clicking on an element) is helpful in completing this task when accessibility tree is provided
            - score = 9: This action you performed is a very critical intermediate step to complete this task when accessibility tree is provided
            - score = 10: This action is the last step to complete the task when accessibility tree is provided
        
        You also need to provide an effective description of making the assessment
        A proper description contains:
            - Why do you give this score? 
            - What is the reason?
            - What would be better advice if given a low score? 
            - REMEMBER DO NOT LEAVE THE DESCRIPTION EMPTY!

        **Output Requirements**:
        - Ensure your output strictly follows this format:
            ```json
            {
                "score": "ACTUAL_SCORE",
                "description": ACTUAL_DESCRIPTION
            }
            ```
        - A VALID JSON BLOB EXAMPLE AS FELLOWS:
            ```
            {
                "score": "10",
                "description":"According to the previous trajectory, the current thought and the action performed are an important part of completing the target task, so it is very important. I give 9 points."
            }
            ```
    '''

    current_reward_prompt_user = "The target task here is described as \"{{user_request}}\".\n\n"\
        "The previous thought and action are:{{stringfy_previous_trace_output}}.\n\n"\
        "The current thought and action is: {{stringfy_current_trace_output}}.\n\nYou have done the current action\n\n"\

    judge_searchbar_prompt_system = "You are an assistant to help navigate and operate the web page to achieve certain goals. Answer the following questions as best you can.\n"\
        "Your target is to judge whether the input is the search bar.\n"
    judge_searchbar_prompt_user = "Now the webpage's accessibility tree(the key information of current web page)is below: {{input_element}}\n"\
        "Last step you have fill in the input(id={{element_id}}) with text:{{action_input}}"\
        "Judge whether the input is the search bar. If the blank is search bar, return yes, else return no. You should only return one word!"

    semantic_match_prompt_system = "Now you are an assistant to judge whether 2 elements are semantically same. I'll provide a judge rule and an answer.\n"\
        "If they are the same, you should return 1. If they are not related, you should return 0. "\
        "If they are related but not identical, return a decimal (two decimal places) between 0 and 1 of the degree of relevance you think.\n"\
        "For example, the judge rule is: Decide whether the place is New York. The score of \"new york\" and \"纽约\" are both 1, \"Brooklyn\" should be 0.\n"\
        "However, if the judge rule is: Decide whether the place is in New York. The score of \"new york\" and \"纽约\" and \"Brooklyn\" are all 1.\n"\
        "Another example, the judge rule is: Decide whether I'm looking for clothes. The score of \"red Clothes\" and \"green jacket\"should also be 1.\n"\
        "However, if the judge rule is: Decide whether I'm looking for red clothes. the score of \"bright red Clothing\" could be 0.85(red include bright red but they are not the same), the score of \"green Clothes\"should be 0.5(red is not green).\n"\
        "Remember, you should return a number with ``` and an explanation. Like output: ```1```, (your explanation)"  # "Remember, you should only return a number without any punctuation or explanation!"

    semantic_match_prompt_user = "You should judge by the rule below:{{semantic_method}}.\n\nmy answer is:{{input_answer}}\n"

class BasePromptConstructor:
    def __init__(self):
        pass

class PlanningPromptConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_system = BasePrompts.planning_prompt_system
        self.prompt_user = BasePrompts.planning_prompt_user

    def construct(
            self,
            user_request: str,
            previous_trace: list,
            observation: str,
            feedback: str = "",
            status_description: str = ""
    ) -> list:
        self.prompt_user = Template(self.prompt_user).render(
            user_request=user_request)
        if len(previous_trace) > 0:
            self.prompt_user += HistoryMemory(
                previous_trace=previous_trace, reflection=status_description).construct_previous_trace_prompt()
            if status_description != "":
                self.prompt_user += \
                    f"Task completion description is {status_description}"
            if feedback != "":
                self.prompt_user += f"Here are some other things you need to know:\n {feedback}\n"
            self.prompt_user += f"\nHere is the accessibility tree that you should refer to for this task:\n{observation}"
        messages = [{"role": "system", "content": self.prompt_system}, {
            "role": "user", "content": self.prompt_user}]
        return messages

    # Previous thought, action and reflection are converted to formatted strings
    def stringfy_thought_and_action(self, input_list: list) -> str:
        input_list = json5.loads(input_list, encoding="utf-8")
        str_output = "["
        for idx, i in enumerate(input_list):
            str_output += f'Step{idx + 1}:\"Thought: {i["thought"]}, Action: {i["action"]}, Reflection:{i["reflection"]}\";\n'
        str_output += "]"
        return str_output

class WebcanvasAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _construct_message(self) -> str:
        messages = PlanningPromptConstructor().construct(
            self.env_info['task_name'], 
            self.env_info['previous_trace'], 
            self.env_info['observation'], 
            self.env_info['feedback'], 
            self.env_info['status_description'])
        return messages
 
    def _extract_action(self, response: str) -> str:
        return response
 
    def call_api(self) -> str:
        messages = self._construct_message()
        try:
            logger.info("[UserAgent]: Calling OpenAI API")
            client = OpenAI(
                api_key=self.api_key,  # This is the default and can be omitted
            )

            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.9,
            )
            content = response.choices[0].message.content
            action = self._extract_action(content)
            logger.info(f"[UserAgent]: Got action: {action}")
            return action
        except Exception as e:
            logger.error(f"[UserAgent]: Error calling OpenAI API: {e}")
            raise