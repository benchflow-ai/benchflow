import logging
import os
import re

from openai import OpenAI

from benchflow import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebarenaAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.system_instruction = (
            """You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.
`hover [id]`: Hover over an element with id.
`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`scroll [down|up]`: Scroll the page up or down.

Tab Management Actions:
`new_tab`: Open a new, empty browser tab.
`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`close_tab`: Close the currently active tab.

URL Navigation Actions:
`goto [url]`: Navigate to a specific URL.
`go_back`: Navigate to the previously viewed page.
`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Homepage:
If you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.
http://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the websites.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside ``````. For example, "In summary, the next action I will perform is ```click [1234]```".
5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop."""
        )

        self.example1_user = (
            """OBSERVATION:\n[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'\n\t\t[1749] StaticText '$279.49'\n\t\t[1757] button 'Add to Cart'\n\t\t[1760] button 'Add to Wish List'\n\t\t[1761] button 'Add to Compare'\nURL: http://onestopmarket.com/office-products/office-electronics.html\nOBJECTIVE: What is the price of HP Inkjet Fax Machine\nPREVIOUS ACTION: None"""
        )

        self.example1_assistant = (
            """Let's think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer. In summary, the next action I will perform is ```stop [$279.49]```"""
        )

        self.example2_user = (
            """OBSERVATION:\n[164] textbox 'Search' focused: True required: False\n[171] button 'Go'\n[174] link 'Find directions between two points'\n[212] heading 'Search Results'\n[216] button 'Close'\nURL: http://openstreetmap.org\nOBJECTIVE: Show me the restaurants near CMU\nPREVIOUS ACTION: None"""
        )

        self.example2_assistant = (
            """Let's think step-by-step. This page has a search box whose ID is [164]. According to the nominatim rule of openstreetmap, I can search for the restaurants near a location by \"restaurants near\". I can submit my typing by pressing the Enter afterwards. In summary, the next action I will perform is ```type [164] [restaurants near CMU] [1]```"""
        )

    def _construct_message(self, env_info):
        return (
            f"OBSERVATION: {env_info['observation']['text']} "
            f"URL: {env_info['url']} "
            f"OBJECTIVE: {env_info['intent']} "
            f"PREVIOUS ACTION: {env_info['previous_action']}"
        )

    def _extract_action(self, response):
        pattern = r"```((.|\n)*?)```"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        raise Exception(f'Cannot find the action in "{response}"')

    def call_api(self, task_step_inputs):
        system_msg_1 = {"role": "system", "content": self.system_instruction}
        user_msg_1 = {"role": "system", "name": "example_user", "content": self.example1_user}
        system_msg_2 = {"role": "system", "name": "example_assistant", "content": self.example1_assistant}
        user_msg_2 = {"role": "system", "name": "example_user", "content": self.example2_user}
        system_msg_3 = {"role": "system", "name": "example_assistant", "content": self.example2_assistant}
        user_msg_final = {"role": "user", "content": self._construct_message(task_step_inputs)}

        messages = [
            system_msg_1,
            user_msg_1,
            system_msg_2,
            user_msg_2,
            system_msg_3,
            user_msg_final
        ]

        try:
            logger.info("[UserAgent]: Calling OpenAI API")
            client = OpenAI(
                api_key=self.api_key,  # This is the default and can be omitted
            )

            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4o",
                temperature=0.9,
            )
            content = response.choices[0].message.content
            action = self._extract_action(content)
            logger.info(f"[UserAgent]: Got action: {action}")
            return action
        except Exception as e:
            logger.error(f"[UserAgent]: Error calling OpenAI API: {e}")
            raise