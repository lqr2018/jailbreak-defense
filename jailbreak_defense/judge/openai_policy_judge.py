from fastchat.model import get_conversation_template
from .languagemodel import LMJudge
from ..language_model import GPT
from ..constant import openai_judge_template

class GPTJudge(LMJudge):
    """
    the OpenAI Policy is from
    https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/gpt-3.5/eval_utils/openai_policy_gpt4_judge.py
    """
    def __init__(self, goal, model_name="gpt-4", **kwargs):
        super().__init__(goal, model_name)
        self.judge_model = GPT(model_name=model_name)

    def create__conv(self, full_prompt):
        conv = get_conversation_template(self.model_name)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def get_judge_prompt(self, attack_prompt, target_response):
        # TODO need to make sure if judge_input is correct
        judge_template = openai_judge_template
        judge_input = judge_template % (self.goal, target_response)
        return judge_input

    def _process_output(self, text):

    def score(self, prompt_list, response_list):
