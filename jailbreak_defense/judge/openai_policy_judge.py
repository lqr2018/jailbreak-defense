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

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.model_name)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def get_judge_prompt(self, attack_prompt, target_response):
        # TODO need to make sure if judge_input is correct
        judge_template = openai_judge_template
        judge_input = judge_template % (self.goal, target_response)
        return judge_input

    def _extract_content(self, tag, text):
        # find the starting position of the tag
        start_idx = text.find(tag)

        # if tag is not found, return None
        if start_idx == -1:
            return None

        # extract the content after the tag
        content_after_tag = text[start_idx + len(tag):].strip()

        # split the content by whitespace
        parts = content_after_tag.split()

        # if the first part (immediately after the tag) is a number, return it, else return the entire content after the tag
        if tag == "#thescore:":
            assert parts[0].isdigit()
            return int(parts[0])
        else:
            end_idx = text.find("#", start_idx + 1)
            return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()

    def score(self, prompt_list, response_list):
        conv_list, skipped_responses = self._get_conv_list(prompt_list, response_list)
        conv_list = [[conv[-1] for conv in conv_list]]
