import re
from fastchat.model import get_conversation_template
from .languagemodel import LMJudge
from jailbreak_defense.language_models import GPT
from .constant import quality_judge_template as template
from .constant import quality_judge_template_reference as template_reference


class QualityJudge(LMJudge):
    """
    Rate the answer from an ai assistant according to its quality, not harmfulness
    The prompt is copied from FastChat/llm_judge
    """
    def __init__(self, goal, model_name, **kwargs):
        super().__init__(goal, model_name)
        self.judge_model = GPT(model_name=model_name)
        self.system_prompt = "You are a helpful assistant"

    def get_judge_prompt(self, attack_prompt, target_response, reference_response=None):
        if reference_response is None:
            prompt = template % (attack_prompt, target_response)
        else:
            prompt = template_reference % (attack_prompt, reference_response, target_response)
        return prompt

    def prepare_conv(self, full_prompt):
        conv = get_conversation_template(self.model_name)
        conv.set_system_message(self.system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def score(self, prompt_list, response_list, reference_list=None):
        if reference_list is None:
            reference_list = [None] * len(prompt_list)
        conv_list = [
            self.prepare_conv(self.get_judge_prompt(prompt, response, reference_response))
            for prompt, response, reference_response in zip(prompt_list, response_list, reference_list)
        ]
        raw_outputs = self.judge_model.batch_generate(
            conv_list,
            max_new_tokens = self.max_new_tokens,
            temperature = self.temperature,
        )
        outputs = [self._extract_content(raw_output) for raw_output in raw_outputs]
        return outputs

    def _extract_content(self, text):
        pattern = r'\[\[(\d+)(\.?)(\d)*\]\]'
        match = re.search(pattern, text)
        output = float(match.group(1)) if match else None
        assert 10 >= output >= 1
        if output is None:
            print(f"Error in processing judge output: {text}" )
            output = -1
        return output



