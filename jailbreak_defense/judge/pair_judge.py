from fastchat.model import get_conversation_template
from .languagemodel import LMJudge
from ..language_models import GPT
from ..constant import pair_judge_template


class PairGPTJudge(LMJudge):
    """
    The system prompt is copied from https://github.com/patrickrchao/JailbreakingLLMs
    """
    def __init__(self, goal, model_name="gpt-4", **kwargs):
        super().__init__(goal, model_name)
        self.judge_model = GPT(model_name=model_name)
        self.system_message = pair_judge_template % self.goal
        self.judge_template = "[PROMPT]: %s\n[ASSISTANT'S RESPONSE]: %s"

    def prepare_conv(self, full_prompt):
        conv = get_conversation_template(self.model_name)
        conv.set_system_message(self.system_message)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def get_judge_prompt(self, attack_prompt, target_response):
        return self.judge_template % (attack_prompt, target_response)

    def score(self, prompt_list, response_list):
        conv_list, skipped_responses = self._prepare_conv_list(prompt_list, response_list)
        raw_outputs = self.judge_model.batch_generate(
            conv_list,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        outputs = [self._extract_content(raw_output) for raw_output in raw_outputs]
        for i in skipped_responses:
            outputs.insert(i, 1)
        return outputs

