import re
from .base import JudgeBase


class LM(JudgeBase):
    def __init__(self, goal, model_name=None, max_new_tokens=10,
               temperature=0, **kwargs):
        super().__init__(goal, model_name)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        assert self.goal is not None

    def get_judge_prompt(self, attack_prompt, target_response):
        raise NotImplementedError()