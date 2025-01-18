import os
import time
import torch

import openai
import anthropic
import gc
from typing import Dict, List


class LanguageModelBase:
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    def batch_generate(self, prompts: List, max_n_tokens: int, temperature: float):
        # Generate responses for a batch of prompts
        raise NotImplementedError

    def evaluate_log_likelihood(self, prompt, output):
        # Evaluate the per-token average likelihood of P(output|prompt)
        raise NotImplementedError

class HuggingFace(LanguageModelBase):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = [self.tokenizer.eos_token_id]

    def batch_generate(self, prompts: List, max_n_tokens: int, temperature: float, top_p: float = 1.0):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

