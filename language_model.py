import os
import time
import torch
import gc
import openai
import anthropic

from typing import Dict, List


class LanguageModelBase:
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    def batch_generate(self, prompts: List, max_n_tokens: int, temperature: float):
        # generate responses for a batch of prompts
        raise NotImplementedError

    def evaluate_log_likelihood(self, prompt, output):
        # evaluate the per-token average likelihood of P(output|prompt)
        raise NotImplementedError


class HuggingFace(LanguageModelBase):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = [self.tokenizer.eos_token_id]

    def batch_generate(self, prompts: List, max_new_tokens: int, temperature: float, top_p: float = 1.0):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}


        # batch generation
        if temperature > 0:
            do_sample = True
        else:
            do_sample = False
            top_p = temperature = 1

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p
        )

        # if the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # batch decoding
        output_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for _ in inputs:
            inputs[_].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return output_list

    def extend_eos_tokens(self):
        # add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_id.extend([
            self.tokenizer.encode("}")[1],
            29913,
            9092,
            16675
        ])

    def evaluate_log_likelihood(self, prompt, output):
        inputs = [prompt + output]
        input_ids = self.tokenizer(
            inputs, padding=True, return_tensors="pt"
        ).input_ids.to(self.model.device)
        prompt_ids = self.tokenizer(
            [prompt], padding=True, return_tensors="pt"
        ).input_ids.to(self.model.device)
        prompt_length = prompt_ids.shape[1]
        input_ids = input_ids[:, :prompt_length+150]

        outputs = self.model(input_ids)
        probs = torch.log_softmax(outputs.logits, dim=-1).detach()


