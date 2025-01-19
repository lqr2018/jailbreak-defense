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

    def batch_generate(self,
                       prompts: List,
                       max_new_tokens: int,
                       temperature: float,
                       top_p: float = 1.0):
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

        # collect the probability of the generated token -- probability
        # at index 0 corresponds to the token at index 1
        probs = probs[:, prompt_length-1:-1, :]
        target_ids = input_ids[:, prompt_length:]
        gen_probs = torch.gather(probs, 2, target_ids[:, :, None])

        return gen_probs.mean()


class GPT(LanguageModelBase):
    API_ERROR_OUTPUT = "$ERROR$"

    def __init__(self, model_name, timeout=200, max_retry=5, retry_sleep=5):
        super().__init__(model_name)
        self.timeout = timeout
        self.max_retry = max_retry
        self.retry_sleep = retry_sleep
        self.client = openai.Client(
            api_key = os.environ.get("OPENAI_API_KEY"),
            timeout = timeout,
        )

    def generate(self,
                 conv: List[Dict],
                 max_new_tokens: int,
                 temperature: float,
                 top_p: float):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_new_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        # TODO add query_gap_time
        output = self.API_ERROR_OUTPUT
        for _ in range(self.max_retry):
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = conv,
                    max_tokens = max_new_tokens,
                    temperature = temperature,
                    top_p = top_p,
                )
                output = response.choices[0].message.content
                return output

            except openai.APIError as e:
                print(type(e), e)
                time.sleep(self.retry_sleep)

        return output

    def batch_generate(self,
                       prompts: List[List[Dict]],
                       max_new_tokens: int,
                       temperature: float,
                       top_p: float = 1.0):
        return [self.generate(prompt, max_new_tokens, temperature, top_p)
                for prompt in prompts]

class Claude(LanguageModelBase):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = os.getenv("ANTHROPIC_API_KEY")

    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = anthropic.Anthropic(api_key = self.API_KEY)

    def generate(self,
                 conv: List,
                 max_new_tokens: int,
                 temperature: float,
                 top_p: float = 1.0):
        """
        Args:
            conv: List of conversations
            max_new_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            time.sleep(self.API_QUERY_SLEEP)
            try:
                completion = self.model.completions.create(
                    model = self.model_name,
                    max_tokens_to_sample = max_new_tokens,
                    prompt = conv,
                    temperature = temperature,
                    top_p = top_p
                )
                output = completion.completion
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

        return output

    def batch_generate(self,
                       prompts: List[List[Dict]],
                       max_new_tokens: int,
                       temperature: float,
                       top_p: float = 1.0):
        return [self.generate(prompt, max_new_tokens, temperature, top_p)
                for prompt in prompts]




