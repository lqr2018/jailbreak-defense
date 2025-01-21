import copy
import torch
import fastchat

from packaging import version
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastchat.model import get_conversation_template
from fastchat.conversation import (Conversation, SeparatorStyle,
                                   register_conv_template, get_conv_template)
from language_model import GPT, Claude, HuggingFace

FULL_MODEL_DICT = {
    "gpt-4o-mini": {
        "path": "gpt-4o-mini",
        "template": "gpt-4"
    },
    "gpt-4o-mini": {
        "path": "gpt-4o-mini-2024-07-18",
        "template": "gpt-4"
    },
    "gpt-4o": {
        "path": "gpt-4o",
        "template": "gpt-4"
    },
    "gpt-4o-2024-08-06": {
        "path": "gpt-4o-2024-08-06",
        "template": "gpt-4"
    },
    "gpt-4o-2024-05-13": {
        "path": "gpt-4o-2024-05-13",
        "template": "gpt-4"
    },
    "gpt-4": {
        "path": "gpt-4",
        "template": "gpt-4"
    },
    "gpt-4-turbo": {
        "path": "gpt-4-turbo",
        "template": "gpt-4"
    },
    "gpt-4-turbo-2024-04-09": {
        "path": "gpt-4-turbo-2024-04-09",
        "template": "gpt-4"
    },
    "gpt-4-0613": {
        "path": "gpt-4-0613",
        "template": "gpt-4"
    },

    "gpt-3.5-turbo": {
        "path": "gpt-3.5-turbo-0613",
        "template": "gpt-3.5-turbo"
    },
    "gpt-3.5-turbo-0301": {
        "path": "gpt-3.5-turbo-0301",
        "template": "gpt-3.5-turbo"
    },
    "gpt-3.5-turbo-0613": {
        "path": "gpt-3.5-turbo-0613",
        "template": "gpt-3.5-turbo"
    },
    "gpt-3.5-turbo-1106": {
        "path": "gpt-3.5-turbo-1106",
        "template": "gpt-3.5-turbo"
    },
    "gpt-3.5-turbo-0125": {
        "path": "gpt-3.5-turbo-0125",
        "template": "gpt-3.5-turbo"
    },

    "vicuna": {
        "path": "lmsys/vicuna-13b-v1.5",
        "template": "vicuna_v1.1"
    },
    "vicuna-7b-v1.5": {
        "path": "lmsys/vicuna-7b-v1.5",
        "template": "vicuna_v1.1"
    },
    "vicuna-13b-v1.5": {
        "path": "lmsys/vicuna-13b-v1.5",
        "template": "vicuna_v1.1"
    },

    "llama-3.1-8b": {
        "path": "meta-llama/Llama-3.1-8B-Instruct",
        "template": "meta-llama-3.1",
    },
    "llama-3.1-70b": {
        "path": "meta-llama/Llama-3.1-70B-Instruct",
        "template": "meta-llama-3.1",
    },
    "llama-3.1-405b": {
        "path": "meta-llama/Llama-3.1-405B-Instruct",
        "template": "meta-llama-3.1",
    },
    "llama-3-8b": {
        "path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "template": "llama-3",
    },
    "llama-3-70b": {
        "path": "meta-llama/Meta-Llama-3-70B-Instruct",
        "template": "llama-3",
    },

    "llama-2": {
        "path": "meta-llama/Llama-2-13b-chat-hf",
        "template": "llama-2"
    },
    "llama-2-7b": {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "template": "llama-2"
    },
    "llama-2-13b": {
        "path": "meta-llama/Llama-2-13b-chat-hf",
        "template": "llama-2"
    },
    "llama-2-70b": {
        "path": "meta-llama/Llama-2-70b-chat-hf",
        "template": "llama-2"
    },

    "claude-instant-1": {
        "path": "claude-instant-1",
        "template": "claude-instant-1"
    },
    "claude-2": {
        "path": "claude-2",
        "template": "claude-2"
    }
}


def conv_template(template_name):
    if template_name == "llama-2-new":
        # for compatibility with GCG
        template = get_conv_template(template_name)
    else:
        template = get_conversation_template(template_name)
    if template.name.startswith("llama-2"):
        template.sep2 = template.sep2.strip()
    return template


def load_model(model_name, max_memory=None, quantization_config=None, **kwargs):
    model_path = get_model_path(model_name)
    if model_name.startswith("gpt-"):
        model = GPT(model_name, **kwargs)
    elif model_name.startswith("claude-"):
        model = Claude(model_name, **kwargs)
    else:
        if max_memory is not None:
            max_memory = {i: f"{max_memory}MB"
                          for i in range(torch.cuda.device_count())}
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
            quantization_config=quantization_config
        ).eval()
        tokenizer = load_tokenizer(model_path)
        model = HuggingFace(model, tokenizer)
    return model


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False
    )
    if "llama-3" in model_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    if "llama-2" in model_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    if "vicuna" in model_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def register_model_path_and_template(model_name, model_path, model_template):
    FULL_MODEL_DICT[model_name] = {
        "path": model_path,
        "template": model_template
    }


def register_modified_llama_template():
    print("Using fastchat 0.2.20. Removing the system prompt for LLaMA-2.")
    register_conv_template(
        Conversation(
            name="llama-2-new",
            system="<s>[INST] <<SYS>>\n\n<<SYS>>\n\n",
            roles=("[INST]", "[/INST]"),
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
            stop_token_ids=[2],
        ),
        override=True
    )
    template = ("llama-2-new")
    print("Using LLaMA-2 with fastchat < 0.2.24."
          f"Template changed to '{template}'.")
    return template


def get_model_path(model_name):
    if model_name not in FULL_MODEL_DICT:
        print(f"No model with name '{model_name}' found in the model dict.")
        return None
    return FULL_MODEL_DICT[model_name]["path"]


def get_template_name(model_name):
    if model_name not in FULL_MODEL_DICT:
        print(f"No model with name '{model_name}' found in the model dict.")
        return None
    template = FULL_MODEL_DICT[model_name]["template"]

    if template == "llama-2" and version.parse(
        fastchat.__version__) < version.parse("0.2.24"):
        template = register_modified_llama_template()

    return template


class TargetLM:
    """
    Base class for target language models.

    Generates responses for prompts using a language model.
    The self.model attribute contains the underlying generation model.
    """

    def __init__(self,
                 model_name: str = None,
                 max_new_tokens: int = 300,
                 max_memory: int = None,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 preloaded_model: object = None,
                 batch_size: int = 1,
                 template_name: str = None,
                 template: 'fastchat.conversation.Conversation' = None,
                 load_in_8bit: bool = False,
                 ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size

        if load_in_8bit:
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            self.quantization_config = None

        assert model_name is not None or preloaded_model is not None

        if preloaded_model is None:
            self.model = load_model(
                model_name,
                max_memory=max_memory,
                quantization_config=self.quantization_config
            )
        else:
            self.model = preloaded_model

        if template_name is not None:
            self.template = template
        else:
            if template_name is None:
                template_name = get_template_name(model_name)
            if template_name is not None:
                self.template = conv_template(template_name)
            else:
                self.template = None

    def get_response(self, prompts_list, template=None, verbose=False, **kwargs):
        only_one_prompt = isinstance(prompts_list, str)
        if only_one_prompt:
            prompts_list = [prompts_list]

        if template is None:
            template = self.template

        batch_size = len(prompts_list)
        convs_list = [copy.deepcopy(template) for _ in range(batch_size)]
        full_prompts = []
        if self.template is not None:
            for conv, prompt in zip(convs_list, prompts_list):
                if isinstance(prompt, str):
                    conv.append_message(conv.roles[0], prompt)
                elif isinstance(prompt, list):
                    for i, p in enumerate(prompt):
                        conv.append_message(conv.roles[i % 2], p)
                else:
                    raise NotImplementedError

                if self.model_name is not None and "gpt" in self.model_name:
                    # openai does not have separators
                    full_prompts.append(conv.to_openai_api_messages())
                else:
                    conv.append_message(conv.roles[1], None)
                    full_prompts.append(conv.get_prompt())

        else:
            convs = [
                [{'role': 'user','content': c}] for c in prompts_list
            ]
            assert hasattr(self.model, "tokenizer"), "The model must be a huggingface model with a tokenizer if no chat template is provided."
            full_prompts = [self.model.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                            for conv in convs]

        if verbose:
            print(f"Calling the TargetLM with {len(full_prompts)} prompts")
        output_list = []
        for i in tqdm(range((len(full_prompts) - 1) // self.batch_size + 1),
                      desc="Target model inference on batch: ",
                      disable=(not verbose)):
            # get the current batch of inputs
            batch = full_prompts[i * self.batch_size:(i + 1) * self.batch_size]
            batch_outputs = self.model.batch_generate(
                batch,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            output_list.extend(batch_outputs)

        if only_one_prompt:
            return output_list[0]
        else:
            return output_list

    def evaluate_log_likelihood(self, prompt, response):
        return self.model.evaluate_log_likelihood(prompt, response)
