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


def get_template_path(model_name):
    if model_name not in FULL_MODEL_DICT:
        print(f"No model with name '{model_name}' found in the model dict.")
        return None
    template = FULL_MODEL_DICT[model_name]["template"]

    if template == "llama-2" and version.parse(
        fastchat.__version__) < version.parse("0.2.24"):
        template = register_modified_llama_template()

    return template


