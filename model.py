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