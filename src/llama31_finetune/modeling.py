"""GPU-only helpers, imported only in worker processes."""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], token=os.environ.get("HF_TOKEN"))
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base(cfg, device):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg["revision"], token=os.environ.get("HF_TOKEN"),
        quantization_config=quant, torch_dtype=torch.float16,
        device_map={"": device}, attn_implementation="sdpa")
