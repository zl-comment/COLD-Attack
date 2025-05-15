import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **kwargs
    ).to(device).eval()





    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
        use_cache=True,
    )

    if 'Mistral-7B-Instruct-v0.2' in tokenizer_path:

        # 设置 bos_token_id，unk_token_id，eos_token_id 和 pad_token_id
        tokenizer.bos_token_id = 1  # 序列开始标记的 ID，通常是 1
        tokenizer.unk_token_id = 0  # 未知标记的 ID，通常是 0
        tokenizer.eos_token_id = 2  # 序列结束标记的 ID，通常是 2
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 将填充标记 ID 设置为 eos_token_id，即 2
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
        tokenizer.padding_side = 'left'
    if 'Llama-2-7b-chat-hf' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'


    return model, tokenizer
