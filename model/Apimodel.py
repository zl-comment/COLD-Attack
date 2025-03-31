import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests


# ---------------------- 批量 API 调用实现 ----------------------
def call_api_completion(api_url, model_name, prompts, max_tokens=50, temperature=0.7):
    """
    批量调用外部 API 服务，接收 prompt 的列表，返回生成文本列表。
    构造 OpenAI 风格的请求，发送到 api_url。
    """
    payload = {
        "model": model_name,
        "prompt": prompts,  # 直接传入 prompt 列表
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1,
        "n": 1,
        "stream": False,
    }
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    response_json = response.json()
    # 假设 API 返回的 choices 是个列表，每个元素对应一个 prompt 的生成结果
    generated_texts = [choice["text"] for choice in response_json["choices"]]
    return generated_texts

