import unittest
import requests
import logging
from typing import List, Tuple ,Optional
from transformers import LlamaTokenizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 LlamaTokenizer（请根据本地路径调整）
tokenizer = LlamaTokenizer.from_pretrained(r'D:\ZLCODE\model\Llama-2-7b-chat-hf')

# vLLM API 的 URL（请根据实际情况修改）
VLLM_COMPLETION_URL = "http://172.20.0.251:8000/v1/completions"


def call_api_completion(
    model_name: str,
    api_url: str,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: Optional[float] = None,
    stop: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> List[str]:
    """
    批量调用 vLLM/OpenAI 兼容的 Completions API。

    参数:
      model_name: 模型名称，如 "facebook/opt-125m"
      api_url:     API 的完整 URL，如 VLLM_COMPLETION_URL
      prompts:     文本列表，每个元素是一个 prompt
      max_tokens:  最大生成 token 数
      temperature: 采样温度
      top_p:       top-p 采样阈值（可选）
      stop:        停止词列表（可选）
      api_key:     API Key，如果服务器需要鉴权则提供（可选）

    返回:
      List[str]: 每个 prompt 对应的生成文本
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if stop is not None:
        payload["stop"] = stop

    resp = requests.post(api_url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    # vLLM 兼容 OpenAI，通常返回 'choices'
    if "choices" in data:
        # 支持 'text' 或 'completion' 字段
        results = []
        for choice in data["choices"]:
            text = choice.get("text") or choice.get("completion") or ""
            results.append(text)
        return results

    # vLLM 也可能返回 'completions'
    if "completions" in data:
        return data["completions"]

    raise ValueError(f"Unexpected response format: {data}")


def get_target_token_logprob(
    model_name: str,
    prompt: str,
    expected_token: str,
    api: str = VLLM_COMPLETION_URL,
    expected_token_id: int = None,
    temperature: float = 0.7,
    bias_value: float = 20
) -> Tuple[float, List[str]]:
    """
    调用 vLLM API 获取在给定 prompt 下生成下一个 token 的 logprob。
    如果第一次调用 top-logprobs 中包含 expected_token（strip 后对齐），则直接返回；
    否则第二次调用时向 expected_token_id 施加 logit_bias，再查看 top-logprobs。
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": temperature,
        "logprobs": 5,
        "echo": True
    }

    try:
        # 第一次调用：无 bias
        resp = requests.post(api, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # 确保有返回 choices
        choices = data.get("choices")
        if not choices:
            logger.error(f"No choices returned: {data}")
            return float('inf'), []
        lp_info = choices[0].get("logprobs", {})
        token_logprobs = lp_info.get("token_logprobs", [])
        top_logprobs_list = lp_info.get("top_logprobs", [])
        if not token_logprobs or not top_logprobs_list:
            logger.error(f"Incomplete logprobs from first call: {data}")
            return float('inf'), []

        # 检查最后一项 top_logprobs
        last_top = top_logprobs_list[-1]  # dict: token_str -> logprob
        top_tokens = list(last_top.keys())

        # 用 strip() 匹配原始 key
        tgt_stripped = expected_token.strip()
        for tok_str, logp in last_top.items():
            if tok_str.strip() == tgt_stripped:
                return logp, top_tokens

        # 第二次调用：施加 logit_bias
        if expected_token_id is None:
            ids = tokenizer.encode(expected_token, add_special_tokens=False)
            if not ids:
                logger.error(f"Cannot re-encode expected_token='{expected_token}'.")
                return float('inf'), []
            expected_token_id = ids[0]

        payload["logit_bias"] = {str(expected_token_id): bias_value}
        resp2 = requests.post(api, json=payload, timeout=30)
        resp2.raise_for_status()
        data2 = resp2.json()

        choices2 = data2.get("choices")
        if not choices2:
            logger.error(f"No choices on second call: {data2}")
            return float('inf'), []
        lp_info2 = choices2[0].get("logprobs", {})
        top2_list = lp_info2.get("top_logprobs", [])
        if not top2_list:
            logger.error(f"Incomplete top_logprobs on second call: {data2}")
            return float('inf'), []

        last2 = top2_list[-1]
        top2 = list(last2.keys())
        for tok_str, logp in last2.items():
            if tok_str.strip() == tgt_stripped:
                return logp, top2

        return float('inf'), top2

    except Exception as e:
        logger.error(f"Error in get_target_token_logprob for prompt '{prompt[:30]}...': {e}")
        return float('inf'), []


def compute_fine_grained_loss(
    model_name: str,
    base_prompt: str,
    target: str,
    api : str,
    temperature: float = 0.7

) -> Tuple[float, List[str]]:
    """
    对 base_prompt + target 的每个 token 计算负 logprob 平均值作为 loss。
    """
    # 去掉 target 两端空白
    target = target.strip()
    # 拆成 token id 列表
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        logger.warning("Target is empty after tokenization.")
        return float('inf'), []

    # 解码为 token 字符串
    target_tokens = [tokenizer.decode([tid]) for tid in target_ids]
    total_loss = 0.0
    predicted = []

    for i, tok in enumerate(target_tokens):
        # 构造带空格隔开的 prompt
        prefix = tokenizer.decode(target_ids[:i], skip_special_tokens=True)
        current_prompt = base_prompt.rstrip() + " " + prefix

        lp, topk = get_target_token_logprob(
            model_name=model_name,
            prompt=current_prompt,
            expected_token=tok,
            api=api,
            expected_token_id=target_ids[i],
            temperature=temperature
        )
        if lp == float('inf'):
            logger.warning(f"Failed to get logprob for token '{tok}' at pos {i}. topk={topk}")
            return float('inf'), predicted

        total_loss += -lp
        predicted.append(topk[0] if topk else "")

    avg_loss = total_loss / len(target_tokens)
    return avg_loss, predicted


class TestVLLMLossComputation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "deepseek-32b"  # 替换为你的模型名
        cls.base_prompt = """<<SYS>> You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible, while being safe. Please ensure that \
your responses are socially unbiased and positive in nature. If a question does \
not make any sense, or is not factually coherent, explain why instead of \
answering something not correct. If you don't know the answer to a question, \
please don't share false information. <</SYS>>

User: What is the capital of France?
Assistant:"""
        # 去掉前置空格
        cls.target = "Paris"

    def test_fine_grained_loss(self):
        loss, predicted = compute_fine_grained_loss(
            self.model_name,
            self.base_prompt,
            self.target,
            api=VLLM_COMPLETION_URL,
            temperature=0.7
        )
        print(f"Fine-grained loss: {loss}, Predicted tokens: {predicted}")
        # 断言 loss 为正，并非 inf
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

        prompts = [
            "Hello, how are you?",
            "What is the capital of France?"
        ]
        completions = call_api_completion(
            model_name=self.model_name,
            api_url=VLLM_COMPLETION_URL,
            prompts=prompts,
            max_tokens=50,
            temperature=0.7
        )
        for i, text in enumerate(completions):
            print(f"Prompt {i}: {prompts[i]}\nCompletion: {text}\n")


if __name__ == "__main__":
    unittest.main()
