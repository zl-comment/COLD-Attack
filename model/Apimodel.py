import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

# ---------------------- 用代理 tokenizer 解码的函数 ----------------------
def simple_decode(y_logits_single, tokenizer):
    """
    将单个样本的 y_logits（形状为 (seq_len, vocab_size)）通过 argmax 得到 token 序列，
    并利用传入的 tokenizer 进行解码（skip_special_tokens）。
    """
    token_ids = torch.argmax(y_logits_single, dim=-1)
    text = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
    return text

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

# ---------------------- 简单隐向量生成函数 ----------------------
def dummy_get_latent_representation(dummy_model, decoding_tokenizer, gen_text, latent_dim):
    """
    使用 Llama2-7B 模型提取文本的隐空间表示。

    参数：
      gen_text: 输入文本字符串
      latent_dim: 希望得到的隐空间向量维度

    返回：
      一个 torch.Tensor，形状为 (latent_dim,)
    """
    inputs = decoding_tokenizer(gen_text, return_tensors="pt")
    inputs = {k: v.to(dummy_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = dummy_model(**inputs, output_hidden_states=True)

    # 获取最后一层隐藏状态，形状 [batch, seq_len, hidden_dim]
    hidden_states = outputs.hidden_states[-1]
    # 平均池化得到固定向量，形状 [hidden_dim]
    latent_vector = hidden_states.mean(dim=1).squeeze(0)
    current_dim = latent_vector.shape[0]
    if current_dim != latent_dim:
        # 将 projection 层移动到 latent_vector 所在的设备上，确保两者在同一设备
        projection = nn.Linear(current_dim, latent_dim).to(latent_vector.device)
        projection.eval()
        with torch.no_grad():
            latent_vector = projection(latent_vector)
    return latent_vector

# ---------------------- 批量 API 调用 + Critic 损失计算 ----------------------
def api_call_future_cost(api_url, model_name, dummy_model, y_logits, topk, soft_forward_x, x_model_past,
                         decoding_tokenizer, device, safety_budget_value, critic_model, latent_dim,
                         max_tokens=50, temperature=0.7):
    """
    对于每个样本：
      1. 使用 decoding_tokenizer 将 y_logits 解码为 prompt 文本；
      2. 批量调用 API 得到生成文本；
      3. 使用 dummy_get_latent_representation 将生成文本转换为隐向量；
      4. 将隐向量与安全预算拼接后送入 Critic 网络，得到未来成本（critic loss）。
    返回的 cost_tensor 形状为 [batch, 1]，同时返回生成文本列表。
    """
    batch_size = y_logits.shape[0]
    # 批量解码得到 prompt 文本
    prompt_texts = [simple_decode(y_logits[i], decoding_tokenizer) for i in range(batch_size)]
    try:
        generated_texts = call_api_completion(api_url, model_name, prompt_texts, max_tokens, temperature)
    except Exception as e:
        print(f"API 调用失败，使用空文本作为返回: {e}")
        generated_texts = ["" for _ in range(batch_size)]
    latent_list = []
    for gen_text in generated_texts:
        latent = dummy_get_latent_representation(dummy_model, decoding_tokenizer, gen_text, latent_dim)
        latent_list.append(latent)
    batch_latents = torch.stack(latent_list)  # [batch, latent_dim]
    safety_tensor = torch.tensor([safety_budget_value] * batch_size, dtype=torch.float32, device=device).unsqueeze(1)
    critic_input = torch.cat([batch_latents, safety_tensor], dim=1)
    # Critic 网络假设返回 (safety_prob, future_cost_pred)
    _, future_cost_pred = critic_model(critic_input)
    return future_cost_pred, generated_texts

# ---------------------- 自定义黑盒梯度估计 (APIDecodeWithGraph) ----------------------
class APIDecodeWithGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, api_url, model_name, dummy_model, y_logits, soft_forward_x, x_model_past,
                decoding_tokenizer, topk, safety_budget_value, critic_model, device, latent_dim,
                num_spsa_samples=2, c=1e-3, max_tokens=50, temperature=0.7):
        """
        forward:
          1. 使用当前 y_logits 通过 API 得到基准成本 f0；
          2. 对 y_logits 施加 SPSA 扰动，分别得到 f_plus 和 f_minus，然后计算梯度近似：
             grad ≈ (f_plus - f_minus) / (2c) * δ ；
          3. 保存梯度估计供 backward 使用，并返回 f0。
        """
        f0, _ = api_call_future_cost(
            api_url, model_name, dummy_model, y_logits, topk, soft_forward_x, x_model_past,
            decoding_tokenizer, device, safety_budget_value, critic_model, latent_dim,
            max_tokens, temperature
        )
        grad_estimate = torch.zeros_like(y_logits)
        for _ in range(num_spsa_samples):
            delta = torch.sign(torch.randn_like(y_logits))
            y_logits_plus = y_logits + c * delta
            y_logits_minus = y_logits - c * delta
            f_plus, _ = api_call_future_cost(
                api_url, model_name, dummy_model, y_logits_plus, topk, soft_forward_x, x_model_past,
                decoding_tokenizer, device, safety_budget_value, critic_model, latent_dim,
                max_tokens, temperature
            )
            f_minus, _ = api_call_future_cost(
                api_url, model_name, dummy_model, y_logits_minus, topk, soft_forward_x, x_model_past,
                decoding_tokenizer, device, safety_budget_value, critic_model, latent_dim,
                max_tokens, temperature
            )
            # 将 (f_plus - f_minus) 强制变为形状 [batch, 1, 1] 以便与 delta 相乘
            diff = (f_plus - f_minus).view(f_plus.shape[0], 1).unsqueeze(1)
            grad_estimate += diff / (2 * c) * delta
        grad_estimate = grad_estimate / num_spsa_samples
        ctx.save_for_backward(grad_estimate)
        return f0

    @staticmethod
    def backward(ctx, grad_future_cost):
        (grad_estimate,) = ctx.saved_tensors
        while grad_future_cost.dim() < grad_estimate.dim():
            grad_future_cost = grad_future_cost.unsqueeze(-1)
        grad_y_logits = grad_estimate * grad_future_cost
        # 此 forward 共有 16 个参数（不含 ctx），所以 backward 返回长度为 16 的元组
        return (None, None, None, grad_y_logits, None, None, None, None, None, None,
                None, None, None, None, None, None)

# ===================== 使用示例 =====================
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # API 地址和模型名称
#     api_url = "http://172.20.0.251:8000/v1/completions"
#     model_name = "deepseek-32b"
#     # 此处 dummy_model 在纯 API 模式下可以设为 None
#     dummy_model = None
#     # 假设对抗攻击中待优化的 soft prompt logits，batch_size=8，序列长度=10，词表大小=代理模型的 vocab_size
#     batch_size = 8
#     seq_len = 10
#     # 假设代理模型的词表大小，比如代理模型使用的 tokenizer 返回的 vocab_size
#     # 此处我们假设为32000（实际使用时请替换为 proxy_tokenizer.vocab_size）
#     vocab_size = 32000
#     y_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
#     # soft_forward_x 与 x_model_past 在此示例中不使用
#     soft_forward_x = None
#     x_model_past = None
#     # 假设 topk 参数
#     topk = 10
#     # 安全预算
#     safety_budget_value = 0.0
#     # Critic 网络：示例中构造一个简单的 Critic 网络
#     class Critic(nn.Module):
#         def __init__(self, input_dim, hidden_dim):
#             super(Critic, self).__init__()
#             self.shared = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.ReLU()
#             )
#             self.safety_head = nn.Sequential(
#                 nn.Linear(hidden_dim, 1),
#                 nn.Sigmoid()
#             )
#             self.cost_head = nn.Linear(hidden_dim, 1)
#         def forward(self, x):
#             shared_out = self.shared(x)
#             safety_prob = self.safety_head(shared_out)
#             future_cost = self.cost_head(shared_out)
#             return safety_prob, future_cost
#     latent_dim = 128
#     critic_input_dim = latent_dim + 1
#     critic_model = Critic(critic_input_dim, 64).to(device)
#     # 其他参数
#     num_spsa_samples = 3
#     c = 1e-3
#     max_tokens = 50
#     temperature = 0.7
#
#     # 这里使用代理模型的 tokenizer 作为解码器，因为 y_logits 是代理生成的
#     # 假设 proxy_tokenizer 已加载，下面仅为示例：
#     from transformers import AutoTokenizer
#     proxy_tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 示例，请替换为你的代理模型 tokenizer
#
#     # 调用 APIDecodeWithGraph，注意此处我们将代理 tokenizer 作为解码器传入
#     future_cost_pred = APIDecodeWithGraph.apply(
#         api_url,           # API 地址
#         model_name,        # 模型名称
#         dummy_model,       # dummy_model（可设为 None）
#         y_logits,          # 待优化的 soft prompt logits
#         soft_forward_x,    # 不使用
#         x_model_past,      # 不使用
#         proxy_tokenizer,   # 使用代理模型的 tokenizer进行解码
#         topk,              # topk 参数
#         safety_budget_value,
#         critic_model,      # Critic 网络
#         device,
#         latent_dim,
#         num_spsa_samples,
#         c,
#         max_tokens,
#         temperature
#     )
#     # future_cost_pred 形状为 [batch, 1]，代表 Critic 损失
#     loss = future_cost_pred.mean()
#     loss.backward()
#     print("Critic Loss:", loss.item())
#     print("Gradient norm on y_logits:", y_logits.grad.norm().item())
