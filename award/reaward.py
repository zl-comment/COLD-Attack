import math
import re
import torch
import torch.nn.functional as F
import time
from award.utils import ReturnStruct
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# -----------------------------------------
# 1. 定义辅助函数（分词、编辑距离、SWES、奖励计算、动态温度）
# -----------------------------------------
def get_reject_token_ids(target_tokenizer, reject_words):
    """将拒绝词列表转换为目标模型的 token IDs"""
    reject_token_ids = set()
    for word in reject_words:
        tokens = target_tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) > 0:
            # 添加所有 token（处理多 token 词汇）
            reject_token_ids.update(tokens)
    return list(reject_token_ids)

#
# def compute_rejection_prob_loss(
#         y_logits,
#         proxy_tokenizer,
#         target_model,
#         target_tokenizer,
#         target_device,
#         reject_token_ids  # 预计算的拒绝 token IDs 列表
# ):
#     """ 计算目标模型对生成文本的拒绝概率 """
#     # 解码生成文本
#     texts = [proxy_tokenizer.decode(ids, skip_special_tokens=True)
#              for ids in torch.argmax(y_logits, dim=-1)]
#
#     # 目标模型推理
#     inputs = target_tokenizer(
#         texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True
#     ).to(target_device)
#
#     with torch.no_grad():
#         outputs = target_model(**inputs)
#         logits = outputs.logits  # [batch, seq_len, vocab]
#
#         # 计算所有拒绝 token 的概率
#         reject_probs = torch.zeros(logits.shape[0], device=target_device)
#         for idx in range(logits.shape[1]):  # 遍历每个 token 位置
#             token_probs = F.softmax(logits[:, idx, :], dim=-1)
#             reject_probs += token_probs[:, reject_token_ids].sum(dim=-1)  # 累加所有拒绝词概率
#
#
#         # 平均每个 token 的拒绝概率
#         reject_probs /= logits.shape[1]
#
#     return reject_probs.mean()
#
def compute_rejection_prob_loss(
        y_logits,
        proxy_tokenizer,
        target_model,
        target_tokenizer,
        target_device,
        reject_token_ids  # 预计算的拒绝 token IDs 列表
):
    """ 计算目标模型对生成文本的拒绝概率（修改为关注局部最高概率）
    通过对每个样本中所有 token 的拒绝词概率取最大值来反映局部高概率问题
    """
    from transformers import logging
    logging.set_verbosity_error()

    # 解码生成文本
    texts = [proxy_tokenizer.decode(ids, skip_special_tokens=True)
             for ids in torch.argmax(y_logits, dim=-1)]

    # 目标模型推理
    inputs = target_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(target_device)

    with torch.no_grad():
        outputs = target_model(**inputs)
        logits = outputs.logits  # [batch, seq_len, vocab]

        # 对每个 token 计算拒绝词的概率
        token_reject_probs = []
        for idx in range(logits.shape[1]):  # 遍历每个 token 位置
            token_probs = F.softmax(logits[:, idx, :], dim=-1)
            # 累加该位置上所有拒绝词的概率
            token_reject = token_probs[:, reject_token_ids].sum(dim=-1)  # [batch]
            token_reject_probs.append(token_reject.unsqueeze(1))
        # 拼接成 [batch, seq_len]
        token_reject_probs = torch.cat(token_reject_probs, dim=1)
        # 取每个样本中拒绝概率最高的那个 token
        max_reject_probs = token_reject_probs.max(dim=1)[0]  # [batch]

    return max_reject_probs.mean()


def compute_cw_loss(z_logits, z_t, topk=5, margin=1.0):
    target_logits = torch.gather(z_logits, -1, z_t.unsqueeze(-1)).squeeze(-1)
    mask = torch.ones_like(z_logits, dtype=torch.bool)
    mask.scatter_(-1, z_t.unsqueeze(-1), False)
    other_logits = z_logits.masked_fill(~mask, -float('inf'))
    topk_logits = other_logits.topk(topk, dim=-1).values  # 取 Top-K 非目标 logits
    cw_loss = F.relu(topk_logits - target_logits.unsqueeze(-1) + margin).mean()  # 多 token 对比
    return cw_loss

def compute_semantic_reject_loss(y_logits, bad_word_ids, embedding_layer, threshold=0.8):
    """
    y_logits: [batch, seq_len, vocab_size]
    bad_word_ids: List[int] 一维的拒绝词 ID 列表（如 [1, 7423, 27746]）
    embedding_layer: 代理模型的 token 嵌入层
    """
    # 检查输入合法性
    assert isinstance(bad_word_ids, list), "bad_word_ids 必须是 List[int]"
    assert all(isinstance(x, int) for x in bad_word_ids), "bad_word_ids 必须全为整数"

    # 获取预测 token 的嵌入
    pred_ids = torch.argmax(y_logits, dim=-1)  # [batch, seq_len]
    pred_embeddings = embedding_layer(pred_ids)  # [batch, seq_len, emb_dim]

    # 获取拒绝词的嵌入（处理梯度警告）
    with torch.no_grad():
        bad_tensor = torch.tensor(bad_word_ids, device=y_logits.device)
        bad_embeddings = embedding_layer(bad_tensor)  # [num_bad, emb_dim]

    # 调整维度以支持广播
    pred_emb = pred_embeddings.unsqueeze(2)  # [batch, seq_len, 1, emb_dim]
    bad_emb = bad_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, num_bad, emb_dim]

    # 计算余弦相似度（自动广播）
    sim = F.cosine_similarity(pred_emb, bad_emb, dim=-1)  # [batch, seq_len, num_bad]

    # 惩罚高相似度
    return F.relu(sim.max(dim=-1).values - threshold).mean()

DEFAULT_HARM_SAFE_MAPPING = {
    "hack": "explore",
    "hacking": "studying",
    "steal": "learn",
    "stealing": "reading",
    "bomb": "cake",
    "virus": "program",
    "weapon": "tool",
    "identity theft": "identity protection",
    "insider trading": "financial literacy",
    "fraud": "financial planning",
    "counterfeit": "art project",
    "assassinate": "interview",
    "kill": "cook",
    "poison": "season",
    "explosive": "baking",
    "suicide": "mental health",
    "anorexia": "healthy diet",
    "bulimia": "nutritional wellness",
    "ransomware": "backup tool",
    "pirate": "download",
    "drugs": "vitamins",
    "cheat": "practice",
    "child pornography": "educational materials",
    "firearms": "sports equipment",
    "malware": "software",
    "phishing": "marketing",
    "stalk": "follow",
    "harass": "contact",
    "hate speech": "debate",
    "defame": "review",
    "sabotage": "debug",
    "infiltrate": "join",
    "threatening": "motivational",
    "poisoning": "cooking",
    "terrorist": "research group",
    "illegal": "complex",
    "crime": "task",
    "violence": "competition",
    "glorify": "simulate",
    "incite": "discuss",
    "panic": "surprise",
}

def build_safe_prompt(harm_prompt: str, mapping: dict = None, safe_prompt_from_data: str = None) -> str:
    """
    优先使用数据集中提供的 safe_prompt（如 goal 列）。
    否则根据 mapping 替换关键词生成安全提示。
    """
    if safe_prompt_from_data:
        return safe_prompt_from_data

    if mapping is None:
        mapping = DEFAULT_HARM_SAFE_MAPPING

    safe_prompt = harm_prompt
    for harm_word, safe_word in mapping.items():
        safe_prompt = re.sub(rf'\b{re.escape(harm_word)}\b', safe_word, safe_prompt, flags=re.IGNORECASE)

    return safe_prompt


def compute_adv_loss_from_safe_baseline_min_gap(
    y_logits_,
    harm_prompt: str,
    proxy_tokenizer,
    target_model,
    target_tokenizer,
    safe_prompt: str = None,
    loss_params: dict = None,
    mapping_dict: dict = None,
    proxy_device="cuda:0",
    target_device="cuda:1",
    max_length=128,           # 输入最大长度，用于编码 prompt
    use_kl=False,             # 是否启用 KL 散度
    classify_token_length=15  # 用于二分类器的生成文本中新生成的 token 数量（比如 10 或 15）
):
    if target_tokenizer.pad_token_id is None:
        target_tokenizer.pad_token_id = target_tokenizer.eos_token_id



    if loss_params is None:
        loss_params = {"hard_labels": True, "reweight_loss": False, "reduction": "none"}

    # 1. 解码扰动后缀（保持原有流程）
    with torch.device(proxy_device):
        pred_ids = torch.argmax(y_logits_, dim=-1)
        suffix_texts = [proxy_tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]

    # 2. 拼接 harm_prompt 与扰动后缀，构成 harm+后缀 文本；同时构造安全端文本（利用 build_safe_prompt 构造安全提示）
    harm_prompts = [harm_prompt + suffix for suffix in suffix_texts]
    resolved_safe_prompt = build_safe_prompt(harm_prompt, mapping=mapping_dict, safe_prompt_from_data=safe_prompt)
    safe_prompts = [resolved_safe_prompt for _ in harm_prompts]

    # 3. 对齐输入长度并编码为 target_model 的输入
    harm_inputs = target_tokenizer(
        harm_prompts,
        return_tensors='pt',
        padding='max_length',
        max_length=max_length,
        truncation=True
    ).to(target_device)
    safe_inputs = target_tokenizer(
        safe_prompts,
        return_tensors='pt',
        padding='max_length',
        max_length=max_length,
        truncation=True
    ).to(target_device)

    # 4. 使用 target_model 生成文本，注意使用 max_new_tokens 生成新增的 token 部分
    generated_harm_ids = target_model.generate(
        **harm_inputs,
        max_new_tokens=classify_token_length,
        do_sample=False,  # 强制使用贪心搜索
        pad_token_id=target_tokenizer.pad_token_id  # 确保pad_token_id正确设置
    )
    generated_safe_ids = target_model.generate(
        **safe_inputs,
        max_new_tokens=classify_token_length,
        do_sample=False,
        pad_token_id=target_tokenizer.pad_token_id
    )

    # 5. 提取新增生成的 token（即剔除 prompt 部分），作为二分类器输入
    harm_prompt_length = harm_inputs.input_ids.shape[1]
    safe_prompt_length = safe_inputs.input_ids.shape[1]
    harm_new_tokens = generated_harm_ids[:, harm_prompt_length:]
    safe_new_tokens = generated_safe_ids[:, safe_prompt_length:]

    # 如果生成的 token 数不足 classify_token_length，则进行截断或补齐处理（此处假设生成数量足够）
    harm_class_ids = harm_new_tokens[:, :classify_token_length]
    safe_class_ids = safe_new_tokens[:, :classify_token_length]
    harm_attention_mask = (harm_class_ids != target_tokenizer.pad_token_id).long()
    safe_attention_mask = (safe_class_ids != target_tokenizer.pad_token_id).long()

    # 6. 加载微调后的二分类器（distilbert_suffix.pt）
    # 采用如下方式加载：先构造预训练模型，再加载 state_dict
    classifier = DistilBertForSequenceClassification.from_pretrained('/home/zl/ZLCODE/model/distilbert-base-uncased')
    classifier.load_state_dict(torch.load("/home/zl/ZLCODE/COLD-Attack/award/distilbert_suffix.pt", map_location=target_device))
    classifier.to(target_device)
    classifier.eval()

    # 7. 使用二分类器得到 logits，形状为 (batch, 2)
    #    规则：若 logits 中 v[0] > v[1] 则判定为有害
    v_harm = classifier(input_ids=harm_class_ids, attention_mask=harm_attention_mask).logits
    v_safe = classifier(input_ids=safe_class_ids, attention_mask=safe_attention_mask).logits

    # 8. 计算交叉熵损失：
    #    harm 分支目标标签为 0（有害），safe 分支目标标签为 1（安全）
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    harm_targets = torch.zeros(v_harm.size(0), dtype=torch.long).to(target_device)
    safe_targets = torch.zeros(v_safe.size(0), dtype=torch.long).to(target_device)
    harm_loss = criterion(v_harm, harm_targets)  # 每个样本的损失
    safe_loss = criterion(v_safe, safe_targets)

    # 9. 可选：计算 KL 散度（这里采用 harm 与 safe 分支在 target_model 前向传播得到的输出）
    if use_kl:
        harm_forward = target_model(**harm_inputs)
        safe_forward = target_model(**safe_inputs)
        # 去除最后一个 token（与原来保持一致）
        harm_logits = harm_forward.logits[:, :-1, :]
        safe_logits = safe_forward.logits[:, :-1, :]
        harm_probs = F.softmax(harm_logits, dim=-1)
        safe_probs = F.softmax(safe_logits, dim=-1)
        kl_loss = F.kl_div(harm_probs.log(), safe_probs, reduction='batchmean')
    else:
        kl_loss = 0.0

    # 10. 组合损失：目标为让 harm 分支（经对抗生成后）越来越趋向不安全（即 harm_loss 降低），而 safe 分支损失较大
    adv_loss = (harm_loss - safe_loss).mean() + kl_loss

    return ReturnStruct(
        loss=adv_loss.to(proxy_device),
        pred=pred_ids.to(proxy_device)
    )


#
# def compute_adv_loss_from_safe_baseline_min_gap(
#     y_logits_,
#     harm_prompt: str,
#     proxy_tokenizer,
#     target_model,
#     target_tokenizer,
#     safe_prompt: str = None,
#     loss_params: dict = None,
#     mapping_dict: dict = None,
#     proxy_device="cuda:0",
#     target_device="cuda:1",
#     max_length=128,  # 新增参数
#     use_kl=False,     # 是否启用 KL 散度
#     bad_word_ids=None,  # 新增参数：拒绝词ID列表
#     threshold=0.8     # 语义拒绝的阈值
# ):
#     if loss_params is None:
#         loss_params = {"hard_labels": True, "reweight_loss": False, "reduction": "none"}
#
#     # 解码扰动后缀（同样保持原来的做法）
#     with torch.device(proxy_device):
#         pred_ids = torch.argmax(y_logits_, dim=-1)
#         suffix_texts = [proxy_tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]
#
#     # 拼接 harm prompt + 后缀
#     harm_prompts = [harm_prompt + suffix for suffix in suffix_texts]
#     resolved_safe_prompt = build_safe_prompt(harm_prompt, mapping=mapping_dict, safe_prompt_from_data=safe_prompt)
#     safe_prompts = [resolved_safe_prompt] * len(harm_prompts)
#
#     # 对齐输入长度
#     harm_inputs = target_tokenizer(
#         harm_prompts,
#         return_tensors='pt',
#         padding='max_length',
#         max_length=max_length,
#         truncation=True
#     ).to(target_device)
#     safe_inputs = target_tokenizer(
#         safe_prompts,
#         return_tensors='pt',
#         padding='max_length',
#         max_length=max_length,
#         truncation=True
#     ).to(target_device)
#
#     # 模型推理得到logits（去掉最后一个token以便于教师强制）
#     harm_out = target_model(**harm_inputs)
#     safe_out = target_model(**safe_inputs)
#
#     harm_logits = harm_out.logits[:, :-1, :]
#     safe_logits = safe_out.logits[:, :-1, :]
#
#     # 获取目标模型的嵌入层
#     embedding_layer = target_model.get_input_embeddings()
#
#     # 如果未提供拒绝词ID，则使用默认的几个示例拒绝词
#     if bad_word_ids is None:
#         default_bad_words = ['sorry', 'cannot', 'unethical', 'illegal']
#         bad_word_ids = []
#         for word in default_bad_words:
#             ids = target_tokenizer.encode(word, add_special_tokens=False)
#             bad_word_ids.extend(ids)
#         bad_word_ids = list(set(bad_word_ids))
#
#     # 使用 compute_semantic_reject_loss 计算 harm 与 safe 两端的语义拒绝损失
#     harm_loss = compute_semantic_reject_loss(harm_logits, bad_word_ids, embedding_layer, threshold=threshold)
#     safe_loss = compute_semantic_reject_loss(safe_logits, bad_word_ids, embedding_layer, threshold=threshold)
#
#     # 可选：计算 KL 散度
#     if use_kl:
#         harm_probs = F.softmax(harm_logits, dim=-1)
#         safe_probs = F.softmax(safe_logits, dim=-1)
#         kl_loss = F.kl_div(harm_probs.log(), safe_probs, reduction='batchmean')
#     else:
#         kl_loss = 0.0
#
#     # 组合损失：差值越低，说明 harm 端越容易回答（语义上更远离拒绝词）
#     adv_loss = (harm_loss - safe_loss) + kl_loss
#
#     return ReturnStruct(
#         loss=adv_loss.to(proxy_device),
#         pred=pred_ids.to(proxy_device)
#     )

#
# def compute_adv_loss_from_safe_baseline_min_gap(
#     y_logits_,
#     harm_prompt: str,
#     proxy_tokenizer,
#     target_model,
#     target_tokenizer,
#     safe_prompt: str = None,
#     loss_params: dict = None,
#     mapping_dict: dict = None,
#     proxy_device="cuda:0",
#     target_device="cuda:1",
#     max_length=128,  # 新增参数
#     use_kl=False      # 是否启用 KL 散度
# ):
#     if loss_params is None:
#         loss_params = {"hard_labels": True, "reweight_loss": False, "reduction": "none"}
#
#     # 解码扰动后缀
#     with torch.device(proxy_device):
#         pred_ids = torch.argmax(y_logits_, dim=-1)
#         suffix_texts = [proxy_tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]
#     # print("resolved_safe_prompt")
#     harm_prompts = [harm_prompt + suffix for suffix in suffix_texts]
#     safe_prompts = [build_safe_prompt(harm_prompt, mapping=mapping_dict, safe_prompt_from_data=safe_prompt) + suffix for suffix in suffix_texts]
#
#     # 对齐输入长度
#     harm_inputs = target_tokenizer(
#         harm_prompts,
#         return_tensors='pt',
#         padding='max_length',  # 填充到统一长度
#         max_length=max_length,  # 显式指定最大长度
#         truncation=True
#     ).to(target_device)
#     safe_inputs = target_tokenizer(
#         safe_prompts,
#         return_tensors='pt',
#         padding='max_length',
#         max_length=max_length,
#         truncation=True
#     ).to(target_device)
#
#     # 模型推理
#     harm_out = target_model(**harm_inputs)
#     safe_out = target_model(**safe_inputs)
#
#     # 取 logits（自动对齐长度）
#     harm_logits = harm_out.logits[:, :-1, :]
#     safe_logits = safe_out.logits[:, :-1, :]
#
#     # 计算交叉熵损失
#     harm_loss = F.cross_entropy(harm_logits.transpose(1, 2), harm_inputs['input_ids'][:, 1:], reduction='none')
#     safe_loss = F.cross_entropy(safe_logits.transpose(1, 2), safe_inputs['input_ids'][:, 1:], reduction='none')
#
#     # 计算 KL 散度（可选）
#     if use_kl:
#         harm_probs = F.softmax(harm_logits, dim=-1)
#         safe_probs = F.softmax(safe_logits, dim=-1)
#         kl_loss = F.kl_div(harm_probs.log(), safe_probs, reduction='batchmean')
#     else:
#         kl_loss = 0.0
#
#     # 组合损失
#     adv_loss = (harm_loss.mean() - safe_loss.mean()) + kl_loss
#
#     return ReturnStruct(
#         loss=adv_loss.to(proxy_device),
#         pred=pred_ids.to(proxy_device)
#     )


def compute_remiss_style_adv_loss(
    y_logits_,
    proxy_tokenizer,
    target_model,
    target_tokenizer,
    baseprompt,                     # base prompt: 原始 harmful prompt
    loss_params=None,
    proxy_device="cuda:0",
    target_device="cuda:1",
):
    """
    结合 reMissOpt 和 compute_base_loss 的对抗奖励函数。
    返回完整结构（disturbed_loss, baseline_loss, adv_loss）。
    """

    if loss_params is None:
        loss_params = {
            'hard_labels': True,
            'reweight_loss': False,
        }

    target_model = target_model.to(target_device)

    # 1. 解码扰动 token -> suffix text
    with torch.device(proxy_device):
        pred_ids = torch.argmax(y_logits_, dim=-1)  # (B, T)
        suffix_texts = [proxy_tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]

    # 2. 拼接 harmful prompt + suffix
    if isinstance(baseprompt, str):
        prompts = [baseprompt + suffix for suffix in suffix_texts]
        base_prompts = [baseprompt] * len(suffix_texts)
    else:
        prompts = [bp + suffix for bp, suffix in zip(baseprompt, suffix_texts)]
        base_prompts = baseprompt

    # 3. tokenizer 编码，送入目标模型（扰动后的输入）
    disturbed_inputs = target_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    disturbed_inputs = {k: v.to(target_device) for k, v in disturbed_inputs.items()}

    baseline_inputs = target_tokenizer(base_prompts, return_tensors="pt", padding=True, truncation=True)
    baseline_inputs = {k: v.to(target_device) for k, v in baseline_inputs.items()}

    # 4. 使用 teacher forcing 计算两组 loss
    with torch.no_grad():
        disturbed_outputs = target_model(
            input_ids=disturbed_inputs["input_ids"],
            attention_mask=disturbed_inputs["attention_mask"],
            labels=disturbed_inputs["input_ids"]
        )
        disturbed_loss_batch = disturbed_outputs.loss

        baseline_outputs = target_model(
            input_ids=baseline_inputs["input_ids"],
            attention_mask=baseline_inputs["attention_mask"],
            labels=baseline_inputs["input_ids"]
        )
        baseline_loss_batch = baseline_outputs.loss

    # 5. 计算对抗差异 reward
    adv_loss = disturbed_loss_batch - baseline_loss_batch

    return ReturnStruct(
        loss=adv_loss.to(proxy_device),
        disturbed_loss=disturbed_loss_batch.to(proxy_device),
        baseline_loss=baseline_loss_batch.to(proxy_device),
        pred=pred_ids.to(proxy_device),  # 这个 pred 实际是 suffix 的 token ids
        label=disturbed_inputs["input_ids"][:, 1:].to(proxy_device),
        mask=disturbed_inputs["attention_mask"][:, 1:].to(proxy_device),
    )


def compute_base_loss(
        y_logits_,
        proxy_tokenizer,
        target_model,
        target_tokenizer,
        baseprompt,  # 新增：基础提示词，字符串或字符串列表（batch_size）
        loss_params=None,
        proxy_device="cuda",
        target_device="cuda"
):
    """
    支持proxy和target在不同设备的对抗损失计算，同时引入奖励基线偏移。

    参数:
        y_logits_: 扰动生成的logits [batch_size, seq_len, proxy_vocab_size] (在proxy_device上)
        proxy_tokenizer: 代理模型的tokenizer
        target_model: 目标模型 (将自动移动到target_device)
        target_tokenizer: 目标tokenizer
        baseprompt: 基础提示词（不包含扰动后缀），字符串或字符串列表，长度等于batch_size
        loss_params: 损失参数配置字典
        proxy_device: proxy模型所在的设备
        target_device: target模型所在的设备

    返回:
        ReturnStruct对象 (所有张量都在proxy_device上)，其中loss为对抗奖励loss。
    """
    # 默认损失参数
    if loss_params is None:
        loss_params = {
            'hard_labels': True,
            'reweight_loss': False,
            'reduction': 'none'
        }

    # 确保target模型在目标设备上
    target_model = target_model.to(target_device)

    # 1. 在proxy设备上生成扰动后缀文本
    with torch.device(proxy_device):
        pred_ids = torch.argmax(y_logits_, dim=-1)
        # 解码扰动后缀
        suffix_texts = [proxy_tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]

    # 2. 构造完整提示：基础提示 + 扰动后缀
    # 如果 baseprompt 为字符串列表，确保和batch_size匹配；如果为单一字符串，则对每个样本重复
    if isinstance(baseprompt, str):
        full_prompt_texts = [baseprompt + suffix for suffix in suffix_texts]
        baseprompts = [baseprompt] * len(suffix_texts)
    else:
        full_prompt_texts = [bp + suffix for bp, suffix in zip(baseprompt, suffix_texts)]
        baseprompts = baseprompt

    # 3. 在target设备上编码完整提示（扰动后的输入）
    with torch.device(target_device):
        full_inputs = target_tokenizer(
            full_prompt_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=target_model.config.max_position_embeddings
        )
        full_inputs = {k: v.to(target_device) for k, v in full_inputs.items()}

    # 4. 在target设备上计算扰动提示的输出（教师强制生成），得到扰动loss
    with torch.device(target_device), torch.no_grad():
        full_outputs = target_model(
            input_ids=full_inputs['input_ids'],
            attention_mask=full_inputs['attention_mask']
        )
        full_logits = full_outputs.logits

    # 5. 构造扰动提示的预测与目标序列
    # 预测序列：去掉最后一个token
    disturbed_pred_logits = full_logits[:, :-1, :]
    disturbed_mask = full_inputs['attention_mask'][:, :-1]
    # 目标序列：去掉第一个token
    disturbed_target_ids = full_inputs['input_ids'][:, 1:]
    disturbed_target_mask = full_inputs['attention_mask'][:, 1:]

    # 6. 计算扰动提示的 loss（在target_device上）
    with torch.device(target_device):
        if loss_params['hard_labels']:
            disturbed_loss_tensor = F.cross_entropy(
                disturbed_pred_logits.transpose(1, 2),
                disturbed_target_ids,
                reduction='none'
            )
        else:
            full_probs = F.softmax(full_logits[:, :-1, :], dim=-1)
            disturbed_loss_tensor = F.cross_entropy(
                disturbed_pred_logits.transpose(1, 2),
                full_probs.transpose(1, 2),
                reduction='none'
            )
        if loss_params.get('reweight_loss', False):
            seq_len = disturbed_loss_tensor.shape[1]
            factor = torch.arange(seq_len, dtype=disturbed_loss_tensor.dtype, device=target_device) + 1
            disturbed_loss_tensor = disturbed_loss_tensor / factor[None, :]

        disturbed_loss_masked = disturbed_loss_tensor * disturbed_target_mask
        disturbed_loss_batch = torch.sum(disturbed_loss_masked, dim=1) / (disturbed_target_mask.sum(dim=1) + 1e-10)
        disturbed_loss = disturbed_loss_batch.mean()

    # 7. 计算基线提示的 loss：仅使用基础提示
    with torch.device(target_device):
        baseline_inputs = target_tokenizer(
            baseprompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=target_model.config.max_position_embeddings
        )
        baseline_inputs = {k: v.to(target_device) for k, v in baseline_inputs.items()}
        baseline_outputs = target_model(
            input_ids=baseline_inputs['input_ids'],
            attention_mask=baseline_inputs['attention_mask']
        )
        baseline_logits = baseline_outputs.logits

    # 构造基线预测与目标序列
    baseline_pred_logits = baseline_logits[:, :-1, :]
    baseline_mask = baseline_inputs['attention_mask'][:, :-1]
    baseline_target_ids = baseline_inputs['input_ids'][:, 1:]
    baseline_target_mask = baseline_inputs['attention_mask'][:, 1:]
    with torch.device(target_device):
        if loss_params['hard_labels']:
            baseline_loss_tensor = F.cross_entropy(
                baseline_pred_logits.transpose(1, 2),
                baseline_target_ids,
                reduction='none'
            )
        else:
            baseline_probs = F.softmax(baseline_logits[:, :-1, :], dim=-1)
            baseline_loss_tensor = F.cross_entropy(
                baseline_pred_logits.transpose(1, 2),
                baseline_probs.transpose(1, 2),
                reduction='none'
            )
        if loss_params.get('reweight_loss', False):
            seq_len = baseline_loss_tensor.shape[1]
            factor = torch.arange(seq_len, dtype=baseline_loss_tensor.dtype, device=target_device) + 1
            baseline_loss_tensor = baseline_loss_tensor / factor[None, :]

        baseline_loss_masked = baseline_loss_tensor * baseline_target_mask
        baseline_loss_batch = torch.sum(baseline_loss_masked, dim=1) / (baseline_target_mask.sum(dim=1) + 1e-10)
        baseline_loss = baseline_loss_batch.mean()

    # 8. 计算对抗奖励loss：这里我们希望扰动使得目标模型的输出与基线有更大偏离，
    # 可以使用 disturbed_loss - baseline_loss 作为对抗信号（也可以根据需求取负号）
    adv_loss = disturbed_loss - baseline_loss

    # 9. 将所有结果移回proxy设备
    with torch.device(proxy_device):
        # 这里返回的loss即对抗loss，其他信息也可按需返回
        return ReturnStruct(
            loss=adv_loss.to(proxy_device),
            disturbed_loss=disturbed_loss.to(proxy_device),
            baseline_loss=baseline_loss.to(proxy_device),
            pred=disturbed_pred_logits.to(proxy_device),
            label=disturbed_target_ids.to(proxy_device),
            mask=disturbed_target_mask.to(proxy_device)
        )


def compute_adv_loss(
        y_logits_,
        proxy_tokenizer,
        target_model,
        target_tokenizer,
        loss_params=None,
        proxy_device="cuda",
        target_device="cuda"
):
    """
    支持proxy和target在不同设备的对抗损失计算

    参数:
        y_logits_: 扰动生成的logits [batch_size, seq_len, proxy_vocab_size] (在proxy_device上)
        proxy_tokenizer: 代理模型的tokenizer
        target_model: 目标模型 (将自动移动到target_device)
        target_tokenizer: 目标tokenizer
        loss_params: 损失参数配置字典
        proxy_device: proxy模型所在的设备
        target_device: target模型所在的设备

    返回:
        ReturnStruct对象 (所有张量都在proxy_device上)
    """
    # 默认损失参数
    if loss_params is None:
        loss_params = {
            'hard_labels': True,
            'reweight_loss': False,
            'reduction': 'none'
        }

    # 确保target模型在目标设备上
    target_model = target_model.to(target_device)

    # 1. 在proxy设备上生成提示词序列
    with torch.device(proxy_device):
        pred_ids = torch.argmax(y_logits_, dim=-1)
        prompt_texts = [proxy_tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]

    # 2. 在target设备上编码输入
    with torch.device(target_device):
        target_inputs = target_tokenizer(
            prompt_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=target_model.config.max_position_embeddings
        )
        # 将输入数据移到target设备
        target_inputs = {k: v.to(target_device) for k, v in target_inputs.items()}

    # 3. 在target设备上教师强制生成目标输出
    with torch.device(target_device), torch.no_grad():
        target_outputs = target_model(
            input_ids=target_inputs['input_ids'],
            attention_mask=target_inputs['attention_mask']
        )
        target_logits = target_outputs.logits

    # 4. 准备序列数据
    # 预测序列 (去掉最后一个token)
    pred_logits = target_logits[:, :-1, :]
    pred_mask = target_inputs['attention_mask'][:, :-1]

    # 目标序列 (去掉第一个token)
    target_ids = target_inputs['input_ids'][:, 1:]
    target_mask = target_inputs['attention_mask'][:, 1:]

    # 5. 计算损失 (在target设备上)
    with torch.device(target_device):
        if loss_params['hard_labels']:
            _loss = F.cross_entropy(
                pred_logits.transpose(1, 2),
                target_ids,
                reduction='none'
            )
        else:
            target_probs = F.softmax(target_logits[:, :-1, :], dim=-1)
            _loss = F.cross_entropy(
                pred_logits.transpose(1, 2),
                target_probs.transpose(1, 2),
                reduction='none'
            )

        if loss_params.get('reweight_loss', False):
            seq_len = _loss.shape[1]
            factor = torch.arange(seq_len, dtype=_loss.dtype, device=target_device) + 1
            _loss = _loss / factor[None, :]

        loss_masked = _loss * target_mask
        loss_batch = torch.sum(loss_masked, dim=1) / (target_mask.sum(dim=1) + 1e-10)
        loss = loss_batch.mean()

    # 6. 将所有结果移回proxy设备
    with torch.device(proxy_device):
        return ReturnStruct(
            loss=loss.to(proxy_device),
            loss_masked=loss_masked.to(proxy_device),
            loss_batch=loss_batch.to(proxy_device),
            pred=pred_logits.to(proxy_device),
            label=target_ids.to(proxy_device),
            mask=target_mask.to(proxy_device)
        )

def word_tokenize(text: str):
    """简单实现按单词切分文本（可根据需要替换为更强的分词器）"""
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


# 在RL梯度应用前插入投影操作
def project_gradient(rl_grad, original_grad):
    """改进的梯度投影方法"""
    # 计算主梯度方向的单位向量
    original_norm = original_grad.norm() + 1e-8
    original_unit = original_grad / original_norm

    # 计算平行分量
    parallel_mag = torch.sum(rl_grad * original_unit)
    parallel = parallel_mag * original_unit

    # 正交分量 = 原始梯度 - 平行分量
    orthogonal = rl_grad - parallel

    # 动态混合比例：当主梯度较大时减少正交分量影响
    mix_ratio = torch.sigmoid(-original_norm / 5.0)  # 自适应比例
    return orthogonal * mix_ratio + parallel * (1 - mix_ratio)


def compute_wed(seq1: list, seq2: list) -> int:
    """计算两个单词序列的编辑距离（Levenshtein距离）"""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[n][m]


def compute_swes(u_text: str, d_text: str) -> float:
    """计算SWES相似度，使用滑动窗口机制返回 -log(min_edit_distance)"""
    u_tokens = word_tokenize(u_text)
    d_tokens = word_tokenize(d_text)
    if len(u_tokens) < len(d_tokens):
        min_dist = compute_wed(u_tokens, d_tokens)
    else:
        d_len = len(d_tokens)
        min_dist = math.inf
        for i in range(len(u_tokens) - d_len + 1):
            sub_u = u_tokens[i:i + d_len]
            dist = compute_wed(sub_u, d_tokens)
            if dist < min_dist:
                min_dist = dist
    if min_dist == 0:
        swes_val = 1e6  # 完全匹配时取一个极大值
    else:
        swes_val = -math.log(min_dist)
    return swes_val


def compute_reward(u_text: str, d_text: str, lambda_val: float = 0.1, k: float = 5, x0: float = 0.6) -> float:
    """
    根据 PrivAgent 方法计算最终奖励:
      R(u, d) = (1-λ)*SWES_norm(u,d) + λ*(1 / abs(|u| - |d|))
    其中 SWES_norm 为经过 Sigmoid 归一化的 SWES 值。
    """
    swes_raw = compute_swes(u_text, d_text)
    if swes_raw == 1e6:
        swes_norm = 1.0
    else:
        swes_norm = 1 / (1 + math.exp(-k * (swes_raw - x0)))
    u_len = len(word_tokenize(u_text))
    d_len = len(word_tokenize(d_text))
    length_diff = abs(u_len - d_len)
    length_reg = 1.0 if length_diff == 0 else 1.0 / length_diff
    reward = (1 - lambda_val) * swes_norm + lambda_val * length_reg
    return reward


def dynamic_temperature(step: int, k_threshold: int, T_high: float, T_base: float) -> float:
    """
    根据生成步骤动态调整温度：
      若 step <= k_threshold：返回 T_high；
      否则返回 T_base。
    """
    return T_high if step <= k_threshold else T_base


def get_rl_conf_sigmoid(success_rate, min_conf=0.05, max_conf=0.2, center=0.5, scale=10):
    # Sigmoid 映射：center 控制中间拐点，scale 控制曲线陡峭程度
    # 当 success_rate 接近 center 时，函数值迅速下降
    sigmoid_value = 1 / (1 + math.exp(scale * (success_rate - center)))
    # 将 sigmoid 值映射到 [min_conf, max_conf]
    return min_conf + (max_conf - min_conf) * sigmoid_value


# -----------------------------------------
# 2. 修改 fast_sample_generate 函数，融入动态温度调整
# -----------------------------------------
def fast_sample_generate(prompt_ids, model, max_length, top_p=0.9, k_threshold=5, T_high=2.0, T_base=1.0,
                         target_length=None):
    """
    利用核采样生成文本，并在每一步根据当前位置动态调整温度。
    参数:
      prompt_ids: 初始输入的 token ids，形状 [batch, prompt_len]
      model: 生成模型
      max_length: 最大生成长度
      top_p: nucleus sampling 的阈值
      k_threshold: 前 k 个 token 使用高温度
      T_high: 高温度值
      T_base: 基础温度值
      target_length: 如果设置，则当生成长度达到 target_length 时提前停止
    返回:
      generated: 生成的 token ids 张量
    """
    generated = prompt_ids.clone()
    for step in range(max_length):
        outputs = model(input_ids=generated)
        logits = outputs.logits[:, -1, :]  # 获取当前步 logits

        # 动态温度调整
        current_temp = dynamic_temperature(step + 1, k_threshold, T_high, T_base)
        logits = logits / current_temp

        # nucleus采样
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')

        # 采样下一个token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)

        if target_length is not None and generated.size(1) >= target_length:
            break
    return generated

# -------------------------------
# 定义 Gumbel-Softmax 函数（支持 STE）
# -------------------------------
def gumbel_softmax_sample(logits, temperature=0.7, hard=True, eps=1e-10):
    U = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    y = logits + gumbel_noise
    y_soft = F.softmax(y / temperature, dim=-1)
    if hard:
        # 得到 one-hot 向量（但梯度通过 y_soft 传递）
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        # STE：前向使用 y_hard，反向梯度用 y_soft
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft


# -----------------------------------------
# 3. RL 训练过程中如何使用
# -----------------------------------------
# 以下变量已在 RL 代码中定义：
#   - model: 用于生成文本的模型（攻击目标）
#   - tokenizer: 对应的tokenizer，用于将token id转为文本
#   - device: torch.device
#   - x_t: 经过处理的危险提示词
#   - y_logits, epsilon: 用于生成软提示词的张量（可微分的部分） 和 扰动
#   - args: 包含 batch_size, gen_max_length 等参数
#   - target_text: 想要得到的目标敏感信息文本（d）
#
# 以下代码展示在 RL 循环中生成文本、计算奖励，并用于梯度反向传播更新的过程。

#
def rl_update_step(x_t, y_logits, epsilon, model, tokenizer, args, target_text, optim, scheduler,
                   device, grad_main, ite, accumulation_steps):
    # 1. 使用 gumbel_softmax_sample 得到软提示词（保持可微分）
    epsilon_old = epsilon.detach().clone()
    learned_prompt_soft = gumbel_softmax_sample(y_logits + epsilon_old, temperature=0.7, hard=True)
    # 2. 拼接左侧上下文 x_onehot 与学习提示词，得到完整软提示词
    full_prompt_soft = torch.cat([x_t, learned_prompt_soft], dim=1)
    # 3. 将软提示词转为离散 token，供生成使用
    learned_prompt_ids = torch.argmax(learned_prompt_soft, dim=-1)
    full_prompt_ids = torch.cat([x_t, learned_prompt_ids], dim=1)
    gen_max_length = args.gen_max_length if hasattr(args, 'gen_max_length') else (full_prompt_ids.shape[1] + 20) // 2

    # 4. 使用修改后的 fast_sample_generate 生成目标文本（调用动态温度策略）
    generated_ids = fast_sample_generate(full_prompt_ids, model, max_length=gen_max_length, top_p=0.9,
                                         k_threshold=5, T_high=2.0, T_base=1.0, target_length=args.gen_max_length)
    # 解码生成文本
    generated_texts = [tokenizer.decode(generated_ids[i].tolist(), skip_special_tokens=True) for i in
                       range(args.batch_size)]

    # 5. 计算软提示词的 log 概率（作为 RL 信号）
    prompt_log_probs = torch.sum(torch.log(learned_prompt_soft + 1e-10), dim=[1, 2])

    # 6. 根据生成的文本和目标文本计算 RL 奖励（：密集奖励函数）
    rl_rewards = []
    for text in generated_texts:
        reward = compute_reward(text, target_text, lambda_val=0.1, k=5, x0=0.6)
        rl_rewards.append(reward)
    rl_rewards_tensor = torch.tensor(rl_rewards, dtype=torch.float, device=device)

    # 7. 使用贪婪生成作为 baseline（同理，计算其奖励）
    greedy_generated_ids = fast_sample_generate(full_prompt_ids, model, max_length=gen_max_length, top_p=0.9,
                                                k_threshold=5, T_high=2.0, T_base=1.0,
                                                target_length=args.gen_max_length)
    greedy_texts = [tokenizer.decode(greedy_generated_ids[i].tolist(), skip_special_tokens=True) for i in
                    range(args.batch_size)]
    baseline_rewards = []
    for text in greedy_texts:
        base_reward = compute_reward(text, target_text, lambda_val=0.1, k=5, x0=0.6)
        baseline_rewards.append(base_reward)
    baseline_rewards_tensor = torch.tensor(baseline_rewards, dtype=torch.float, device=device)

    # 8. 计算优势 (Advantage)
    advantage = rl_rewards_tensor - baseline_rewards_tensor
    adv_mean = advantage.mean()
    adv_std = advantage.std() + 1e-8
    advantage = (advantage - adv_mean) / adv_std
    advantage = advantage.clamp(min=-2.0, max=2.0)

    # 根据是否成功（例如由query_ollama返回）给出附加权重（此处可根据实际需要调整）
    # 此处假设 success_flags_tensor 已经计算得到，此处简化处理
    success_flags_tensor = torch.ones_like(advantage)  # 示例中全部设为1
    alpha_rl = 1.0
    sample_weights = 1.0 + alpha_rl * success_flags_tensor
    weighted_advantage = advantage * sample_weights

    # 9. 计算 RL 损失：PPO框架下通常为 -advantage * log_prob
    rl_loss = - (weighted_advantage * prompt_log_probs).mean()

    # 可选：根据当前成功率进行动态缩放（例如 get_rl_conf_sigmoid 函数）
    current_success_rate = success_flags_tensor.mean().item()
    rl_conf = get_rl_conf_sigmoid(current_success_rate, min_conf=0.05, max_conf=0.2, center=0.5, scale=10)
    rl_loss_scaled = rl_conf * rl_loss

    # 10. 反向传播 RL 损失，并与原梯度融合（例如利用投影方法）
    optim.zero_grad()
    rl_loss_scaled.backward(retain_graph=True)
    rl_grad = epsilon.grad - grad_main  # grad_main 为主任务梯度
    projected_rl_grad = project_gradient(rl_grad, grad_main)
    epsilon.grad = grad_main + projected_rl_grad * 0.5  # 融合梯度权重0.5
    torch.nn.utils.clip_grad_norm_([epsilon], max_norm=1.0)

    # 更新参数
    if (ite + 1) % accumulation_steps == 0:
        optim.step()
        scheduler.step()



    return rl_loss_scaled, generated_texts



# -----------------------------------------
# 使用示例说明
# -----------------------------------------
# 假设在您的 RL 训练循环中，每隔一定步数进行一次 RL 更新：
#
# for ite in range(total_iterations):
#     # ...生成主任务梯度 grad_main 等
#     if ite >= 1000 and ite % rl_eval_interval == 0:
#         rl_loss_scaled, gen_texts = rl_update_step(x_t, y_logits, epsilon, model, tokenizer, args, target_text,
#                                                     optim, scheduler, success_memory, device, grad_main, ite, rl_eval_interval, accumulation_steps)
#         print(f"Iteration {ite}: RL loss {rl_loss_scaled.item()}, Generated texts: {gen_texts}")
#
# 在上述示例中：
# - fast_sample_generate 调用了 dynamic_temperature，根据当前token步数选择 T_high 或 T_base。
# - compute_reward 根据生成文本和目标文本计算密集奖励。
# - RL 损失（使用优势加权的对数概率损失）反向传播后，与主任务梯度融合进行更新。
#
# 您可以将该代码片段整合到您的训练循环中，实现完全按照 PrivAgent 论文的方法更新RL代理模型。
