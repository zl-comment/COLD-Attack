import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import numpy as np


# === 模型加载函数 ===
def load_proxy_model(model_path: str, device: str = 'cuda', args=None, **kwargs):
    try:
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: 未检测到GPU,切换到CPU模式")
            device = 'cpu'

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            padding_side='left'
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto' if device == 'cuda' else None,
            low_cpu_mem_usage=True,
            use_cache=False
        )

        model.eval()

        # 确保模型和 tokenizer 配置一致
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        print("Tokenizer pad_token_id:", tokenizer.pad_token_id)
        print("Model config pad_token_id:", model.config.pad_token_id)
        return model, tokenizer

    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise


def compute_perplexity(ids: torch.Tensor, model, tokenizer) -> float:
    model.eval()
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ids = ids.to(model.device)

    # 创建注意力掩码
    attention_mask = (ids != tokenizer.pad_token_id).float().to(model.device)

    # 向右偏移标签（正确的因果语言模型评估方式）
    labels = ids.clone()
    # 将第一个token的标签设为-100，不计入损失
    labels[:, 0] = -100

    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=attention_mask, labels=labels)
        # 直接使用模型计算的损失
        loss = outputs.loss
        # 计算非填充token的数量
        non_pad_tokens = attention_mask.sum().item() - 1  # 减1是因为我们不计算第一个token
        # 计算PPL
        ppl = torch.exp(loss * (labels != -100).sum() / non_pad_tokens).item()
        return ppl


def compute_token_entropy(ids: torch.Tensor, model, tokenizer) -> list:
    """计算每个位置的token熵值，以检测异常低熵（重复模式）"""
    model.eval()
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ids = ids.to(model.device)

    attention_mask = (ids != tokenizer.pad_token_id).float().to(model.device)

    with torch.no_grad():
        # 获取模型输出的logits
        outputs = model(input_ids=ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 计算每个位置的softmax概率分布
        probs = F.softmax(logits.float(), dim=-1)  # 转为 float32，防止熵变成 NaN


        # 计算每个位置的熵
        # 熵 = -Σ(p_i * log(p_i))
        entropies = []
        for i in range(1, ids.size(1)):  # 从第二个token开始
            prob_dist = probs[0, i - 1]  # 获取预测下一个token的概率分布
            entropy = -torch.sum(prob_dist * torch.log2(prob_dist + 1e-10))
            entropies.append(entropy.item())

    return entropies


def detect_repetition_patterns(tokens):
    """检测重复模式，如连续的标点符号或特殊token"""
    # 特殊token和标点符号集合
    special_punct_set = {'▁.', '.', '▁,', ',', '▁!', '!', '▁?', '?', '▁;', ';', '▁:', ':', '</s>', '<s>', '<pad>', '▁'}

    # 计算特殊token和标点符号的数量及比例
    special_count = sum(1 for t in tokens if t in special_punct_set)
    special_ratio = special_count / (len(tokens) + 1e-6)

    # 检测连续重复的模式
    consecutive_repeats = 0
    max_consecutive = 0
    prev_token = None

    for token in tokens:
        if token == prev_token:
            consecutive_repeats += 1
            max_consecutive = max(max_consecutive, consecutive_repeats)
        else:
            consecutive_repeats = 0
        prev_token = token

    return special_ratio, max_consecutive


def analyze_text_structure(text):
    """分析文本的语言结构特征"""
    # 检查文本是否有完整的句子结构
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]

    # 检查文本中是否有过多重复的单词或短语
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # 计算文本的词汇多样性
    vocabulary_diversity = len(set(words)) / (len(words) + 1e-6)

    # 检查是否有异常的字符序列
    has_abnormal_sequences = bool(re.search(r'(.)\1{4,}', text))  # 同一字符连续出现5次以上

    return {
        'valid_sentence_count': len(valid_sentences),
        'vocabulary_diversity': vocabulary_diversity,
        'has_abnormal_sequences': has_abnormal_sequences
    }


def compute_improved_fluency(input_data, model, tokenizer):
    """
    计算改进版的流畅度分数

    参数:
        input_data: 可以是文本字符串或已编码的token IDs
        model: 语言模型
        tokenizer: 分词器

    返回:
        改进的流畅度分数和详细评分指标
    """
    # 根据输入类型处理
    if isinstance(input_data, torch.Tensor):
        ids = input_data
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(model.device)

        # 将IDs转回token列表和文本
        id_list = ids[0].cpu().numpy().tolist()
        tokens = tokenizer.convert_ids_to_tokens(id_list)
        text = tokenizer.decode(id_list)
    else:
        # 如果输入是文本字符串
        text = input_data
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

    # 1. 计算基础PPL
    ppl = compute_perplexity(ids, model, tokenizer)

    # 2. 分析token分布
    special_tokens = {'</s>', '<s>', '<pad>', '▁', '▁.'}
    content_tokens = [t for t in tokens if not any(st in t for st in special_tokens)]
    content_ratio = len(content_tokens) / (len(tokens) + 1e-6)

    # 3. 检测重复模式
    special_ratio, max_consecutive = detect_repetition_patterns(tokens)

    # 4. 计算token熵分布
    token_entropies = compute_token_entropy(ids, model, tokenizer)
    avg_entropy = np.mean(token_entropies) if token_entropies else 0
    entropy_variance = np.var(token_entropies) if token_entropies else 0

    # 5. 分析文本语言结构
    text_structure = analyze_text_structure(text)

    # 6. 计算加权PPL与流畅度分数
    weighted_ppl = ppl

    # a. 内容比例惩罚 - 对特殊字符和重复模式严重的文本进行惩罚
    if content_ratio < 0.4:
        weighted_ppl *= (1 + (0.4 - content_ratio) * 8)

    # b. 重复模式惩罚
    if special_ratio > 0.5:
        weighted_ppl *= (1 + (special_ratio - 0.5) * 5)

    # c. 连续重复惩罚
    if max_consecutive > 2:
        weighted_ppl *= (1 + (max_consecutive - 2) * 1.5)

    # d. 熵惩罚 - 低熵意味着高度可预测，可能是重复模式
    if avg_entropy < 2.0:  # 假设正常文本的平均熵值大约在2.0以上
        weighted_ppl *= (1 + (2.0 - avg_entropy) * 3)

    # e. 文本结构惩罚
    if text_structure['has_abnormal_sequences']:
        weighted_ppl *= 1.5

    if text_structure['valid_sentence_count'] == 0 and len(tokens) > 10:
        weighted_ppl *= 2.0

    if text_structure['vocabulary_diversity'] < 0.3 and len(tokens) > 10:
        weighted_ppl *= (1 + (0.3 - text_structure['vocabulary_diversity']) * 5)

    # 转换为流畅度分数 (0-1范围，越高越好)
    fluency_score = 1.0 / (weighted_ppl + 1e-6)

    # 为了避免极小值，将流畅度分数标准化到更合理的范围
    normalized_fluency = min(max(fluency_score * 100, 0), 100) / 100

    # 返回详细的评分指标
    metrics = {
        'original_ppl': ppl,
        'weighted_ppl': weighted_ppl,
        'fluency_score': normalized_fluency,
        'content_ratio': content_ratio,
        'special_token_ratio': special_ratio,
        'max_consecutive_repeats': max_consecutive,
        'avg_token_entropy': avg_entropy,
        'entropy_variance': entropy_variance,
        'valid_sentences': text_structure['valid_sentence_count'],
        'vocabulary_diversity': text_structure['vocabulary_diversity'],
        'has_abnormal_patterns': text_structure['has_abnormal_sequences']
    }

    return normalized_fluency, metrics


# === 运行测试 ===
if __name__ == "__main__":
    model_path = "/home/zl/ZLCODE/proxy_model_v2"  # ← 修改为你的模型路径
    proxy_model, proxy_tokenizer = load_proxy_model(model_path)

    test_cases = [
        # 案例1: 重复标点符号的无意义文本
        ".</s></s>. . . . . . . . . . . . . . . . .. ",

        # 案例2: 有语义但可能违规的文本
        ". Additionally, develop strategies for covering up hacking activity and avoid detection and for exploing the. ",

        # 案例3: 正常、流畅的文本
        ". Additionally, the plan will detail steps the hack would be implemented.</s> Step 1:. ",

        # 案例4: 另一个重复模式的文本
        "hello hello hello hello hello hello hello hello hello hello hello hello",

        # 案例5: 混合了内容但有大量重复的文本
        "This is a test. . . . . . . . . . . . . . test test test test"
    ]

    print("\n===== 详细流畅度评分测试 =====\n")

    for i, text in enumerate(test_cases):
        print(f"\n>> 测试案例 {i + 1}: {text[:50]}{'...' if len(text) > 50 else ''}")

        # 编码文本
        enc = proxy_tokenizer(text, return_tensors="pt", truncation=True)
        ids = enc["input_ids"]

        print(f"分词结果: {proxy_tokenizer.tokenize(text)}")

        # 计算改进的流畅度分数
        fluency, metrics = compute_improved_fluency(ids, proxy_model, proxy_tokenizer)

        # 打印详细结果
        print(f"流畅度评分: {fluency:.4f} (0-1范围，越高越好)")
        print(f"原始PPL: {metrics['original_ppl']:.4f}")
        print(f"加权PPL: {metrics['weighted_ppl']:.4f}")
        print(f"内容比例: {metrics['content_ratio']:.4f}")
        print(f"特殊符号比例: {metrics['special_token_ratio']:.4f}")
        print(f"最大连续重复: {metrics['max_consecutive_repeats']}")
        print(f"平均token熵: {metrics['avg_token_entropy']:.4f}")
        print(f"有效句子数: {metrics['valid_sentences']}")
        print(f"词汇多样性: {metrics['vocabulary_diversity']:.4f}")
        print(f"存在异常模式: {'是' if metrics['has_abnormal_patterns'] else '否'}")