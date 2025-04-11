import pandas as pd
from opt_util import load_model_and_tokenizer
import torch
import torch.nn as nn
import logging

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 读取 CSV 数据并获取 prompt 列表
data = pd.read_csv("./outputs/Llama-2-7b-chat-hf/output_hf-v1/0_50_proxy_8_2000_100.0_100.0_500.0_100.0.csv")
prompt_with_adv = data['prompt_with_adv'].tolist()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和分词器
model, tokenizer = load_model_and_tokenizer("D:\\ZLCODE\\model\\Llama-2-7b-chat-hf",
                                            low_cpu_mem_usage=True,
                                            use_cache=False,
                                            device=device)

print(model)          # 观察整体结构
print(model.model)    # 查看主模型模块
print(model.model.layers)  # 确认层列表是否存在

# 设置模型为评估模式
model.eval()

# 定义前向钩子函数，用于禁用目标神经元（假定目标神经元在第一层，索引为 2533）
target_neuron_index = 2533
def disable_neuron_hook(module, input, output):
    # 假设输出形状为 [batch_size, seq_length, hidden_size]
    if isinstance(output, torch.Tensor):
        output[..., target_neuron_index] = 0
    return output

decoded_texts = []  # 用于存储每个 prompt 的正常和干预后的生成结果

# 对每个 prompt 生成文本
for bi in range(len(prompt_with_adv)):
    print("\n=== 开始生成过程 ===")
    print(f"当前处理第 {bi + 1} 个 prompt")
    try:
        prompt = prompt_with_adv[bi]
        print(f"原始 prompt 内容 (前100字符): {prompt[:100]}...")

        if not prompt or prompt.isspace():
            print("警告: 检测到空 prompt, 跳过生成")
            decoded_texts.append({"normal": "", "disabled": ""})
            continue

        # 移除特殊 token 并清理 prompt
        prompt = prompt.replace("</s>", " ").strip()

        # 对 prompt 进行 tokenization
        inputs = tokenizer(prompt,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=512,
                           return_attention_mask=True)

        # 将输入移动到设备上
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        print(f"Tokenization 后 input_ids 形状: {input_ids.shape}")
        print(f"前10个 token: {input_ids[0, :10].tolist()}")

        if input_ids.numel() == 0 or torch.all(input_ids == 0):
            print("警告: 检测到无效的 input_ids, 跳过生成")
            decoded_texts.append({"normal": "", "disabled": ""})
            continue

        # 生成正常文本
        try:
            output_ids_normal = model.generate(
                input_ids=input_ids,
                temperature=0.7,
                max_length=512,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_k=10
            )
            # 截取生成部分（去除原 prompt 部分）
            output_ids_normal = output_ids_normal[:, input_ids.shape[1]:]
            text_normal = tokenizer.decode(output_ids_normal[0], skip_special_tokens=True).strip()
            print(f"正常生成文本, 长度: {len(text_normal)}")
            print(f"正常生成文本: {text_normal}")
        except RuntimeError as e:
            print(f"生成正常文本过程中的 CUDA 错误: {str(e)}")
            text_normal = ""
        except Exception as e:
            print(f"生成正常文本过程中的其他错误: {str(e)}")
            text_normal = ""

        # 生成禁用目标神经元后的文本
        # 假设目标层为 model.model.layers[0]
        hook_handle = model.model.layers[0].register_forward_hook(disable_neuron_hook)
        try:
            output_ids_disabled = model.generate(
                input_ids=input_ids,
                temperature=0.7,
                max_length=512,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_k=10
            )
            # 截取生成部分（去除原 prompt 部分）
            output_ids_disabled = output_ids_disabled[:, input_ids.shape[1]:]
            text_disabled = tokenizer.decode(output_ids_disabled[0], skip_special_tokens=True).strip()
            print(f"禁用神经元后生成文本, 长度: {len(text_disabled)}")
            print(f"禁用神经元后生成文本: {text_disabled}")
        except RuntimeError as e:
            print(f"生成禁用神经元文本过程中的 CUDA 错误: {str(e)}")
            text_disabled = ""
        except Exception as e:
            print(f"生成禁用神经元文本过程中的其他错误: {str(e)}")
            text_disabled = ""
        hook_handle.remove()

        decoded_texts.append({"normal": text_normal, "disabled": text_disabled})
    except Exception as e:
        print(f"处理 prompt 时发生错误: {str(e)}")
        decoded_texts.append({"normal": "", "disabled": ""})
        continue

# 将所有生成的文本合并，并同时输出到控制台和写入文件
all_outputs = ""
for idx, texts in enumerate(decoded_texts):
    all_outputs += f"=== Prompt {idx+1} ===\n"
    all_outputs += "【正常生成】:\n" + texts["normal"] + "\n"
    all_outputs += "【禁用神经元后生成】:\n" + texts["disabled"] + "\n"
    all_outputs += "\n=== 分割线 ===\n\n"

print("\n=== 生成过程完成 ===")
print("全部生成的文本输出如下：\n")
print(all_outputs)

output_file = "all_generated_outputs.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_outputs)
print(f"\n全部生成的文本已写入文件: {output_file}")
