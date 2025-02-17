
import pandas as pd
from opt_util import load_model_and_tokenizer
import torch


data = pd.read_csv("./outputs/Llama-2-7b-chat-hf/final_model-10/0_50.csv")
prompt_with_adv = data['prompt_with_adv'].tolist()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, tokenizer = load_model_and_tokenizer("D:\ZLCODE\model\Llama-2-7b-chat-hf",
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device=device)
#目标模型评估模式
model.eval()


decoded_text = []
# 对每个batch生成完整文本
for bi in range(len(prompt_with_adv)):
        print("\n=== 开始生成过程 ===")
        print(f"批次大小: {bi}")



        # 构建并验证prompt
        try:
            prompt =prompt_with_adv[bi]
            print(f"原始prompt内容: {prompt[:100]}...")  # 只打印前100个字符

            # 验证prompt
            if not prompt or prompt.isspace():
                print(f"警告: 检测到空prompt,跳过生成")
                decoded_text.append("")
                continue

            # 清理特殊token
            prompt = prompt.replace("</s>", " ").strip()  # 移除特殊token

            input = tokenizer(prompt,
                          return_tensors="pt",
                          padding=True,
                          truncation=True,
                          max_length=512,
                          return_attention_mask=True)

            # 移动到设备并验证
            input_ids = input["input_ids"].to(device)
            attention_mask = input["attention_mask"].to(device)

            print(f"tokenization后的input_ids形状: {input_ids.shape}")
            print(f"前10个token: {input_ids[0 ,:10].tolist()}")

            # 验证input_ids
            if input_ids.numel() == 0 or torch.all(input_ids == 0):
                print(f"警告: 检测到无效的input_ids,跳过生成")
                decoded_text.append("")
                continue

            # 生成文本
            try:
                output_ids = model.generate(
                input_ids=input_ids,
                temperature=0.7,
                max_length=512,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_k=10
                )

                output_ids = output_ids[:, input_ids.shape[1]:]
                text_dec = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                decoded_text.append(text_dec.strip())
                print(f"成功生成文本,长度: {len(text_dec)}")
                print(f"成功生成文本：{decoded_text[bi]}")

            except RuntimeError as e:
                print(f"生成过程中的CUDA错误: {str(e)}")
                decoded_text.append("")
            except Exception as e:
                print(f"生成过程中的其他错误: {str(e)}")
                decoded_text.append("")

        except Exception as e:
            print(f"处理prompt时发生错误: {str(e)}")
            decoded_text.append("")
            continue

print("\n=== 生成过程完成 ===")
print(f"成功生成的文本数量: {len([t for t in decoded_text if t])}/{8}")



