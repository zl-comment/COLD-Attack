import pandas as pd
import os
from collections import defaultdict
import re

import torch

from opt_util import load_model_and_tokenizer
from util import get_gpt_ppl


def clean_answer(o):
    """
    清理文本，移除特殊字符和多余空格
    """
    o = o.replace("\n\n", "\n").strip()
    o = ' '.join(o.split())
    return o



def post_process(string):
    string = string.strip(" ")
    string = string.strip(".")
    string = string.strip("!")
    string = string.strip("?")
    string = string.strip(",")
    string = string.strip("(")
    string = string.strip(")")
    string = string.strip("```")
    string = string.strip("\"")
    return string

def remove_special_characters(input_string):
    output_string = re.sub(r'\d+\.(?:\d+\.)*\s*', '', input_string)
    output_string = re.sub(r'[^a-zA-Z0-9\s]', '', output_string)
    output_string = ' '.join(output_string.split())
    return output_string




def process_file(csv_file,gpt_model,gpt_tokenizer):
    """
    处理单个 CSV 文件，并返回处理后的 DataFrame 与统计信息
    """
    print(f"读取文件：{csv_file}")
    results = pd.read_csv(csv_file)

    total_responses = 0
    total_success_responses = 0
    total_success_responses_D = 0

    ref = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])

    # 获取所有唯一的 goal（原始提示）
    unique_goals = results['goal'].unique()
    ppls_prompt_adv = []

    for i, goal in enumerate(unique_goals):
        sub_results = results[results['goal'] == goal]

        for index, row in sub_results.iterrows():
            prompt = row['goal']
            suffix = row['suffix']
            prompt_with_adv = str(prompt) + str(suffix)


            if type(prompt_with_adv) != str:
                continue

            prompt_with_adv = clean_answer(str(prompt_with_adv))
            prompt_with_adv = post_process(prompt_with_adv)
            prompt_with_adv = prompt_with_adv.strip(" ")
            prompt_with_adv = remove_special_characters(prompt_with_adv)

            ppl_prompt_adv = get_gpt_ppl([prompt_with_adv], gpt_model, gpt_tokenizer, "cuda:0")
            ppls_prompt_adv.extend(ppl_prompt_adv)

    print("PPL:", sum(ppls_prompt_adv) / len(ppls_prompt_adv))

    return sum(ppls_prompt_adv) / len(ppls_prompt_adv)

def run():
    # 检查 GPU 可用性
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if available_gpus > 0:
            torch.cuda.set_device(0)
            print(f"Using GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPUs are available.")
    else:
        print("CUDA is not available, using CPU instead.")

    if os.name == 'nt':  # Windows系统
        model_name = os.path.join("D:", "\ZLCODE", "model", "vicuna-7b-v1.5")
    else:  # Linux 或其他系统
        model_name = os.path.join("/home/zl/ZLCODE/model", "vicuna-7b-v1.5")

    print(model_name)

    gpt_model, gpt_tokenizer = load_model_and_tokenizer(model_name, low_cpu_mem_usage=True, use_cache=False,
                                                        device="cuda")

    # 指定存放 CSV 文件的文件夹路径
    folder_path = r"D:\ZLCODE\COLD-Attack\comparison\AutoDAN"  # 请根据实际情况修改
    # 获取文件夹下所有 CSV 文件，排除已处理过的（包含"_with_responses"）
    files = [f for f in os.listdir(folder_path)
             if f.endswith(".csv") and "_with_responses" not in f]

    if not files:
        print("未找到需要处理的 CSV 文件！")
        return

    # 遍历处理每个 CSV 文件
    ppls=[]
    for file_name in files:
        csv_file = os.path.join(folder_path, file_name)
        ppl = process_file(csv_file,gpt_model,gpt_tokenizer)
        ppls.append(ppl)

    print(ppls)



if __name__ == "__main__":
    run()
