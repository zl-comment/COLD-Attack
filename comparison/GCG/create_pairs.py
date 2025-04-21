import pandas as pd
import json
import os

# Read the harmful behaviors CSV
csv_path = r"d:\ZLCODE\COLD-Attack\data\advbench\harmful_behaviors_custom.csv"
df_behaviors = pd.read_csv(csv_path)

# Create a list to store the pairs (moved this line outside the loop)
pairs = []

# For each index from 0 to 49
for j in range(50):
    # Construct the filename
    filename = rf"D:\ZLCODE\COLD-Attack\comparison\GCG\Mistral-7B-Instruct-v0.2\gcg\len20_2000step_bs8_seed20_l50_ce-all-t1.0_static_space_k256_orig\AdvBenchAll_{j:02d}.jsonl"

    # Read the JSONL file - 每行一个JSON对象
    with open(filename, 'r', encoding='utf-8') as f:
        # 读取所有行
        lines = f.readlines()
        # 计算步长来确保均匀间隔
        step = len(lines) // 8

        for i in range(8):
            # 选择均匀间隔的数据
            line = lines[i * step]
            data = json.loads(line)
            suffix = data['suffix']
            generated = data['generated']

            # 获取对应的目标
            goal = df_behaviors.iloc[j]['goal']

            # 将数据添加到pairs
            pairs.append({
                'prompt': goal,
                'adv': suffix,
                'prompt_with_adv': goal + " " + suffix,
                'output': generated
            })

# Create DataFrame and save to CSV
df_pairs = pd.DataFrame(pairs)
os.makedirs('../../comparison/GCG/', exist_ok=True)
df_pairs.to_csv('../../comparison/GCG/Mistral_0_50.csv', index=False)
