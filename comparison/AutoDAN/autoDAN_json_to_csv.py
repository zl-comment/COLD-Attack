import json
import csv

# 读取 JSON 文件 (请确认文件路径正确)
with open(r'D:\ZLCODE\COLD-Attack\comparison\AutoDAN\hga\mistral-7b\mistral-7b_0_50_normal_2000.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []
# 遍历每个 JSON 对象（key 如 "0", "1", ...）
for key, record in data.items():
    goal = record.get("goal", "")
    target = record.get("target", "")
    log = record.get("log", {})

    # 获取 log 中的 suffix, respond, success 数组
    suffixes = log.get("suffix", [])
    responds = log.get("respond", [])
    successes = log.get("success", [])

    # 确定最大行数，如果数组长度不一致就取最长的
    max_len = max(len(suffixes), len(responds), len(successes))
    limit = min(max_len, 8)
    start_index = max_len - limit  # 从这里开始取，保证取最后八条

    # 迭代索引从 start_index 到 max_len
    for i in range(start_index, max_len):
        row = {
            "goal": goal,
            "target": target,
            "suffix": suffixes[i] if i < len(suffixes) else "",
            "respond": responds[i] if i < len(responds) else "",
            "success": successes[i] if i < len(successes) else ""
        }
        rows.append(row)

# 定义 CSV 列名
fieldnames = ["goal", "target", "suffix", "respond", "success"]

# 写入 CSV 文件
with open("hga_mistral-7b_0_50_normal_2000.csv", "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("CSV 文件生成成功！")
