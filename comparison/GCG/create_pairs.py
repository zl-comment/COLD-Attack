import os
import json
import pandas as pd
from pathlib import Path

def process_jsonl_folder(jsonl_folder: Path, df_behaviors: pd.DataFrame, output_csv: Path):
    pairs = []
    # 按文件名排序，确保 AdvBenchAll_00 … AdvBenchAll_49 对应 index 0–49
    jsonl_files = sorted(jsonl_folder.glob("*.jsonl"))
    for j, jsonl_path in enumerate(jsonl_files):
        # 读取所有行
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        step = max(1, len(lines) // 8)
        for i in range(8):
            idx = min(i * step, len(lines) - 1)
            data = json.loads(lines[idx])
            suffix    = data.get("suffix", "")
            generated = data.get("generated", "")
            # 从 df_behaviors 取对应的 goal
            goal = df_behaviors.iloc[j]["goal"]
            pairs.append({
                "prompt":           goal,
                "adv":              suffix,
                "prompt_with_adv":  f"{goal} {suffix}",
                "output":           generated
            })

    # 写出 CSV
    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(output_csv, index=False)
    print(f"→ Saved CSV: {output_csv}")

def main():
    df_behaviors = pd.read_csv(r"d:\ZLCODE\COLD-Attack\data\advbench\harmful_behaviors_custom.csv")

    root = Path(r"D:\ZLCODE\COLD-Attack\comparison\GCG")
    # 确保输出目录存在（在这里就是 root 本身）
    output_dir = root
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        for method_dir in model_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for config_dir in method_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                if any(config_dir.glob("*.jsonl")):
                    # 统一写到 root 目录下
                    filename = f"{model_dir.name}_0_50.csv"
                    out_csv = output_dir / filename
                    process_jsonl_folder(config_dir, df_behaviors, out_csv)
                    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
