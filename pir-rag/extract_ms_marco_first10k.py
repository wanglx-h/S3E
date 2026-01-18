import os
import json

# 1) 输入文件：已经解压好的 10k 子集 TSV
#    你说文件在 /root/siton-tmp/data/msmarco_10k_subset.tsv
TSV_PATH = "/root/siton-tmp/data/msmarco_10k_subset.tsv"

# 2) 输出文件：分别包含 1000 / 5000 / 10000 条文档（doc_id + text）
OUTPUTS = {
    1000: "/root/siton-tmp/data/msmarco_1k.json",
    5000: "/root/siton-tmp/data/msmarco_5k.json",
    10000: "/root/siton-tmp/data/msmarco_10k.json",
}

# 确保输出目录存在
for path in OUTPUTS.values():
    os.makedirs(os.path.dirname(path), exist_ok=True)

# 为不同规模准备缓存与计数器
buffers = {k: [] for k in OUTPUTS}
counters = {k: 0 for k in OUTPUTS}
max_needed = max(OUTPUTS.keys())

print(f"Reading TSV from: {TSV_PATH}")
num_lines = 0

with open(TSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        num_lines += 1

        # MS MARCO 文档行格式：docid \t url \t title \t body
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            # 有异常行就跳过
            continue

        doc_id = parts[0]
        body = parts[3]  # 第 4 列是正文 body

        record = {
            "doc_id": doc_id,
            "text": body,   # 方案 B：仅 doc_id + text
        }

        # 分别填充 1k / 5k / 10k 的 buffer
        for k in OUTPUTS:
            if counters[k] < k:
                buffers[k].append(record)
                counters[k] += 1

        # 当前已经取得 10,000 条后就可以停止
        if counters[max_needed] >= max_needed:
            break

print(f"Read {num_lines} lines from TSV (stopped after reaching {max_needed} docs).")

# 写出 JSON 文件
for k, path in OUTPUTS.items():
    docs = buffers[k]
    with open(path, "w", encoding="utf-8") as out_f:
        json.dump(docs, out_f, ensure_ascii=False, indent=2)
    print(f"Saved {len(docs)} docs -> {path}")

print("All done.")
