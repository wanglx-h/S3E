#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明文 baseline，与密态聚类完全对齐版本
- 载入密态保存的 centroids_unit.npy / assignments.npy
- 不重新聚类
- Sentence-BERT 编码查询 & 文档
"""

import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 数据集映射：键是数据集名称，值是对应的 JSON 路径
DATASETS = {
    "msmarco_1k": "/root/siton-tmp/data/msmarco_1k.json",
    "msmarco_5k": "/root/siton-tmp/data/msmarco_5k.json",
    "msmarco_10k": "/root/siton-tmp/data/msmarco_10k.json",
}

# 与密态索引路径保持一致：假定每个数据集在 outputs/index/<dataset_name>/ 下
BASE_INDEX_DIR = "/root/siton-tmp/outputs/index"

# 明文检索结果输出目录
OUTPUT_DIR = "/root/siton-tmp/outputs/plain_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 查询语句（15 条）
queries = [
    "How do you use the Stefan-Boltzmann law to calculate the radius of a star such as Rigel from its luminosity and surface temperature?",
    "What developmental milestones and typical behaviors should you expect from an 8 year old child at home and at school?",
    "What are the symptoms of a head lice infestation and how can you check for lice, eggs, and nits on a child's scalp?",
    "What special features does the Burj Khalifa in Dubai have and why was it renamed from Burj Dubai?",
    "What kinds of homes and land are for sale near La Grange, California, and what are their typical sizes and prices?",
    "What are the main characteristics, temperament, and exercise needs of the Dogo Argentino dog breed?",
    "How are custom safety nets used in industry and what kinds of clients and applications does a company like US Netting serve?",
    "What are effective ways to remove weeds from a garden and prevent them from coming back?",
    "How common is urinary incontinence in the United States, what can cause it, and is it just a normal part of aging?",
    "How did President Franklin D. Roosevelt prepare the United States for World War II before Pearl Harbor while the country was still isolationist?",
    "If you have multiple sclerosis and difficulty swallowing pills, is it safe to crush Valium and other medications to make them easier to swallow?",
    "What strategies can help you get better results when dealing with customer service representatives at cable companies or airlines?",
    "In Spanish, what does the word 'machacado' mean and how is the verb 'machacar' used in different contexts?",
    "When building a concrete path, how should you design and support plywood formwork so that it is strong enough and keeps the concrete in place?",
    "Why do people join political parties, and which political party did U.S. presidents Woodrow Wilson and Herbert Hoover belong to?",
]

# 明文 baseline 使用的编码模型（保持你原来的设置）
model = SentenceTransformer("/root/siton-tmp/all-MiniLM-L6-v2")


# def load_docs(json_path):
#     """加载 JSON 文档，兼容 content/text 字段"""
#     with open(json_path, "r", encoding="utf-8") as f:
#         docs = json.load(f)
#     texts = [d.get("content") or d.get("text") for d in docs]
#     ids = [d.get("id", str(i)) for i, d in enumerate(docs)]
#     return docs, texts, ids
def load_docs(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = []
    ids = []

    for i, d in enumerate(docs):
        text = d.get("content") or d.get("text")
        if text is None:
            continue
        # 关键修改：优先用 MS MARCO 的 doc_id
        doc_id = d.get("doc_id")
        if doc_id is None:
            # 兜底：没有 doc_id 才用 index
            doc_id = str(i)

        ids.append(str(doc_id))
        texts.append(text)

    return docs, texts, ids



def encode(texts):
    """对文本列表编码为向量（单位化）"""
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def run_plain_eval(dataset_name: str, json_path: str):
    """
    针对单个数据集运行明文 baseline：
    - 读取 docs
    - 载入密态侧的聚类信息（centroids_unit.npy / assignments.npy）
    - 按密态聚类结构在簇内重排
    """
    print(f"\n========== DATASET: {dataset_name} ==========")
    print(f"[INFO] JSON path : {json_path}")

    index_dir = f"{BASE_INDEX_DIR}/{dataset_name}"

    if not os.path.exists(json_path):
        print(f"[ERROR] Data file not found: {json_path}")
        return
    if not os.path.exists(index_dir):
        print(f"[ERROR] Index dir not found: {index_dir}")
        return

    # Load docs
    docs, texts, ids = load_docs(json_path)
    print(f"[INFO] Loaded {len(docs)} docs")

    # Load vectors if cached, else encode
    vec_path = f"{index_dir}/doc_vecs.npy"
    if os.path.exists(vec_path):
        doc_vecs = np.load(vec_path)
        print(f"[INFO] Loaded cached doc vectors: {doc_vecs.shape}")
    else:
        doc_vecs = encode(texts)
        os.makedirs(index_dir, exist_ok=True)
        np.save(vec_path, doc_vecs)
        print(f"[INFO] Encoded docs and saved cache to {vec_path}")

    # Load encrypted-side cluster info
    centroids_path = f"{index_dir}/centroids_unit.npy"
    assignments_path = f"{index_dir}/assignments.npy"
    if not (os.path.exists(centroids_path) and os.path.exists(assignments_path)):
        print(f"[ERROR] Missing centroids or assignments in {index_dir}")
        return

    centroids = np.load(centroids_path)       # (n_clusters, dim)
    assignments = np.load(assignments_path)   # (num_docs,)

    results_all = {}
    total_trap = 0.0   # 查询构建时间
    total_time = 0.0   # 查询 + 簇内重排时间

    for q in queries:
        print(f"\n[QUERY] {q}")
        # 查询向量构建时间
        t_trap1 = time.time()
        q_vec = encode([q])
        total_trap += time.time() - t_trap1

        # 查询执行时间（选簇 + 簇内重排）
        t0 = time.time()

        # 选最近簇
        sim_cent = cosine_similarity(q_vec, centroids)[0]
        best_cluster = int(np.argmax(sim_cent))

        # docs in cluster
        idxs = np.where(assignments == best_cluster)[0]
        sims = cosine_similarity(q_vec, doc_vecs[idxs])[0]

        order = sims.argsort()[::-1]
        results = []
        for rank, sel in enumerate(order):
            did = ids[idxs[sel]]
            txt = docs[idxs[sel]].get("content") or docs[idxs[sel]].get("text")
            results.append({
                "rank": rank + 1,
                "id": did,
                "text": txt,
                "similarity": float(sims[sel]),
                "cluster": best_cluster
            })
            if rank < 5:
                print(f"  [{rank+1}] sim={sims[sel]:.4f} id={did}")

        total_time += time.time() - t0
        results_all[q] = results

    avg_trap = total_trap / len(queries)
    avg_t = total_time / len(queries)

    print(f"\n✅ Avg query-build (trap) time : {avg_trap:.4f}s")
    print(f"✅ Avg query+reorder time     : {avg_t:.4f}s")

    # 保存结果：文件名中使用数据集名称（msmarco_1k / 5k / 10k）
    out_file = f"{OUTPUT_DIR}/{dataset_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved results to {out_file}")


if __name__ == "__main__":
    for ds_name, ds_path in DATASETS.items():
        run_plain_eval(ds_name, ds_path)
