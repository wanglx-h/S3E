#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明文 baseline，与密态聚类完全对齐版本（适配 index_k 下的 k=10/20/40 实验）
- 载入密态保存的 centroids_unit.npy / assignments.npy
- 不重新聚类
- Sentence-BERT 编码查询 & 文档
"""

import os, json, time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DATA_DIR = "/root/siton-tmp/data"
# 使用新的索引目录：和上一段构建索引的脚本保持一致
BASE_INDEX_DIR = "/root/siton-tmp/outputs/index_k"
# 结果输出目录：区分 k，以便后续实验
OUTPUT_DIR = "/root/siton-tmp/outputs/plain_results_k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 文档向量缓存目录（与聚类索引解耦，k=10/20/40 共用一份）
DOC_VEC_DIR = "/root/siton-tmp/outputs/doc_vecs"
os.makedirs(DOC_VEC_DIR, exist_ok=True)

queries = [
    "What meetings are scheduled?",
    "Tell me about energy trading",
    "What contracts were discussed?",
    "What are the price forecasts?",
    "What reports need analysis?",
    "What projects are in development?",
    "What companies are involved?",
    "What emails need attention?",
    "What conference calls are planned?",
    "What financial information is available?",
    "Emails about SEC strategy meetings",
    "Messages mentioning building access or badges",
    "HR newsletters on labor or employment policy",
    "Forwards with BNA Daily Labor Report content",
    "Memos on minimum wage or unemployment issues",
    "Emails discussing union negotiations or wage increases",
    "Messages about post-9/11 employment impacts",
    "Notes on federal worker discrimination or whistleblower cases",
    "Emails that list multiple labor news headlines",
    "Messages sharing external news links with login info",
    "Internal calendar or on-call notification emails",
    "Emails between facilities or admin staff about office locations",
    "Messages referencing ILO or international labor standards",
    "Forwards about appointments to U.S. labor-related posts",
    "Emails on benefit or donation program changes",
    "Threads with multiple HR recipients in one blast",
    "Messages mentioning airport security or related legislation",
    "Emails summarizing congressional labor actions",
    "Messages about court rulings on workplace drug testing",
    "Long digest-style labor and employment updates"
]

# 与你之前一致的本地模型路径
model = SentenceTransformer("/root/siton-tmp/all-MiniLM-L6-v2")

def load_docs(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    texts = [d.get("content") or d.get("text") for d in docs]
    ids = [d.get("id", str(i)) for i, d in enumerate(docs)]
    return docs, texts, ids

def encode(texts):
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def run_plain_eval(dataset, K):
    """
    在 index_k/k{K}/{dataset} 下的索引上做实验，并把结果保存到
    /root/siton-tmp/outputs/plain_results_k/{dataset}_k{K}.json
    """
    print(f"\n========== DATASET: {dataset}, K={K} ==========")

    data_file = f"{BASE_DATA_DIR}/{dataset}.json"
    # 索引目录示例：/root/siton-tmp/outputs/index_k/k10/enron_5k
    index_dir = f"{BASE_INDEX_DIR}/k{K}/{dataset}"

    # Load docs
    docs, texts, ids = load_docs(data_file)

    # 文档向量缓存（与 k 无关，共用一份）
    vec_path = os.path.join(DOC_VEC_DIR, f"{dataset}_doc_vecs.npy")
    if os.path.exists(vec_path):
        doc_vecs = np.load(vec_path)
        print(f"[INFO] Loaded cached vectors: {doc_vecs.shape}")
    else:
        doc_vecs = encode(texts)
        np.save(vec_path, doc_vecs)
        print(f"[INFO] Encoded docs and saved cache: {doc_vecs.shape}")

    # Load cluster info from encrypted-side index (但这里是明文实验)
    centroids = np.load(os.path.join(index_dir, "centroids_unit.npy"))
    assignments = np.load(os.path.join(index_dir, "assignments.npy"))

    results_all = {}
    total_trap = 0.0   # query->vec 时间
    total_time = 0.0   # 聚类内搜索时间

    for q in queries:
        print(f"\n[QUERY] {q}")
        # trapdoor / query embedding time
        t_trap1 = time.time()
        q_vec = encode([q])
        total_trap += time.time() - t_trap1

        # 实际检索时间
        t0 = time.time()

        # 先和所有簇中心算相似度，选最相关簇
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
                "rank": rank+1,
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

    print(f"\n✅ Avg trap time (query encoding): {avg_trap:.4f}s")
    print(f"✅ Avg query time (cluster search): {avg_t:.4f}s")

    # save
    out_file = os.path.join(OUTPUT_DIR, f"{dataset}_k{K}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    # 只对 enron_5k，且在 k=10,20,40 三种容量下分别跑实验
    dataset = "enron_5k"
    for K in [10, 20, 40]:
        run_plain_eval(dataset, K)
