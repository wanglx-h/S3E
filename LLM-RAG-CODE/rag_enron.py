#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明文 baseline，与密态聚类完全对齐版本
- 载入密态保存的 centroids_unit.npy / assignments.npy
- 不重新聚类
- Sentence-BERT 编码查询 & 文档
"""

import os, json, time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DATA_DIR = "/root/siton-tmp/data"
BASE_INDEX_DIR = "/root/siton-tmp/outputs/index"
OUTPUT_DIR = "/root/siton-tmp/outputs/plain_results"


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

model = SentenceTransformer("/root/siton-tmp/all-MiniLM-L6-v2")

def load_docs(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    texts = [d.get("content") or d.get("text") for d in docs]
    ids = [d.get("id", str(i)) for i, d in enumerate(docs)]
    return docs, texts, ids

def encode(texts):
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def run_plain_eval(dataset):
    print(f"\n========== DATASET: {dataset} ==========")

    data_file = f"{BASE_DATA_DIR}/{dataset}.json"
    index_dir = f"{BASE_INDEX_DIR}/{dataset}"

    # Load docs
    docs, texts, ids = load_docs(data_file)

    # Load vectors if cached, else encode
    vec_path = f"{BASE_INDEX_DIR}/{dataset}/doc_vecs.npy"
    if os.path.exists(vec_path):
        doc_vecs = np.load(vec_path)
        print(f"[INFO] Loaded cached vectors: {doc_vecs.shape}")
    else:
        doc_vecs = encode(texts)
        np.save(vec_path, doc_vecs)
        print(f"[INFO] Encoded docs and saved cache")

    # Load encrypted-side cluster info
    centroids = np.load(f"{index_dir}/centroids_unit.npy")
    assignments = np.load(f"{index_dir}/assignments.npy")

    results_all = {}
    total_trap = 0.0
    total_time = 0.0

    for q in queries:
        print(f"\n[QUERY] {q}")
        t_trap1 = time.time()
        q_vec = encode([q])
        total_trap += time.time() - t_trap1

        t0 = time.time()

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
    avg_trap = total_trap/len(queries)
    avg_t = total_time/len(queries)

    print(f"\n✅ Avg trap time: {avg_trap:.4f}s")
    print(f"\n✅ Avg query time: {avg_t:.4f}s")

    # save
    out_file = f"{OUTPUT_DIR}/{dataset}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    for ds in ["enron_1k", "enron_5k", "enron_10k"]:
        run_plain_eval(ds)

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Enron 多数据集语义聚类与查询实验系统 (增强版)
# ================================================
# 功能：
# 1. 自动处理 enron_1k.json / enron_5k.json / enron_10k.json；
# 2. 使用 Sentence-BERT 编码文档向量；
# 3. 使用 MiniBatchKMeans 聚类；
# 4. 对 10 条查询语句执行语义检索；
# 5. 输出索引建立与查询耗时统计；
# 6. 保存每个数据集的 Top-k 搜索结果。
# """
#
# import json
# import time
# import numpy as np
# from typing import List, Dict, Any
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import os
#
#
# class SemanticClusterSearch:
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", n_clusters: int = 50):
#         self.model = SentenceTransformer(model_name)
#         self.n_clusters = n_clusters
#         self.kmeans = None
#         self.centroids = None
#         self.doc_vectors = None
#         self.documents = None
#
#     def load_dataset(self, json_path: str):
#         """加载 JSON 格式数据集"""
#         print(f"[INFO] Loading dataset from {json_path} ...")
#         with open(json_path, "r", encoding="utf-8") as f:
#             self.documents = json.load(f)
#         print(f"[INFO] Loaded {len(self.documents)} documents.")
#
#     def encode_documents(self) -> float:
#         """Sentence-BERT 向量化，返回耗时（批量编码）"""
#         print("[INFO] Encoding documents ...")
#
#         # 兼容 text 或 content 字段
#         texts = []
#         valid_docs = []
#         for d in self.documents:
#             t = d.get("text") or d.get("content")
#             if t:
#                 texts.append(t)
#                 valid_docs.append(d)
#         self.documents = valid_docs  # 更新有效文档列表
#
#         start = time.time()
#         self.doc_vectors = self.model.encode(
#             texts,
#             show_progress_bar=True,
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#             batch_size=32   # <--- 添加批量编码
#         )
#         end = time.time()
#         duration = end - start
#         print(f"[INFO] Document vectors shape: {self.doc_vectors.shape} (time: {duration:.2f}s)")
#         return duration
#
#     def cluster_documents(self) -> float:
#         """MiniBatchKMeans 聚类，返回耗时"""
#         print(f"[INFO] Clustering {len(self.documents)} documents into {self.n_clusters} clusters...")
#         start = time.time()
#         self.kmeans = MiniBatchKMeans(
#             n_clusters=self.n_clusters, random_state=42, batch_size=512, max_iter=100
#         )
#         self.kmeans.fit(self.doc_vectors)
#         self.centroids = self.kmeans.cluster_centers_
#         end = time.time()
#         duration = end - start
#         print(f"[INFO] Clustering complete. (time: {duration:.2f}s)")
#         return duration
#
#     # def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#     #     """输入查询语句，返回最相关的 top-k 文档"""
#     #     start = time.time()
#     #     query_vec = self.model.encode([query], normalize_embeddings=True)
#     #     sim_to_clusters = cosine_similarity(query_vec, self.centroids)[0]
#     #     most_sim_cluster_idx = np.argmax(sim_to_clusters)
#     #
#     #     cluster_labels = self.kmeans.labels_
#     #     target_docs_idx = np.where(cluster_labels == most_sim_cluster_idx)[0]
#     #     cluster_vectors = self.doc_vectors[target_docs_idx]
#     #     sims = cosine_similarity(query_vec, cluster_vectors)[0]
#     #
#     #     top_indices = sims.argsort()[-top_k:][::-1]
#     #     results = []
#     #     for rank, idx in enumerate(top_indices):
#     #         doc = self.documents[target_docs_idx[idx]]
#     #         results.append({
#     #             "rank": rank + 1,
#     #             "id": doc.get("id", str(target_docs_idx[idx])),
#     #             "text": doc["content"],
#     #             "similarity": float(sims[idx])
#     #         })
#     #     end = time.time()
#     #     results_time = end - start
#     #     return results, results_time
#
#
#     def search(self, query: str) -> List[Dict[str, Any]]:
#         """
#         输入查询语句，返回最相关簇中的所有文档，按相似度降序排列
#         """
#         start = time.time()
#         # 计算查询向量
#         query_vec = self.model.encode([query], normalize_embeddings=True)
#         # 计算查询向量与簇中心的相似度
#         sim_to_clusters = cosine_similarity(query_vec, self.centroids)[0]
#         # 找到最相关的簇索引
#         most_sim_cluster_idx = np.argmax(sim_to_clusters)
#
#         # 找出该簇内所有文档
#         cluster_labels = self.kmeans.labels_
#         target_docs_idx = np.where(cluster_labels == most_sim_cluster_idx)[0]
#         cluster_vectors = self.doc_vectors[target_docs_idx]
#         sims = cosine_similarity(query_vec, cluster_vectors)[0]
#
#         # 按相似度降序排列
#         sorted_indices = sims.argsort()[::-1]
#
#         results = []
#         for rank, idx in enumerate(sorted_indices):
#             doc = self.documents[target_docs_idx[idx]]
#             results.append({
#                 "rank": rank + 1,
#                 "id": doc.get("id", str(target_docs_idx[idx])),
#                 "text": doc.get("content") or doc.get("text"),
#                 "similarity": float(sims[idx])
#             })
#
#         end = time.time()
#         results_time = end - start
#         return results, results_time
#
#
#
# # ====================================================
# # 主程序入口
# # ====================================================
# if __name__ == "__main__":
#     dataset_paths = [
#         "D:/SSE/paper/data/enron_1k.json",
#         "D:/SSE/paper/data/enron_5k.json",
#         "D:/SSE/paper/data/enron_10k.json"
#     ]
#
#     queries = [
#         "What meetings are scheduled?",
#         "Tell me about energy trading",
#         "What contracts were discussed?",
#         "What are the price forecasts?",
#         "What reports need analysis?",
#         "What projects are in development?",
#         "What companies are involved?",
#         "What emails need attention?",
#         "What conference calls are planned?",
#         "What financial information is available?"
#     ]
#
#     n_clusters = 200
#     system = SemanticClusterSearch(n_clusters=n_clusters)
#
#     for data_path in dataset_paths:
#         print("\n" + "=" * 100)
#         print(f"🔹 Running experiment on dataset: {data_path}")
#         print("=" * 100)
#
#         if not os.path.exists(data_path):
#             print(f"[WARN] Dataset file not found: {data_path}")
#             continue
#
#         with open(data_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         N = len(data)
#         top_k = max(1, N // n_clusters)
#         print(f"[INFO] Dataset size: {N}, n_clusters: {n_clusters}, top_k: {top_k}")
#
#         # ---- 索引建立时间统计 ----
#         system.load_dataset(data_path)
#         t_encode = system.encode_documents()
#         t_cluster = system.cluster_documents()
#         t_index_total = t_encode + t_cluster
#         print(f"[TIME] Indexing time: encode={t_encode:.2f}s, cluster={t_cluster:.2f}s, total={t_index_total:.2f}s")
#
#         all_results = {}
#         total_search_time = 0.0
#
#         # ---- 搜索与计时 ----
#         for q in queries:
#             print(f"\n[QUERY] {q}")
#             results, t_search = system.search(q)
#             total_search_time += t_search
#
#             for r in results:
#                 print(f"  [{r['rank']}] (sim={r['similarity']:.4f}) {r['id']}: {r['text'][:120]}...")
#
#             all_results[q] = results
#
#         avg_search_time = total_search_time / len(queries)
#         print(f"\n[TIME] Average search time per query: {avg_search_time:.4f}s")
#
#         # ---- 保存结果 ----
#         output_file = f"results_{os.path.splitext(os.path.basename(data_path))[0]}_11.json"
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(all_results, f, ensure_ascii=False, indent=2)
#
#         print(f"[INFO] Results saved to {output_file}")
#         print(f"[SUMMARY] Dataset: {N} docs | Index time: {t_index_total:.2f}s | Avg search: {avg_search_time:.4f}s\n")
