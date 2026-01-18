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
    "What is the history of artificial intelligence?",
    "Tell me about the structure of the human brain.",
    "What are the major events of World War II?",
    "Explain the theory of evolution by Charles Darwin.",
    "What are the moons of Jupiter?",
    "Describe the process of photosynthesis.",
    "Who discovered gravity?",
    "What are the causes of climate change?",
    "Explain quantum mechanics basics.",
    "Tell me about the culture of ancient Egypt.",
    "April month overview in the Gregorian calendar",
    "Etymology or origin of the name April",
    "April holidays and observances worldwide",
    "Seasonal description of April in both hemispheres",
    "Movable Christian feasts that fall in April",
    "Sayings or phrases about April weather",
    "Historical events that happened in April",
    "April cultural festivals in Europe or Asia",
    "Sports or major events usually held in April",
    "August month overview and calendar facts",
    "Etymology or origin of the name August",
    "August national or religious holidays",
    "August historical events in the 20th century",
    "Definition of art as human creative activity",
    "Categories of art such as visual or performing",
    "Discussion of art versus design",
    "Short history outline of art across eras",
    "Examples of everyday objects treated as art",
    "Comparison of April seasons across hemispheres",
    "August cultural festivals and public holidays"
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

    data_file = f"{BASE_DATA_DIR}/simple{dataset}.json"
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
    centroids = np.load(f"{index_dir}/centroids_raw.npy")
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

    avg_trap = total_trap / len(queries)
    avg_t = total_time / len(queries)

    print(f"\n✅ Avg trap time: {avg_trap:.4f}s")
    print(f"\n✅ Avg query time: {avg_t:.4f}s")

    # save
    out_file = f"{OUTPUT_DIR}/{dataset}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    for ds in ["wiki_1k", "wiki_5k", "wiki_10k"]:
        run_plain_eval(ds)
