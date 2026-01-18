#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_bloom_coverage.py
用于诊断 secure_search 中的候选集合 (BloomFilter + pieces) 是否包含明文最优簇。
"""

import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pybloom_live import BloomFilter
from scipy.stats import spearmanr

INDEX_DIR = r'../outputs/index'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def load_index(dataset):
    dataset_dir = os.path.join(INDEX_DIR, dataset)
    centroids_unit = np.load(os.path.join(dataset_dir, 'centroids_unit.npy'))
    with open(os.path.join(dataset_dir, 'pieces.json'), 'r', encoding='utf-8') as f:
        pieces = json.load(f)
    with open(os.path.join(dataset_dir, 'index_meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return centroids_unit, pieces, meta


def encode_query(q):
    model = SentenceTransformer(MODEL_NAME)
    v = model.encode([q], convert_to_numpy=True)[0]
    v /= np.linalg.norm(v) + 1e-12
    return v


def generate_trapdoor(query, dataset):
    """轻量版本，仅用于诊断"""
    from generate_trapdoor import generate_trapdoor
    return generate_trapdoor(query, dataset)


def diagnose_query(query, dataset):
    centroids_unit, pieces, meta = load_index(dataset)
    t1, t2 = generate_trapdoor(query, dataset)

    bf = pickle.loads(t1['bloom'])
    candidate_keys = [k for k in pieces.keys() if k in bf]
    centroid_indices = sorted({int(idx) for k in candidate_keys for idx in pieces[k]})

    q_unit = encode_query(query)
    all_cos = np.dot(centroids_unit, q_unit)
    plain_best = int(np.argmax(all_cos))
    covered = plain_best in centroid_indices

    # 计算排名相似性（可选）
    sim_cos_candidates = [all_cos[idx] for idx in centroid_indices]
    if len(sim_cos_candidates) > 2:
        rho, _ = spearmanr(sim_cos_candidates, list(range(len(sim_cos_candidates))))
    else:
        rho = np.nan

    print(f"\n[QUERY] {query}")
    print(f" - plain_best={plain_best}")
    print(f" - candidate_pieces={len(candidate_keys)} | candidate_centroids={len(centroid_indices)}")
    print(f" - covered={covered} | Spearman≈{rho:.3f}")

    return covered


def main():
    dataset = 'enron_1k'
    queries = [
        "What meetings are scheduled?",
        "Tell me about energy trading",
        "What contracts were discussed?",
        "What are the price forecasts?",
        "What companies are involved?",
        "What conference calls are planned?"
    ]

    print(f"\n=== Diagnosing Bloom coverage for dataset={dataset} ===")
    covered_count = 0

    for q in queries:
        if diagnose_query(q, dataset):
            covered_count += 1

    print("\n=== Summary ===")
    print(f"Total queries: {len(queries)}")
    print(f"Covered plain_best: {covered_count}")
    print(f"Coverage ratio: {covered_count / len(queries):.3f}")


if __name__ == '__main__':
    main()
