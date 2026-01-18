# import os
# import json
# import pickle
# import numpy as np
#
# BASE_DIR = "/root/siton-tmp"
# INDEX_DIR = os.path.join(BASE_DIR, "outputs", "index")
#
# DATASETS = [
#     "enron_1k",
#     "enron_5k",
#     "enron_10k",
#     "wiki_1k",
#     "wiki_5k",
#     "wiki_10k",
# ]
#
# def _split_centroids(ddir: str, seed: int = 2025):
#     """
#     使用 centroids_unit.npy -> SCALE -> 四舍五入 -> 拆成 share_a / share_b
#     """
#     unit_path = os.path.join(ddir, "centroids_unit.npy")
#     meta_path = os.path.join(ddir, "index_meta.json")
#     if not os.path.exists(unit_path) or not os.path.exists(meta_path):
#         raise FileNotFoundError(f"{unit_path} 或 {meta_path} 不存在，请先跑 build_index.py")
#
#     # 读取单位化聚类中心
#     centroids_unit = np.load(unit_path)
#     with open(meta_path, "r", encoding="utf-8") as f:
#         meta = json.load(f)
#
#     SCALE = int(meta["SCALE"])
#     P = int(meta["P"])
#
#     # 缩放并转换为整数（round -> int64）
#     centroids_int = np.round(centroids_unit * SCALE).astype(np.int64)
#
#     # 使用随机数生成器生成 share_a 和 share_b
#     rng = np.random.default_rng(seed)
#     cA = rng.integers(low=0, high=P, size=centroids_int.shape, dtype=np.int64)
#     cB = (centroids_int.astype(object) - cA.astype(object)) % P
#
#     # 保存两份 share 到文件
#     np.save(os.path.join(ddir, "centroids_share_a.npy"), cA)
#     np.save(os.path.join(ddir, "centroids_share_b.npy"), cB)
#
#     print(f"[OK] {ddir} -> 使用单位化向量拆分完成")
#
#
# def _split_th_all(ddir: str):
#     """
#     拆分 Th_all.pkl 为两份 server_a / server_b
#     """
#     th_all_path = os.path.join(ddir, "Th_all.pkl")
#     if not os.path.exists(th_all_path):
#         print(f"[WARN] {th_all_path} 不存在，跳过 Th 拆分")
#         return
#
#     with open(th_all_path, "rb") as f:
#         th_all = pickle.load(f)
#
#     th_a = {}
#     th_b = {}
#
#     for table_id, bucket in th_all.items():
#         piece_str = bucket.get("piece")
#         clusters = bucket.get("clusters", {})
#         docs_enc = bucket.get("docs_enc", {})
#
#         a_bucket = {
#             "piece": piece_str,
#             "clusters": clusters,
#             "docs_enc_part": {},
#         }
#         b_bucket = {
#             "piece": piece_str,
#             "clusters": clusters,
#             "docs_enc_part": {},
#         }
#
#         for cid_str, blob_hex in docs_enc.items():
#             blob = bytes.fromhex(blob_hex)
#             if len(blob) <= 28:
#                 part_a = blob
#                 part_b = b""
#             else:
#                 nonce_tag = blob[:28]
#                 ct = blob[28:]
#                 half_ct = len(ct) // 2
#                 part_a = nonce_tag + ct[:half_ct]
#                 part_b = ct[half_ct:]
#
#             a_bucket["docs_enc_part"][cid_str] = part_a.hex()
#             b_bucket["docs_enc_part"][cid_str] = part_b.hex()
#
#         th_a[table_id] = a_bucket
#         th_b[table_id] = b_bucket
#
#     server_a_dir = os.path.join(ddir, "server_a")
#     server_b_dir = os.path.join(ddir, "server_b")
#     os.makedirs(server_a_dir, exist_ok=True)
#     os.makedirs(server_b_dir, exist_ok=True)
#
#     with open(os.path.join(server_a_dir, "Th.pkl"), "wb") as f:
#         pickle.dump(th_a, f)
#     with open(os.path.join(server_b_dir, "Th.pkl"), "wb") as f:
#         pickle.dump(th_b, f)
#
#     print(f"[OK] {ddir} -> server_a/Th.pkl & server_b/Th.pkl")
#
#
# def make_index_shares_for_dataset(dataset: str):
#     ddir = os.path.join(INDEX_DIR, dataset)
#     if not os.path.isdir(ddir):
#         print(f"[WARN] {ddir} 不存在，跳过")
#         return
#
#     # 调用拆分函数
#     _split_centroids(ddir)
#     _split_th_all(ddir)
#
#
# if __name__ == "__main__":
#     for ds in DATASETS:
#         make_index_shares_for_dataset(ds)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import numpy as np

BASE_DIR = "/root/siton-tmp"
INDEX_DIR = os.path.join(BASE_DIR, "outputs", "index")

# 改为 MS MARCO 三个子集
DATASETS = [
    "msmarco_1k",
    "msmarco_5k",
    "msmarco_10k",
]


def _split_centroids(ddir: str, seed: int = 2025):
    """
    使用 centroids_unit.npy -> SCALE -> 四舍五入 -> 拆成 share_a / share_b
    """
    unit_path = os.path.join(ddir, "centroids_unit.npy")
    meta_path = os.path.join(ddir, "index_meta.json")
    if not os.path.exists(unit_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"{unit_path} 或 {meta_path} 不存在，请先跑 build_index.py")

    # 读取单位化聚类中心
    centroids_unit = np.load(unit_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    SCALE = int(meta["SCALE"])
    P = int(meta["P"])

    # 缩放并转换为整数（round -> int64）
    centroids_int = np.round(centroids_unit * SCALE).astype(np.int64)

    # 使用随机数生成器生成 share_a 和 share_b
    rng = np.random.default_rng(seed)
    cA = rng.integers(low=0, high=P, size=centroids_int.shape, dtype=np.int64)
    cB = (centroids_int.astype(object) - cA.astype(object)) % P

    # 保存两份 share 到文件
    np.save(os.path.join(ddir, "centroids_share_a.npy"), cA)
    np.save(os.path.join(ddir, "centroids_share_b.npy"), cB)

    print(f"[OK] {ddir} -> 使用单位化向量拆分完成")


def _split_th_all(ddir: str):
    """
    拆分 Th_all.pkl 为两份 server_a / server_b
    """
    th_all_path = os.path.join(ddir, "Th_all.pkl")
    if not os.path.exists(th_all_path):
        print(f"[WARN] {th_all_path} 不存在，跳过 Th 拆分")
        return

    with open(th_all_path, "rb") as f:
        th_all = pickle.load(f)

    th_a = {}
    th_b = {}

    for table_id, bucket in th_all.items():
        piece_str = bucket.get("piece")
        clusters = bucket.get("clusters", {})
        docs_enc = bucket.get("docs_enc", {})

        a_bucket = {
            "piece": piece_str,
            "clusters": clusters,
            "docs_enc_part": {},
        }
        b_bucket = {
            "piece": piece_str,
            "clusters": clusters,
            "docs_enc_part": {},
        }

        for cid_str, blob_hex in docs_enc.items():
            blob = bytes.fromhex(blob_hex)
            if len(blob) <= 28:
                # 太短的话全部放到 A，B 为空
                part_a = blob
                part_b = b""
            else:
                nonce_tag = blob[:28]
                ct = blob[28:]
                half_ct = len(ct) // 2
                part_a = nonce_tag + ct[:half_ct]
                part_b = ct[half_ct:]

            a_bucket["docs_enc_part"][cid_str] = part_a.hex()
            b_bucket["docs_enc_part"][cid_str] = part_b.hex()

        th_a[table_id] = a_bucket
        th_b[table_id] = b_bucket

    server_a_dir = os.path.join(ddir, "server_a")
    server_b_dir = os.path.join(ddir, "server_b")
    os.makedirs(server_a_dir, exist_ok=True)
    os.makedirs(server_b_dir, exist_ok=True)

    with open(os.path.join(server_a_dir, "Th.pkl"), "wb") as f:
        pickle.dump(th_a, f)
    with open(os.path.join(server_b_dir, "Th.pkl"), "wb") as f:
        pickle.dump(th_b, f)

    print(f"[OK] {ddir} -> server_a/Th.pkl & server_b/Th.pkl")


def make_index_shares_for_dataset(dataset: str):
    ddir = os.path.join(INDEX_DIR, dataset)
    if not os.path.isdir(ddir):
        print(f"[WARN] {ddir} 不存在，跳过")
        return

    # 调用拆分函数
    _split_centroids(ddir)
    _split_th_all(ddir)


if __name__ == "__main__":
    for ds in DATASETS:
        make_index_shares_for_dataset(ds)

