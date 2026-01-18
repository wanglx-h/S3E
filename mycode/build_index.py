"""
build_index.py

生成索引：
1. 聚类 → 容量约束 → 单位化中心
2. 切 piece
3. 把单位化中心按 SCALE 量化，写 centroids_int.npy
4. 生成多重映射表 Th(k,l)：
   - key = HMAC(index_key, piece_str)
   - value 里放：这个 piece 里的所有 cluster 的量化中心 + 这个 cluster 下的文档 id 列表的 AES 密文
5. index_key.bin 里存 AES key (32 bytes)
"""

import os
import time

import math
import json
import random
import hmac
import hashlib
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# ----- 简单工具 -----
def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class Timer:
    def start(self):
        import time
        self._t = time.time()
    def stop(self):
        import time
        return time.time() - self._t

# ====== 配置 ======
VEC_DIR = "../outputs/vectors"
OUT_DIR = "../outputs/index"

T = 32
P = 2 ** 61 - 1
PER_PIECE = T // 8
RANDOM_SEED = 42
ALPHA = 0.85
MAX_PER_CLUSTER = 20

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def aes_encrypt_aesgcm(plaintext: bytes, key: bytes) -> bytes:
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce + tag + ciphertext  # 12 + 16 + N


def unitize(vectors: np.ndarray) -> np.ndarray:
    """将向量归一化为单位向量"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12  # 防止除以零
    return vectors / norms


# def angle(center, vec):
#     cos_val = np.clip(
#         np.dot(center, vec)
#         / ((np.linalg.norm(center) + 1e-12) * (np.linalg.norm(vec) + 1e-12)),
#         -1.0, 1.0
#     )
#     return math.acos(cos_val)


# def build_pieces_strict(centroids_raw: np.ndarray):
#     # 1) 用原始聚类中心算 O
#     O = np.mean(centroids_raw, axis=0)
#
#     # 2) 按角度分 4 个扇区，同时记录半径
#     sector = {0: [], 1: [], 2: [], 3: []}
#     for idx, c in enumerate(centroids_raw):
#         th = angle(O, c)
#         if 0 < th <= math.pi / 4:
#             f = 0
#         elif math.pi / 4 < th <= math.pi / 2:
#             f = 1
#         elif math.pi / 2 < th <= 3 * math.pi / 4:
#             f = 2
#         else:
#             f = 3
#         r = np.linalg.norm(c - O)
#         sector[f].append((idx, r))
#
#     # 3) 每个扇区按半径升序
#     for f in range(4):
#         sector[f].sort(key=lambda x: x[1])
#
#     PER = PER_PIECE  # 你的配置里是 4
#     ptr = {f: 0 for f in range(4)}  # 每个扇区当前用到了哪儿
#
#     pieces = {}
#     j = 1  # 第几圈，从 1 开始，满足你说的 (0,1)(0,2)...
#
#     while True:
#         # 看这一圈能不能确定半径
#         candidate = {}
#         for f in range(4):
#             start = ptr[f]
#             if start >= len(sector[f]):
#                 continue
#             end = min(start + PER, len(sector[f]))
#             candidate[f] = sector[f][end - 1][1]
#
#         if not candidate:
#             break
#
#         r_layer = min(candidate.values())
#
#         # 这一圈 4 个扇区都生成一次
#         for f in range(4):
#             key = (f, j)               # ★ 先扇区 f，再圈 j
#             picked = []
#
#             # 先拿本扇区半径 <= r_layer 的新点
#             while (
#                 ptr[f] < len(sector[f])
#                 and sector[f][ptr[f]][1] <= r_layer
#                 and len(picked) < PER
#             ):
#                 picked.append(sector[f][ptr[f]][0])
#                 ptr[f] += 1
#
#             # 如果这一圈这个扇区一个新的都没有，也要能填出来
#             if len(picked) == 0:
#                 if ptr[f] < len(sector[f]):
#                     base_id = sector[f][ptr[f]][0]
#                 elif ptr[f] > 0:
#                     base_id = sector[f][ptr[f] - 1][0]
#                 else:
#                     base_id = 0
#                 picked = [base_id]
#
#             # 不够 PER 的，从本扇区已经选到的里复制
#             while len(picked) < PER:
#                 picked.append(picked[-1])
#
#             pieces[key] = picked
#
#         j += 1
#
#         if all(ptr[f] >= len(sector[f]) for f in range(4)):
#             break
#
#     pieces_str = {str(k): [int(x) for x in v] for k, v in pieces.items()}
#     return pieces_str, O
def build_pieces_strict(centroids_raw: np.ndarray):
    import math
    # 1) 平均向量 O
    O = np.mean(centroids_raw, axis=0)

    # 2) 先算每个中心的角度 & 半径
    items = []
    for idx, c in enumerate(centroids_raw):
        cos_val = np.clip(
            np.dot(O, c) / ((np.linalg.norm(O) + 1e-12) * (np.linalg.norm(c) + 1e-12)),
            -1.0, 1.0
        )
        th = math.acos(cos_val)
        r = np.linalg.norm(c - O)
        items.append((idx, th, r))

    # 3) 按角度排，按排名切四份
    items.sort(key=lambda x: x[1])
    n = len(items)
    q1 = n // 4
    q2 = n // 2
    q3 = (3 * n) // 4

    # 分位角记录下来给查询用
    a1 = items[q1 - 1][1] if q1 > 0 else 0.0
    a2 = items[q2 - 1][1] if q2 > 0 else a1
    a3 = items[q3 - 1][1] if q3 > 0 else a2
    angle_bounds = [float(a1), float(a2), float(a3)]

    # 4) 切成4个扇区，并在扇区里按半径排
    sectors = {0: [], 1: [], 2: [], 3: []}
    for rank, (idx, th, r) in enumerate(items):
        if rank < q1:
            f = 0
        elif rank < q2:
            f = 1
        elif rank < q3:
            f = 2
        else:
            f = 3
        sectors[f].append((idx, r))

    for f in range(4):
        sectors[f].sort(key=lambda x: x[1])
        if not sectors[f]:
            # 极端兜底：保证每个扇区至少有一个点
            sectors[f].append((items[0][0], items[0][2]))

    PER = PER_PIECE  # 一般 = 4
    # 指针：指到“下次要看的那个元素” —— 只在真正 <= r_layer 的时候才前进
    ptr = {f: 0 for f in range(4)}

    pieces = {}
    layer = 1

    while True:
        # 这一圈各扇区“先看一眼”自己的最多4个，但先不改指针
        candidates_per_f = {}
        last_r_per_f = {}
        any_left = False

        for f in range(4):
            start = ptr[f]
            if start < len(sectors[f]):
                any_left = True
            end = min(start + PER, len(sectors[f]))
            cand = sectors[f][start:end]  # 只是看这一圈准备拿哪几个
            candidates_per_f[f] = cand
            if cand:
                last_r_per_f[f] = cand[-1][1]
            else:
                # 这个扇区已经没新点了，就别让它影响本圈半径
                # 用 +inf 表示这圈它不参与 r_layer 的竞争
                last_r_per_f[f] = float("inf")

        if not any_left:
            break  # 全部扇区都没数据了

        # 本圈公共半径 = 这次真的参与的扇区的“这4个里的最后一个半径”的最小值
        r_layer = min(last_r_per_f.values())

        # 为了“从邻居区域复制”时有东西可拿，先把这一圈所有候选收集起来
        # 注意这里收集的是“这一圈看到的所有中心的 id”，不论半径
        layer_all_ids = []
        for f in range(4):
            for cid, _r in candidates_per_f[f]:
                layer_all_ids.append(cid)
        if not layer_all_ids:
            layer_all_ids = [sectors[0][0][0]]

        # 真正生成这一圈的4个 piece
        for f in range(4):
            cand = candidates_per_f[f]  # 我这一圈看到的那些 (cid, r)
            piece_ids = []
            advance_count = 0  # 这一圈我真正消耗了多少个

            # ① 先把“本扇区这一圈里半径 <= r_layer 的”放进去，同时准备消耗它们
            for cid, r in cand:
                if r <= r_layer:
                    piece_ids.append(cid)
                    advance_count += 1

            # ② 不够 PER，就优先从“我这一圈看到的这几个 cand”里复制
            if len(piece_ids) < PER and cand:
                for cid, _r in cand:
                    if cid not in piece_ids:
                        piece_ids.append(cid)
                    if len(piece_ids) >= PER:
                        break

            # ③ 还不够，再从这一圈其他扇区看到的里复制
            if len(piece_ids) < PER and layer_all_ids:
                for cid in layer_all_ids:
                    if cid not in piece_ids:
                        piece_ids.append(cid)
                    if len(piece_ids) >= PER:
                        break

            # ④ 兜底
            while len(piece_ids) < PER:
                piece_ids.append(sectors[f][0][0])

            # 写入这一圈这个扇区的 piece
            pieces[(f, layer)] = piece_ids

            # ⑤ 最后再推进指针：只推进“真正 <= r_layer 的那些”
            ptr[f] += advance_count

        layer += 1

    pieces_str = {str(k): v for k, v in pieces.items()}
    return pieces_str, O, angle_bounds




def compute_safe_scale(centroids_unit, P, alpha=ALPHA):
    """
    计算安全的 SCALE 参数
    """
    SCALE = 10000  # 适当选择 SCALE
    print(f"[info] Using fixed SCALE={SCALE}")
    return SCALE


def choose_n_clusters_by_name(dataset_name: str) -> int:
    name = dataset_name.lower()
    if name.endswith("1k") or "_1k" in name:
        return 50
    if name.endswith("5k") or "_5k" in name:
        return 250
    if name.endswith("10k") or "_10k" in name:
        return 500
    return 50


def capacity_constrained_exact(vecs, base_assignments, base_centroids, K=20):
    N = vecs.shape[0]
    C = base_centroids.shape[0]
    clusters = {cid: [] for cid in range(C)}
    for i, cid in enumerate(base_assignments):
        clusters[int(cid)].append(i)

    need_more = []
    extra_pool = []
    for cid in range(C):
        if len(clusters[cid]) < K:
            need_more.append((cid, K - len(clusters[cid])))
        elif len(clusters[cid]) > K:
            extra_pool.extend(clusters[cid][K:])
            clusters[cid] = clusters[cid][:K]

    for cid, need in need_more:
        while need > 0 and extra_pool:
            pid = extra_pool.pop()
            clusters[cid].append(pid)
            need -= 1

    new_assign = np.full(N, -1, dtype=int)
    for cid, members in clusters.items():
        for i in members:
            new_assign[i] = cid
    for i in range(N):
        if new_assign[i] == -1:
            dists = np.linalg.norm(base_centroids - vecs[i], axis=1)
            new_assign[i] = int(np.argmin(dists))

    new_centroids = []
    for cid in range(C):
        members = clusters[cid]
        if members:
            new_centroids.append(vecs[members].mean(axis=0))
        else:
            new_centroids.append(base_centroids[cid])
    new_centroids = np.vstack(new_centroids)
    return new_assign, new_centroids


def build_index_for_dataset(dataset_name):
    vec_path = os.path.join(VEC_DIR, f"{dataset_name}_vecs.npy")
    if not os.path.exists(vec_path):
        print(f"[skip] vectors not found: {vec_path}")
        return None

    print(f"\n[build_index] dataset={dataset_name}")
    t = Timer(); t.start()
    t1 = time.time()
    vecs = np.load(vec_path)
    print(f" - docs={vecs.shape[0]} dim={vecs.shape[1]}")

    n_clusters = choose_n_clusters_by_name(dataset_name)
    print(f" - using n_clusters={n_clusters}")

    mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=RANDOM_SEED)
    t2 = time.time()
    print(f" - kmeans time: {t2 - t1:.2f} sec")
    base_assignments = mbk.fit_predict(vecs)
    base_centroids = mbk.cluster_centers_

    assignments, centroids = capacity_constrained_exact(
        vecs, base_assignments, base_centroids, K=MAX_PER_CLUSTER
    )

    # pieces_dict, O_center = build_pieces_strict(centroids)
    pieces_dict, O_center, angle_bounds = build_pieces_strict(centroids)
    centroids_unit = unitize(centroids)  # 单位化聚类中心
    SCALE = compute_safe_scale(centroids_unit, P, alpha=ALPHA)

    # 按 SCALE 量化并存储
    centroids_int = np.round(centroids_unit * SCALE).astype(np.int64)
    centroids_field = [[int(x) % P for x in row] for row in centroids_int]

    dataset_dir = os.path.join(OUT_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    np.save(os.path.join(dataset_dir, "centroids_raw.npy"), centroids.astype(np.float32))
    np.save(os.path.join(dataset_dir, "centroids_unit.npy"), centroids_unit.astype(np.float32))
    np.save(os.path.join(dataset_dir, "centroids_int.npy"), np.array(centroids_int, dtype=object), allow_pickle=True)
    np.save(os.path.join(dataset_dir, "centroids_field.npy"), np.array(centroids_field, dtype=object), allow_pickle=True)
    np.save(os.path.join(dataset_dir, "assignments.npy"), assignments)
    save_json(os.path.join(dataset_dir, "pieces.json"), pieces_dict)
    # save_json(os.path.join(dataset_dir, "center.json"), O_center.tolist())
    save_json(os.path.join(dataset_dir, "center.json"), {
        "O": O_center.tolist(),
        "angle_bounds": angle_bounds
    })
    save_json(os.path.join(dataset_dir, "index_meta.json"), {
        "P": int(P),
        "SCALE": int(SCALE),
        "N_CLUSTERS": int(centroids_unit.shape[0]),
        "T": T,
        "PER_PIECE": PER_PIECE,
    })

    # ===== AES key =====
    key_path = os.path.join(dataset_dir, "index_key.bin")
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            index_key = f.read()
    else:
        index_key = get_random_bytes(32)  # AES-256
        with open(key_path, "wb") as f:
            f.write(index_key)

    # ===== 构造 Th_all.pkl =====
    n_clusters_final = centroids_unit.shape[0]
    cluster_docs = {cid: [] for cid in range(n_clusters_final)}
    for i, cid in enumerate(assignments):
        cluster_docs[int(cid)].append(int(i))

    th_dict = {}
    for piece_str, cluster_id_list in pieces_dict.items():
        table_id = hmac.new(index_key, piece_str.encode("utf-8"), hashlib.sha256).hexdigest()
        bucket_clusters = {}
        bucket_docs_enc = {}
        for cid in cluster_id_list:
            cid_int = int(cid)
            bucket_clusters[str(cid_int)] = centroids_int[cid_int]
            docs = cluster_docs.get(cid_int, [])
            pt = json.dumps(docs).encode("utf-8")
            blob = aes_encrypt_aesgcm(pt, index_key)
            bucket_docs_enc[str(cid_int)] = blob.hex()

        th_dict[table_id] = {
            "piece": piece_str,
            "clusters": bucket_clusters,
            "docs_enc": bucket_docs_enc,
        }

    with open(os.path.join(dataset_dir, "Th_all.pkl"), "wb") as f:
        pickle.dump(th_dict, f)

    elapsed = t.stop()
    print(f"[done] dataset={dataset_name} pieces={len(pieces_dict)} time={elapsed:.2f}s")


if __name__ == "__main__":
    print("=== build_index: start ===")
    for f in sorted(os.listdir(VEC_DIR)):
        if f.endswith("_vecs.npy"):
            name = f.replace("_vecs.npy", "")
            build_index_for_dataset(name)
    print("=== build_index: done ===")


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# build_index.py
#
# 生成索引（仅针对 MS MARCO 向量）：
# 1. 聚类 → 容量约束 → 单位化中心
# 2. 切 piece
# 3. 把单位化中心按 SCALE 量化，写 centroids_int.npy / centroids_field.npy
# 4. 生成多重映射表 Th(k,l)：
#    - key = HMAC(index_key, piece_str)
#    - value 里放：这个 piece 里的所有 cluster 的量化中心 + 这个 cluster 下的文档 id 列表的 AES 密文
# 5. index_key.bin 里存 AES key (32 bytes)
# """
#
# import os
# import math
# import json
# import random
# import hmac
# import hashlib
# import pickle
# import numpy as np
# from sklearn.cluster import MiniBatchKMeans
# from Crypto.Cipher import AES
# from Crypto.Random import get_random_bytes
#
# # ----- 简单工具 -----
# def save_json(path, obj):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False, indent=2)
#
#
# class Timer:
#     def start(self):
#         import time
#         self._t = time.time()
#     def stop(self):
#         import time
#         return time.time() - self._t
#
# # ====== 配置 ======
#
# # 向量目录与索引目录：统一到 /root/siton-tmp/outputs 下
# VEC_DIR = "/root/siton-tmp/outputs/vectors"
# OUT_DIR = "/root/siton-tmp/outputs/index"
#
# # 只处理 MS MARCO 三个子集
# DATASETS = [
#     "msmarco_1k",
#     "msmarco_5k",
#     "msmarco_10k",
# ]
#
# T = 32
# P = 2 ** 61 - 1
# PER_PIECE = T // 8
# RANDOM_SEED = 42
# ALPHA = 0.85
# MAX_PER_CLUSTER = 20
#
# os.makedirs(OUT_DIR, exist_ok=True)
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
#
#
# def aes_encrypt_aesgcm(plaintext: bytes, key: bytes) -> bytes:
#     nonce = get_random_bytes(12)
#     cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
#     ciphertext, tag = cipher.encrypt_and_digest(plaintext)
#     return nonce + tag + ciphertext  # 12 + 16 + N
#
#
# def unitize(vectors: np.ndarray) -> np.ndarray:
#     """将向量归一化为单位向量"""
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
#     return vectors / norms
#
#
# def build_pieces_strict(centroids_raw: np.ndarray):
#     """
#     将聚类中心按角度和半径分成 4 个“扇区”，再按圈分 piece：
#     每个 piece 包含 PER_PIECE 个中心索引。
#     """
#     import math
#     # 1) 平均向量 O
#     O = np.mean(centroids_raw, axis=0)
#
#     # 2) 先算每个中心的角度 & 半径
#     items = []
#     for idx, c in enumerate(centroids_raw):
#         cos_val = np.clip(
#             np.dot(O, c) / ((np.linalg.norm(O) + 1e-12) * (np.linalg.norm(c) + 1e-12)),
#             -1.0, 1.0
#         )
#         th = math.acos(cos_val)
#         r = np.linalg.norm(c - O)
#         items.append((idx, th, r))
#
#     # 3) 按角度排，按排名切四份
#     items.sort(key=lambda x: x[1])
#     n = len(items)
#     q1 = n // 4
#     q2 = n // 2
#     q3 = (3 * n) // 4
#
#     a1 = items[q1 - 1][1] if q1 > 0 else 0.0
#     a2 = items[q2 - 1][1] if q2 > 0 else a1
#     a3 = items[q3 - 1][1] if q3 > 0 else a2
#     angle_bounds = [float(a1), float(a2), float(a3)]
#
#     # 4) 切成4个扇区，并在扇区里按半径排
#     sectors = {0: [], 1: [], 2: [], 3: []}
#     for rank, (idx, th, r) in enumerate(items):
#         if rank < q1:
#             f = 0
#         elif rank < q2:
#             f = 1
#         elif rank < q3:
#             f = 2
#         else:
#             f = 3
#         sectors[f].append((idx, r))
#
#     for f in range(4):
#         sectors[f].sort(key=lambda x: x[1])
#         if not sectors[f]:
#             # 兜底：保证每个扇区至少有一个点
#             sectors[f].append((items[0][0], items[0][2]))
#
#     PER = PER_PIECE
#     ptr = {f: 0 for f in range(4)}
#
#     pieces = {}
#     layer = 1
#
#     while True:
#         candidates_per_f = {}
#         last_r_per_f = {}
#         any_left = False
#
#         for f in range(4):
#             start = ptr[f]
#             if start < len(sectors[f]):
#                 any_left = True
#             end = min(start + PER, len(sectors[f]))
#             cand = sectors[f][start:end]
#             candidates_per_f[f] = cand
#             if cand:
#                 last_r_per_f[f] = cand[-1][1]
#             else:
#                 last_r_per_f[f] = float("inf")
#
#         if not any_left:
#             break
#
#         r_layer = min(last_r_per_f.values())
#
#         layer_all_ids = []
#         for f in range(4):
#             for cid, _r in candidates_per_f[f]:
#                 layer_all_ids.append(cid)
#         if not layer_all_ids:
#             layer_all_ids = [sectors[0][0][0]]
#
#         for f in range(4):
#             cand = candidates_per_f[f]
#             piece_ids = []
#             advance_count = 0
#
#             # ① 本扇区中 r <= r_layer 的点
#             for cid, r in cand:
#                 if r <= r_layer:
#                     piece_ids.append(cid)
#                     advance_count += 1
#
#             # ② 不够 PER 时，优先从 cand 复制
#             if len(piece_ids) < PER and cand:
#                 for cid, _r in cand:
#                     if cid not in piece_ids:
#                         piece_ids.append(cid)
#                     if len(piece_ids) >= PER:
#                         break
#
#             # ③ 还不够，从 layer_all_ids 复制
#             if len(piece_ids) < PER and layer_all_ids:
#                 for cid in layer_all_ids:
#                     if cid not in piece_ids:
#                         piece_ids.append(cid)
#                     if len(piece_ids) >= PER:
#                         break
#
#             # ④ 再兜底
#             while len(piece_ids) < PER:
#                 piece_ids.append(sectors[f][0][0])
#
#             pieces[(f, layer)] = piece_ids
#             ptr[f] += advance_count
#
#         layer += 1
#
#     pieces_str = {str(k): v for k, v in pieces.items()}
#     return pieces_str, O, angle_bounds
#
#
# def compute_safe_scale(centroids_unit, P, alpha=ALPHA):
#     """
#     计算安全的 SCALE 参数（这里直接固定为 10000）
#     """
#     SCALE = 10000
#     print(f"[info] Using fixed SCALE={SCALE}")
#     return SCALE
#
#
# def choose_n_clusters_by_name(dataset_name: str) -> int:
#     name = dataset_name.lower()
#     if name.endswith("1k") or "_1k" in name:
#         return 50
#     if name.endswith("5k") or "_5k" in name:
#         return 250
#     if name.endswith("10k") or "_10k" in name:
#         return 500
#     return 50
#
#
# def capacity_constrained_exact(vecs, base_assignments, base_centroids, K=20):
#     N = vecs.shape[0]
#     C = base_centroids.shape[0]
#     clusters = {cid: [] for cid in range(C)}
#     for i, cid in enumerate(base_assignments):
#         clusters[int(cid)].append(i)
#
#     need_more = []
#     extra_pool = []
#     for cid in range(C):
#         if len(clusters[cid]) < K:
#             need_more.append((cid, K - len(clusters[cid])))
#         elif len(clusters[cid]) > K:
#             extra_pool.extend(clusters[cid][K:])
#             clusters[cid] = clusters[cid][:K]
#
#     for cid, need in need_more:
#         while need > 0 and extra_pool:
#             pid = extra_pool.pop()
#             clusters[cid].append(pid)
#             need -= 1
#
#     new_assign = np.full(N, -1, dtype=int)
#     for cid, members in clusters.items():
#         for i in members:
#             new_assign[i] = cid
#     for i in range(N):
#         if new_assign[i] == -1:
#             dists = np.linalg.norm(base_centroids - vecs[i], axis=1)
#             new_assign[i] = int(np.argmin(dists))
#
#     new_centroids = []
#     for cid in range(C):
#         members = clusters[cid]
#         if members:
#             new_centroids.append(vecs[members].mean(axis=0))
#         else:
#             new_centroids.append(base_centroids[cid])
#     new_centroids = np.vstack(new_centroids)
#     return new_assign, new_centroids
#
#
# def build_index_for_dataset(dataset_name):
#     vec_path = os.path.join(VEC_DIR, f"{dataset_name}_vecs.npy")
#     if not os.path.exists(vec_path):
#         print(f"[skip] vectors not found for {dataset_name}: {vec_path}")
#         return None
#
#     print(f"\n[build_index] dataset={dataset_name}")
#     t = Timer(); t.start()
#
#     vecs = np.load(vec_path)
#     print(f" - docs={vecs.shape[0]} dim={vecs.shape[1]}")
#
#     n_clusters = choose_n_clusters_by_name(dataset_name)
#     print(f" - using n_clusters={n_clusters}")
#
#     mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=RANDOM_SEED)
#     base_assignments = mbk.fit_predict(vecs)
#     base_centroids = mbk.cluster_centers_
#
#     assignments, centroids = capacity_constrained_exact(
#         vecs, base_assignments, base_centroids, K=MAX_PER_CLUSTER
#     )
#
#     pieces_dict, O_center, angle_bounds = build_pieces_strict(centroids)
#     centroids_unit = unitize(centroids)
#     SCALE = compute_safe_scale(centroids_unit, P, alpha=ALPHA)
#
#     centroids_int = np.round(centroids_unit * SCALE).astype(np.int64)
#     centroids_field = [[int(x) % P for x in row] for row in centroids_int]
#
#     dataset_dir = os.path.join(OUT_DIR, dataset_name)
#     os.makedirs(dataset_dir, exist_ok=True)
#
#     np.save(os.path.join(dataset_dir, "centroids_raw.npy"), centroids.astype(np.float32))
#     np.save(os.path.join(dataset_dir, "centroids_unit.npy"), centroids_unit.astype(np.float32))
#     np.save(os.path.join(dataset_dir, "centroids_int.npy"), np.array(centroids_int, dtype=object), allow_pickle=True)
#     np.save(os.path.join(dataset_dir, "centroids_field.npy"), np.array(centroids_field, dtype=object), allow_pickle=True)
#     np.save(os.path.join(dataset_dir, "assignments.npy"), assignments)
#     save_json(os.path.join(dataset_dir, "pieces.json"), pieces_dict)
#     save_json(os.path.join(dataset_dir, "center.json"), {
#         "O": O_center.tolist(),
#         "angle_bounds": angle_bounds
#     })
#     save_json(os.path.join(dataset_dir, "index_meta.json"), {
#         "P": int(P),
#         "SCALE": int(SCALE),
#         "N_CLUSTERS": int(centroids_unit.shape[0]),
#         "T": T,
#         "PER_PIECE": PER_PIECE,
#     })
#
#     # ===== AES key =====
#     key_path = os.path.join(dataset_dir, "index_key.bin")
#     if os.path.exists(key_path):
#         with open(key_path, "rb") as f:
#             index_key = f.read()
#     else:
#         index_key = get_random_bytes(32)  # AES-256
#         with open(key_path, "wb") as f:
#             f.write(index_key)
#
#     # ===== 构造 Th_all.pkl =====
#     n_clusters_final = centroids_unit.shape[0]
#     cluster_docs = {cid: [] for cid in range(n_clusters_final)}
#     for i, cid in enumerate(assignments):
#         cluster_docs[int(cid)].append(int(i))
#
#     th_dict = {}
#     for piece_str, cluster_id_list in pieces_dict.items():
#         table_id = hmac.new(index_key, piece_str.encode("utf-8"), hashlib.sha256).hexdigest()
#         bucket_clusters = {}
#         bucket_docs_enc = {}
#         for cid in cluster_id_list:
#             cid_int = int(cid)
#             bucket_clusters[str(cid_int)] = centroids_int[cid_int]
#             docs = cluster_docs.get(cid_int, [])
#             pt = json.dumps(docs).encode("utf-8")
#             blob = aes_encrypt_aesgcm(pt, index_key)
#             bucket_docs_enc[str(cid_int)] = blob.hex()
#
#         th_dict[table_id] = {
#             "piece": piece_str,
#             "clusters": bucket_clusters,
#             "docs_enc": bucket_docs_enc,
#         }
#
#     with open(os.path.join(dataset_dir, "Th_all.pkl"), "wb") as f:
#         pickle.dump(th_dict, f)
#
#     elapsed = t.stop()
#     print(f"[done] dataset={dataset_name} pieces={len(pieces_dict)} time={elapsed:.2f}s")
#
#
# if __name__ == "__main__":
#     print("=== build_index: start ===")
#     for name in DATASETS:
#         build_index_for_dataset(name)
#     print("=== build_index: done ===")
