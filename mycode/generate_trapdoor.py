"""
generate_trapdoor.py
- 编码查询
- 找到查询所在的 piece (f, j)
- 选定一组要放进 BloomFilter 的 piece 标签
"""

import os
import json
import math
import hashlib
import hmac
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pybloom_live import BloomFilter

INDEX_DIR = "../outputs/index"
MODEL_DIR = "/root/siton-tmp/all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_DIR):
            raise RuntimeError(f"本地模型不存在: {MODEL_DIR}")
        _model = SentenceTransformer(MODEL_DIR)
    return _model

# def load_index_meta(dataset):
#     dataset_dir = os.path.join(INDEX_DIR, dataset)
#     with open(os.path.join(dataset_dir, "index_meta.json"), "r", encoding="utf-8") as f:
#         meta = json.load(f)
#
#     center = None
#     with open(os.path.join(dataset_dir, "center.json"), "r", encoding="utf-8") as f:
#         center = np.array(json.load(f))
#
#     with open(os.path.join(dataset_dir, "pieces.json"), "r", encoding="utf-8") as f:
#         pieces = json.load(f)
#
#     # 新增：把原始聚类中心读进来
#     centroids_raw = np.load(os.path.join(dataset_dir, "centroids_raw.npy"))
#
#     # index_key 还是照旧
#     index_key = None
#     key_path = os.path.join(dataset_dir, "index_key.bin")
#     if os.path.exists(key_path):
#         with open(key_path, "rb") as f:
#             index_key = f.read()
#
#     return meta, int(meta["P"]), int(meta["SCALE"]), center, pieces, centroids_raw, index_key
def load_index_meta(dataset):
    dataset_dir = os.path.join(INDEX_DIR, dataset)

    # 1) index_meta.json
    with open(os.path.join(dataset_dir, "index_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 2) center.json
    with open(os.path.join(dataset_dir, "center.json"), "r", encoding="utf-8") as f:
        center_obj = json.load(f)
        if isinstance(center_obj, list):
            center = np.array(center_obj)
            angle_bounds = None
        else:
            center = np.array(center_obj["O"])
            angle_bounds = center_obj.get("angle_bounds", None)

    # 3) pieces.json
    with open(os.path.join(dataset_dir, "pieces.json"), "r", encoding="utf-8") as f:
        pieces = json.load(f)

    # 4) 原始聚类中心
    centroids_raw = np.load(os.path.join(dataset_dir, "centroids_raw.npy"))

    # 5) index_key
    key_path = os.path.join(dataset_dir, "index_key.bin")
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            index_key = f.read()
    else:
        index_key = None

    return (
        meta,
        int(meta["P"]),
        int(meta["SCALE"]),
        center,
        pieces,
        centroids_raw,
        index_key,
        angle_bounds,
    )



def encode_query_raw(query: str) -> np.ndarray:
    """返回未归一化的查询向量"""
    model = get_model()
    v = model.encode([query], convert_to_numpy=True)[0]
    return v


# def find_piece_with_raw_q(q_raw: np.ndarray,
#                           O_center: np.ndarray,
#                           pieces: dict,
#                           centroids_raw: np.ndarray,
#                           per_piece: int):
#     # 1) 先按角度分 f
#     cos_val = np.clip(
#         np.dot(q_raw, O_center)
#         / ((np.linalg.norm(q_raw) + 1e-12) * (np.linalg.norm(O_center) + 1e-12)),
#         -1.0, 1.0
#     )
#     theta = math.acos(cos_val)
#     if 0 < theta <= math.pi / 4:
#         f = 0
#     elif math.pi / 4 < theta <= math.pi / 2:
#         f = 1
#     elif math.pi / 2 < theta <= 3 * math.pi / 4:
#         f = 2
#     else:
#         f = 3
#
#     # 2) 把这个 f 下面当初切好的所有 centroid 拿出来
#     f_indices = []
#     for k_str, idx_list in pieces.items():
#         k = eval(k_str)        # k 是 (f, j)
#         if k[0] == f:
#             f_indices.extend(int(x) for x in idx_list)
#
#     if not f_indices:
#         return f, 1
#
#     # 3) 用“原始 centroid”跟“原始 O”算半径，并按半径排
#     dists = []
#     for cid in f_indices:
#         ci = centroids_raw[cid]               # ✅ 用原始的，不是 unit
#         ri = float(np.linalg.norm(ci - O_center))
#         dists.append((cid, ri))
#     dists.sort(key=lambda x: x[1])
#     dist_vals = [r for (_cid, r) in dists]
#
#     # 4) q 自己的半径
#     r_q = float(np.linalg.norm(q_raw - O_center))
#
#     # 5) 找到 q 插入的位置
#     pos = 0
#     while pos < len(dist_vals) and dist_vals[pos] <= r_q:
#         pos += 1
#
#     j = (pos // per_piece) + 1
#
#     # 6) 把 j 限制在这个 f 实际存在的范围里
#     max_j = 1
#     for k_str in pieces.keys():
#         k = eval(k_str)
#         if k[0] == f and k[1] > max_j:
#             max_j = k[1]
#     if j > max_j:
#         j = max_j
#     if j < 1:
#         j = 1
#
#     return f, j
def find_piece_with_raw_q(q_raw: np.ndarray,
                          O_center: np.ndarray,
                          pieces: dict,
                          centroids_raw: np.ndarray,
                          per_piece: int,
                          angle_bounds=None):
    # 1) q 的角度
    cos_val = np.clip(
        np.dot(q_raw, O_center)
        / ((np.linalg.norm(q_raw) + 1e-12) * (np.linalg.norm(O_center) + 1e-12)),
        -1.0, 1.0
    )
    theta = math.acos(cos_val)

    # 2) 用构建时的分位角度来决定 f
    if angle_bounds is not None:
        a1, a2, a3 = angle_bounds
        if theta <= a1:
            f = 0
        elif theta <= a2:
            f = 1
        elif theta <= a3:
            f = 2
        else:
            f = 3
    else:
        if 0 < theta <= math.pi / 4:
            f = 0
        elif math.pi / 4 < theta <= math.pi / 2:
            f = 1
        elif math.pi / 2 < theta <= 3 * math.pi / 4:
            f = 2
        else:
            f = 3

    # 3) 在这个 f 里按半径排，然后用 q 的半径定位 j
    #    注意这里要用真正的 centroids_raw 来排，别用 pieces 里的，因为 pieces 里有复制的
    f_centroids = []
    for cid, c in enumerate(centroids_raw):
        # 先算这个中心的角度，按同一套 angle_bounds 决定它属于哪个 f
        cos_c = np.clip(
            np.dot(c, O_center)
            / ((np.linalg.norm(c) + 1e-12) * (np.linalg.norm(O_center) + 1e-12)),
            -1.0, 1.0
        )
        th_c = math.acos(cos_c)

        if angle_bounds is not None:
            if th_c <= a1:
                f_c = 0
            elif th_c <= a2:
                f_c = 1
            elif th_c <= a3:
                f_c = 2
            else:
                f_c = 3
        else:
            if 0 < th_c <= math.pi / 4:
                f_c = 0
            elif math.pi / 4 < th_c <= math.pi / 2:
                f_c = 1
            elif math.pi / 2 < th_c <= 3 * math.pi / 4:
                f_c = 2
            else:
                f_c = 3

        if f_c == f:
            r_c = float(np.linalg.norm(c - O_center))
            f_centroids.append((cid, r_c))

    if not f_centroids:
        return f, 1

    f_centroids.sort(key=lambda x: x[1])
    radii_sorted = [r for (_cid, r) in f_centroids]

    # q 的半径
    r_q = float(np.linalg.norm(q_raw - O_center))
    pos = 0
    while pos < len(radii_sorted) and radii_sorted[pos] <= r_q:
        pos += 1

    j = (pos // per_piece) + 1

    # 4) j 限制在这个 f 真正存在的范围里（pieces 里可能只有几圈）
    max_j = 1
    for k_str in pieces.keys():
        k = eval(k_str)
        if k[0] == f and k[1] > max_j:
            max_j = k[1]
    if j > max_j:
        j = max_j
    if j < 1:
        j = 1

    return f, j


# def generate_trapdoor(query, dataset, bf_capacity=64, bf_error=0.1, deterministic_seed_for_share=True):
#     meta, P, SCALE, center, pieces, centroids_raw, index_key = load_index_meta(dataset)
#
#     per_piece = int(meta.get("PER_PIECE", max(1, meta.get("T", 32) // 8)))
#
#     # ① 得到“原始 q”
#     model = get_model()
#     q_raw = model.encode([query], convert_to_numpy=True)[0]
#
#     # ② 用原始 q 找 (f, j)
#     f, j = find_piece_with_raw_q(q_raw, center, pieces, centroids_raw, per_piece)
#     print("(f,j):", f, j)

def generate_trapdoor(query, dataset, bf_capacity=64, bf_error=0.1, deterministic_seed_for_share=True):
    meta, P, SCALE, center, pieces, centroids_raw, index_key, angle_bounds = load_index_meta(dataset)

    per_piece = int(meta.get("PER_PIECE", max(1, meta.get("T", 32) // 8)))

    model = get_model()
    q_raw = model.encode([query], convert_to_numpy=True)[0]

    # 用新的查找函数
    f, j = find_piece_with_raw_q(q_raw, center, pieces, centroids_raw, per_piece, angle_bounds)
    print("(f,j):", f, j)

    # ③ 把 f-1, f, f+1 的标签塞进 Bloom
    def wrap_f(x): return x % 4

    # # 先把要的坐标都列出来
    # neighbor_coords = set()
    # # 当前这一片
    # neighbor_coords.add((f, j))
    # # 同一行的 j±1, j±2
    # for dj in (-2, -1, 1, 2):
    #     neighbor_coords.add((f, j + dj))
    #
    # # 上一行、下一行的 5 个
    # for df in (-1, 1):
    #     ff = wrap_f(f + df)
    #     # (f±1, j)
    #     neighbor_coords.add((ff, j))
    #     # (f±1, j±1, j±2)
    #     for dj in (-2, -1, 1, 2):
    #         neighbor_coords.add((ff, j + dj))
    #
    # # ③.1 真正塞进 BF 的时候，用现有的 pieces 过滤一遍
    # would_insert = []
    # for key_str in pieces.keys():
    #     # key_str 形如 "(1, 11)"
    #     k = eval(key_str)  # -> (f, j)
    #     if k in neighbor_coords:
    #         would_insert.append(key_str)

    stable_f = {wrap_f(f + df) for df in (-1, 0, 1)}
    # stable_f = {wrap_f(f + df) for df in (-1, 0)}
    # stable_f = {wrap_f(f)}

    # ③.1 先数一下真正要塞多少条
    would_insert = []
    for key_str in pieces.keys():
        k = eval(key_str)          # k 是 (f, j)
        if k[0] in stable_f:
            would_insert.append(key_str)

    # ③.2 根据要塞的条数动态放大容量，保底还是用传进来的 bf_capacity
    needed_capacity = len(would_insert) + 16   # 留一点余量，避免刚好顶满
    real_capacity = max(bf_capacity, needed_capacity)

    bf = BloomFilter(capacity=real_capacity, error_rate=bf_error)

    # ③.3 真正往里塞
    for key_str in would_insert:
        k = eval(key_str)
        l_str = str(k)
        if index_key is not None:
            h = hmac.new(index_key, l_str.encode("utf-8"), hashlib.sha256).hexdigest()
            bf.add(h)
        else:
            bf.add(l_str)

    bf_bytes = pickle.dumps(bf)

    # ④ 到这一步才归一化 + 量化 + 拆分
    q_unit = q_raw / (np.linalg.norm(q_raw) + 1e-12)
    q_scaled = np.round(q_unit * SCALE).astype(np.int64)
    q_scaled = q_scaled % P  # 先把真正的向量放进域里

    if deterministic_seed_for_share:
        seed = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # ✅ 在域里做 secret sharing
    share1 = rng.integers(low=0, high=P, size=q_scaled.shape, dtype=np.int64)
    share2 = (q_scaled - share1) % P

    t1 = {"bloom": bf_bytes, "q_share": [int(x) for x in share1.tolist()]}
    t2 = {"bloom": bf_bytes, "q_share": [int(x) for x in share2.tolist()]}

    return t1, t2, (f, j)



if __name__ == "__main__":
    t1, t2, qp = generate_trapdoor("Tell me about energy trading", "enron_1k")
    print("q_piece =", qp)
    print("share1 head:", t1["q_share"][:6])









