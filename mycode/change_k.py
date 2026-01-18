import os
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
OUT_DIR = "../outputs/index_k"   # 输出目录

T = 32
P = 2 ** 61 - 1
PER_PIECE = T // 8
RANDOM_SEED = 42
ALPHA = 0.85

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


def angle(center, vec):
    cos_val = np.clip(
        np.dot(center, vec)
        / ((np.linalg.norm(center) + 1e-12) * (np.linalg.norm(vec) + 1e-12)),
        -1.0, 1.0
    )
    return math.acos(cos_val)


def build_pieces_strict(centroids_raw: np.ndarray):
    import math
    # 1) 平均向量 O
    O = np.mean(centroids_raw, axis=0)

    # 2) 先算每个中心的角度 & 半径
    items = []
    for idx, c in enumerate(centroids_raw):
        cos_val = np.clip(
            np.dot(O, c)
            / ((np.linalg.norm(O) + 1e-12) * (np.linalg.norm(c) + 1e-12)),
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
    ptr = {f: 0 for f in range(4)}  # 指针：指到“下次要看的那个元素”

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
            cand = sectors[f][start:end]
            candidates_per_f[f] = cand
            if cand:
                last_r_per_f[f] = cand[-1][1]
            else:
                last_r_per_f[f] = float("inf")

        if not any_left:
            break  # 全部扇区都没数据了

        # 本圈公共半径
        r_layer = min(last_r_per_f.values())

        # 为了从邻居区域复制，先把这一圈所有候选收集起来
        layer_all_ids = []
        for f in range(4):
            for cid, _r in candidates_per_f[f]:
                layer_all_ids.append(cid)
        if not layer_all_ids:
            layer_all_ids = [sectors[0][0][0]]

        # 真正生成这一圈的4个 piece
        for f in range(4):
            cand = candidates_per_f[f]
            piece_ids = []
            advance_count = 0

            # ① 本扇区半径 <= r_layer 的
            for cid, r in cand:
                if r <= r_layer:
                    piece_ids.append(cid)
                    advance_count += 1

            # ② 不够 PER，优先从 cand 里复制
            if len(piece_ids) < PER and cand:
                for cid, _r in cand:
                    if cid not in piece_ids:
                        piece_ids.append(cid)
                    if len(piece_ids) >= PER:
                        break

            # ③ 再从这一圈其他扇区看到的里复制
            if len(piece_ids) < PER and layer_all_ids:
                for cid in layer_all_ids:
                    if cid not in piece_ids:
                        piece_ids.append(cid)
                    if len(piece_ids) >= PER:
                        break

            # ④ 兜底
            while len(piece_ids) < PER:
                piece_ids.append(sectors[f][0][0])

            pieces[(f, layer)] = piece_ids
            ptr[f] += advance_count

        layer += 1

    pieces_str = {str(k): v for k, v in pieces.items()}
    return pieces_str, O, angle_bounds


def compute_safe_scale(centroids_unit, P, alpha=ALPHA):
    """
    计算安全的 SCALE 参数
    """
    SCALE = 10000
    print(f"[info] Using fixed SCALE={SCALE}")
    return SCALE


def capacity_constrained_exact(vecs, base_assignments, base_centroids, K=20):
    """
    K: 每个簇的目标容量（这里用来承载 k=10,20,40）

    保留原有逻辑：先裁到 <=K，再用 extra_pool 填充到 >=K，
    最后对剩余未赋值样本用最近质心补 assign（但这些不会影响下面我们
    用 K 限制 cluster_docs 的逻辑）。
    """
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


def build_index_for_dataset(dataset_name, K):
    """
    只针对指定 dataset_name 构建索引，并使用 K 控制簇数：
    C ≈ N / K，同时用 capacity_constrained_exact 修整。
    输出目录：../outputs/index_k/k{K}/{dataset_name}
    """
    vec_path = os.path.join(VEC_DIR, f"{dataset_name}_vecs.npy")
    if not os.path.exists(vec_path):
        print(f"[skip] vectors not found: {vec_path}")
        return None

    print(f"\n[build_index] dataset={dataset_name}, K={K}")
    t = Timer(); t.start()

    vecs = np.load(vec_path)
    N, dim = vecs.shape
    print(f" - docs={N} dim={dim}")

    # ★ 关键：由 K 控制簇数 C ≈ N / K
    n_clusters = max(1, int(round(N / float(K))))
    print(f" - using n_clusters={n_clusters} (≈ N/K)")

    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1024,
        random_state=RANDOM_SEED
    )
    base_assignments = mbk.fit_predict(vecs)
    base_centroids = mbk.cluster_centers_

    # 保留 capacity_constrained_exact
    assignments, centroids = capacity_constrained_exact(
        vecs, base_assignments, base_centroids, K=K
    )

    # 分 piece
    pieces_dict, O_center, angle_bounds = build_pieces_strict(centroids)
    centroids_unit = unitize(centroids)
    SCALE = compute_safe_scale(centroids_unit, P, alpha=ALPHA)

    # 量化
    centroids_int = np.round(centroids_unit * SCALE).astype(np.int64)
    centroids_field = [[int(x) % P for x in row] for row in centroids_int]

    dataset_dir = os.path.join(OUT_DIR, f"k{K}", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    np.save(os.path.join(dataset_dir, "centroids_raw.npy"), centroids.astype(np.float32))
    np.save(os.path.join(dataset_dir, "centroids_unit.npy"), centroids_unit.astype(np.float32))
    np.save(os.path.join(dataset_dir, "centroids_int.npy"), np.array(centroids_int, dtype=object), allow_pickle=True)
    np.save(os.path.join(dataset_dir, "centroids_field.npy"), np.array(centroids_field, dtype=object), allow_pickle=True)
    np.save(os.path.join(dataset_dir, "assignments.npy"), assignments)
    save_json(os.path.join(dataset_dir, "pieces.json"), pieces_dict)
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
        "K": int(K),
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
    # 这里我们真正控制“每个簇 K 条 doc”，用于写入索引
    cluster_docs = {cid: [] for cid in range(n_clusters_final)}

    # 先按 assignments 顺序填，每个簇最多 K 条
    for i, cid in enumerate(assignments):
        cid = int(cid)
        if cid < 0 or cid >= n_clusters_final:
            continue
        if len(cluster_docs[cid]) < K:
            cluster_docs[cid].append(int(i))

    # 再把不足 K 的簇用自身文档重复填满；极端情况再用 0 兜底
    for cid in range(n_clusters_final):
        if len(cluster_docs[cid]) == 0:
            # 没有文档就兜底放一个 0（或随机 doc）
            cluster_docs[cid].append(0)
        while len(cluster_docs[cid]) < K:
            cluster_docs[cid].append(cluster_docs[cid][-1])

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
    print(
        f"[done] dataset={dataset_name}, K={K}, "
        f"clusters={n_clusters_final}, pieces={len(pieces_dict)} time={elapsed:.2f}s"
    )


if __name__ == "__main__":
    print("=== build_index: start ===")

    dataset_name = "enron_5k"

    # 分别用 k=10,20,40 构建三套索引
    for K in [10, 20, 40]:
        build_index_for_dataset(dataset_name, K)

    print("=== build_index: done ===")
