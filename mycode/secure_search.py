import os
import json
import pickle
import time
import hmac
import hashlib
import numpy as np
from pybloom_live import BloomFilter
from Crypto.Cipher import AES
import torch
from protocols_stub import (
    beaver_multiply,
    rabbit_argmax_two_shares,
    rabbit_argmax
)

BASE_DIR = "/root/siton-tmp"
INDEX_DIR = os.path.join(BASE_DIR, "outputs", "index")
META_DIR = os.path.join(BASE_DIR, "outputs", "meta")


def _load_index_meta(dataset: str):
    ddir = os.path.join(INDEX_DIR, dataset)
    with open(os.path.join(ddir, "index_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(ddir, "pieces.json"), "r", encoding="utf-8") as f:
        pieces = json.load(f)
    P = int(meta["P"])
    SCALE = int(meta["SCALE"])
    return pieces, P, SCALE


def _load_index_key(dataset: str):
    ddir = os.path.join(INDEX_DIR, dataset)
    key_path = os.path.join(ddir, "index_key.bin")
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return f.read()
    return None


def _to_piece_index(x):
    #找j
    if isinstance(x, int):
        return x
    if isinstance(x, (list, tuple)):
        return int(x[1])
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            return int(s)
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
            parts = [p.strip() for p in s.split(",")]
            return int(parts[1])
        return int(s)
    raise ValueError(f"Cannot parse piece index from {x!r}")

def _parse_piece_label(x):
    """把 piece 标识统一成 (f, j)"""
    if isinstance(x, tuple) or isinstance(x, list):
        if len(x) == 2:
            return int(x[0]), int(x[1])
        else:
            return 0, int(x[0])
    if isinstance(x, int):
        return 0, x
    s = str(x).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        a, b = [p.strip() for p in s.split(",")]
        return int(a), int(b)
    # 回退：老格式只有 j
    return 0, int(s)

def secure_dot_two_party(q1: np.ndarray,
                         q2: np.ndarray,
                         c1: np.ndarray,
                         c2: np.ndarray,
                         P: int):
    # 全部进大整数域
    q1 = np.array(q1, dtype=object) % P
    q2 = np.array(q2, dtype=object) % P
    c1 = np.array(c1, dtype=object) % P
    c2 = np.array(c2, dtype=object) % P

    # 四次 Beaver 乘法
    p11 = beaver_multiply(q1, c1, P=P)
    p12 = beaver_multiply(q1, c2, P=P)
    p21 = beaver_multiply(q2, c1, P=P)
    p22 = beaver_multiply(q2, c2, P=P)

    # 汇总成一条向量
    total_vec = (p11 + p12 + p21 + p22) % P

    # 求点积
    dot_val = int(sum(total_vec) % P)

    # 随机拆成两份
    rnd = np.random.randint(0, P, dtype=np.int64) % P
    sA = int(rnd)
    sB = int((dot_val - rnd) % P)
    return sA, sB




# ================= AES 相关开始 =================

def aes_decrypt_aesgcm(blob: bytes, key: bytes, aad: bytes | None = None) -> bytes:
    if len(blob) < 28:
        raise ValueError(f"AES-GCM blob too short: {len(blob)}")

    # 方案1: nonce | tag | ct
    try:
        nonce = blob[:12]
        tag = blob[12:28]
        ct = blob[28:]
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        if aad:
            cipher.update(aad)
        return cipher.decrypt_and_verify(ct, tag)
    except Exception as e1:
        # 方案2: nonce | ct | tag
        try:
            nonce = blob[:12]
            tag = blob[-16:]
            ct = blob[12:-16]
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            if aad:
                cipher.update(aad)
            return cipher.decrypt_and_verify(ct, tag)
        except Exception as e2:
            head = blob[:32].hex()
            raise ValueError(
                f"AES-GCM decrypt failed. blob_len={len(blob)}, head32={head}, "
                f"err1={e1}, err2={e2}"
            )


def merge_and_decrypt_docs(dataset: str,
                           enc_part_a_hex: str,
                           enc_part_b_hex: str):
    """
    合并 server_a / server_b 返回的两半密文并解密。
    """
    ddir = os.path.join(INDEX_DIR, dataset)
    key_path = os.path.join(ddir, "index_key.bin")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"{key_path} 不存在")
    with open(key_path, "rb") as f:
        key = f.read()

    # 为了猜 AAD，要读 pieces
    with open(os.path.join(ddir, "pieces.json"), "r", encoding="utf-8") as f:
        pieces = json.load(f)

    part_a = bytes.fromhex(enc_part_a_hex or "")
    part_b = bytes.fromhex(enc_part_b_hex or "")
    candidate_blobs = [
        part_a + part_b,
        part_b + part_a,
    ]

    # 所有可能的 AAD：None / piece_label / hmac(piece_label)
    aad_candidates: list[bytes | None] = [None]
    for piece_str in pieces.keys():
        aad_candidates.append(piece_str.encode("utf-8"))
        hval = hmac.new(key, piece_str.encode("utf-8"), hashlib.sha256).hexdigest()
        aad_candidates.append(hval.encode("utf-8"))

    last_err = None
    tried = 0
    for blob in candidate_blobs:
        for aad in aad_candidates:
            tried += 1
            try:
                # pt = aes_decrypt_aesgcm(blob, key, aad=aad)
                # return json.loads(pt.decode("utf-8"))
                pt = aes_decrypt_aesgcm(blob, key, aad=aad)
                ids_as_index = json.loads(pt.decode("utf-8"))  # 现在是 [180, 224, ...]
                # 把索引号变成真正的 id
                meta_path = os.path.join(META_DIR, f"{dataset}_meta.json")
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                real_ids = []
                for i in ids_as_index:
                    if 0 <= i < len(meta):
                        rid = str(meta[i].get("id", "")).strip()
                        # enron 有 '36.' 这种的话可以顺手去个点
                        real_ids.append(rid)
                return real_ids
            except Exception as e:
                last_err = e
                continue

    raise ValueError(
        f"AES-GCM 解密失败：尝试了 {tried} 种 blob/aad 组合仍然不行。"
        f"最后错误: {last_err}"
    )

# ================= AES 相关结束 =================


def secure_search(t1: dict, t2: dict, dataset: str, q_piece=None, debug: bool = False):
    # 加载 meta 信息和 index key
    pieces, P, SCALE = _load_index_meta(dataset)
    index_key = _load_index_key(dataset)

    # 加载 centroids 的两部分
    ddir = os.path.join(INDEX_DIR, dataset)
    cA_path = os.path.join(ddir, "centroids_share_a.npy")
    cB_path = os.path.join(ddir, "centroids_share_b.npy")
    if not os.path.exists(cA_path) or not os.path.exists(cB_path):
        raise FileNotFoundError("请先运行 preprocess_index_shares.py")

    # 读取已量化的 centroids 的两部分
    centroids_a = np.load(cA_path, allow_pickle=True)
    centroids_b = np.load(cB_path, allow_pickle=True)
    # # 打印 centroids 的形状和部分数据，确认是否正常
    # print("centroids_a shape:", centroids_a.shape)
    # print("centroids_b shape:", centroids_b.shape)
    #
    # # 打印前几个聚类中心数据，检查是否一致
    # print("centroids_a sample:", centroids_a[:3])  # 打印 centroids_a 的前3个聚类中心
    # print("centroids_b sample:", centroids_b[:3])  # 打印 centroids_b 的前3个聚类中心

    # 获取查询向量的两部分
    q1 = np.array(t1["q_share"], dtype=np.int64)
    q2 = np.array(t2["q_share"], dtype=np.int64)

    bf: BloomFilter = pickle.loads(t1["bloom"])

    # 用 h(k,l) 匹配 BF，找到相关的 pieces
    cand_ids = []
    for piece_label, cluster_id_list in pieces.items():
        piece_str = str(piece_label)
        if index_key is not None:
            ph = hmac.new(index_key, piece_str.encode("utf-8"), hashlib.sha256).hexdigest()
        else:
            ph = piece_str
        if ph in bf:
            cand_ids.extend(int(cid) for cid in cluster_id_list)

    cand_ids = sorted(set(cand_ids))
    print(cand_ids)

    if not cand_ids:
        print("[DEBUG] BloomFilter failed — fallback to all clusters")
        cand_ids = list(range(len(centroids_a)))  # Fallback to all clusters if no match

    # 初始化两个列表来存储计算结果
    scores_a = []
    scores_b = []

    # 对每个候选聚类，计算安全点积
    for cid in cand_ids:
        c1 = centroids_a[cid]
        c2 = centroids_b[cid]
        sA, sB = secure_dot_two_party(q1, q2, c1, c2, P)
        scores_a.append(sA)
        scores_b.append(sB)
    scores = [x + y for x, y in zip(scores_a, scores_b)]



    # 调用 rabbit 协议找到最相关的聚类
    best_local_idx, best_val = rabbit_argmax(scores)
    best_cluster = int(cand_ids[best_local_idx])
    print(best_cluster)

    # 找到最接近的 piece
    candidate_piece_keys = [k for k, v in pieces.items() if best_cluster in v]
    if candidate_piece_keys:
        if q_piece is not None:
            q_f, q_j = _parse_piece_label(q_piece)
            def piece_sort_key(k):
                k_f, k_j = _parse_piece_label(k)
                # 第一关键字：f 是否相同（相同的排前面 -> 用 0 / 1 表示）
                same_f = 0 if k_f == q_f else 1
                # 第二关键字：j 距离
                j_dist = abs(k_j - q_j)
                return (same_f, j_dist)
            best_piece = min(candidate_piece_keys, key=piece_sort_key)
        else:
            # 没有 q_piece，只能随便拿一个
            best_piece = candidate_piece_keys[0]
    else:
        print(f"[DEBUG] No piece contains cluster {best_cluster}")
        best_piece = None

    # 从 server_a / server_b 的 Th.pkl 中取文档密文的一半
    table_ids = t1.get("table_ids") or []
    main_table_id = t1.get("table_id")
    if main_table_id and main_table_id not in table_ids:
        table_ids = [main_table_id] + table_ids
    if not table_ids and best_piece is not None and index_key is not None:
        hid = hmac.new(index_key, str(best_piece).encode("utf-8"), hashlib.sha256).hexdigest()
        table_ids = [hid]

    th_a_path = os.path.join(ddir, "server_a", "Th.pkl")
    th_b_path = os.path.join(ddir, "server_b", "Th.pkl")

    docs_part_a_hex = None
    docs_part_b_hex = None

    # 从 server_a 获取文档密文的一半
    if os.path.exists(th_a_path):
        with open(th_a_path, "rb") as f:
            th_a = pickle.load(f)
        for tid in table_ids:
            bucket = th_a.get(tid)
            if bucket:
                part = bucket["docs_enc_part"].get(str(best_cluster))
                if part:
                    docs_part_a_hex = part
                    break

    # 从 server_b 获取文档密文的一半
    if os.path.exists(th_b_path):
        with open(th_b_path, "rb") as f:
            th_b = pickle.load(f)
        for tid in table_ids:
            bucket = th_b.get(tid)
            if bucket:
                part = bucket["docs_enc_part"].get(str(best_cluster))
                if part:
                    docs_part_b_hex = part
                    break

    return best_cluster, best_piece, docs_part_a_hex, docs_part_b_hex


def get_cluster_docs(cluster_idx: int, dataset: str):
    ddir = os.path.join(INDEX_DIR, dataset)
    assignments = np.load(os.path.join(ddir, 'assignments.npy'), allow_pickle=True)
    with open(os.path.join(META_DIR, f'{dataset}_meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)

    return [
        str(meta[i].get('id', ''))
        for i, cid in enumerate(assignments)
        if int(cid) == int(cluster_idx)
    ]


if __name__ == "__main__":
    from generate_trapdoor import generate_trapdoor
    ds = "enron_1k"
    t1, t2, qp = generate_trapdoor("Tell me about energy trading", ds)
    t0 = time.time()
    best_cluster, best_piece, part_a, part_b = secure_search(t1, t2, ds, q_piece=qp, debug=True)
    t1_ = time.time()
    print("best_cluster:", best_cluster, "best_piece:", best_piece)
    s1 = get_cluster_docs(best_cluster, ds)
    print(s1)
    print("time:", t1_ - t0)
    if part_a or part_b:
        docs = merge_and_decrypt_docs(ds, part_a or "", part_b or "")
        print("decrypted docs:", docs[:20])
