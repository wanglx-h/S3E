# # offline_build.py
# # 作用：离线一次性构建嵌入、聚类和 SimplePIR 协议，并保存到磁盘
# # 后续在线实验直接加载，不再重复构建
#
# import os
# import json
# import pickle
# import time
# from dataclasses import dataclass
# from typing import List, Dict, Optional
#
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity  # 虽然这里没用到，但保留无妨
# from sentence_transformers import SentenceTransformer
#
# import sys
# sys.path.append("/root/siton-tmp/SIMPLE-PIR-CODE")
#
# from simple_pir.config.pir_config import SimplePIRConfig, SecurityLevel
# from simple_pir.core.pir_protocol import SimplePIRProtocol
#
#
# # =========================
# # 全局配置
# # =========================
#
# DATASETS = {
#     "msmarco_1k": "/root/siton-tmp/data/msmarco_1k.json",
#     "msmarco_5k": "/root/siton-tmp/data/msmarco_5k.json",
#     "msmarco_10k": "/root/siton-tmp/data/msmarco_10k.json",
# }
#
# # 本地已下载好的 BGE 模型目录
# EMBEDDING_MODEL_NAME = "/root/siton-tmp/bge-base-en-v1.5"
#
# # 每个数据集的聚类数
# N_CLUSTERS_CONFIG = {
#     "msmarco_1k": 32,
#     "msmarco_5k": 64,
#     "msmarco_10k": 128,
# }
#
# # SimplePIR 安全等级
# SIMPLEPIR_SECURITY_LEVEL = SecurityLevel.MEDIUM
#
# # 离线状态输出目录（按你的要求：直接保存在 /root/siton-tmp/pir-rag 下）
# STATE_DIR = "/root/siton-tmp/pir-rag"
#
# GLOBAL_RANDOM_SEED = 42
# np.random.seed(GLOBAL_RANDOM_SEED)
#
#
# # =========================
# # 数据结构 & 工具函数
# # =========================
#
# @dataclass
# class Document:
#     doc_id: str
#     text: str
#     embedding: Optional[np.ndarray] = None
#     cluster_id: Optional[int] = None
#
#
# def ensure_dir(path: str):
#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)
#
#
# def load_documents_from_json(path: str, max_docs: Optional[int] = None) -> List[Document]:
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     docs: List[Document] = []
#     for i, item in enumerate(data):
#         if max_docs is not None and i >= max_docs:
#             break
#         doc_id = item.get("doc_id", str(i))
#         text = item.get("text", "")
#         docs.append(Document(doc_id=doc_id, text=text))
#     return docs
#
#
# def build_embedder(model_name: str) -> SentenceTransformer:
#     # 使用本地目录加载 BGE，不会访问网络
#     return SentenceTransformer(model_name)
#
#
# def embed_documents(
#     docs: List[Document], model: SentenceTransformer, batch_size: int = 32
# ) -> np.ndarray:
#     from tqdm import tqdm
#
#     texts = [d.text for d in docs]
#     embeddings = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Embedding docs"):
#         batch = texts[i:i + batch_size]
#         emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
#         embeddings.append(emb)
#     embeddings = np.vstack(embeddings)
#     for doc, emb in zip(docs, embeddings):
#         doc.embedding = emb
#     return embeddings
#
#
# def l2_normalize(vecs: np.ndarray) -> np.ndarray:
#     norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
#     return vecs / norm
#
#
# # =========================
# # 离线构建主逻辑
# # =========================
#
# def build_offline_state_for_dataset(
#     dataset_name: str,
#     json_path: str,
#     n_clusters: int,
#     embedder: SentenceTransformer,
# ):
#     print(f"\n====== Offline build for dataset: {dataset_name} ======")
#     print(f"Loading docs from: {json_path}")
#
#     t_total_start = time.time()  # 整个离线流程计时起点
#
#     docs = load_documents_from_json(json_path)
#     print(f"Loaded {len(docs)} documents.")
#
#     # 1) 文档嵌入
#     t_embed_start = time.time()
#     doc_embeddings = embed_documents(docs, embedder)
#     doc_embeddings = l2_normalize(doc_embeddings)
#     t_embed_end = time.time()
#
#     # 2) KMeans 聚类，得到 n 个簇和公共质心
#     print(f"[{dataset_name}] Running KMeans clustering with n_clusters={n_clusters} ...")
#     t_kmeans_start = time.time()
#     kmeans = KMeans(
#         n_clusters=n_clusters,
#         random_state=GLOBAL_RANDOM_SEED,
#         n_init="auto",
#     )
#     cluster_ids = kmeans.fit_predict(doc_embeddings)
#     cluster_centroids = l2_normalize(kmeans.cluster_centers_)
#     t_kmeans_end = time.time()
#
#     # 给 Document 标注 cluster_id
#     cluster_to_doc_indices: Dict[int, List[int]] = {cid: [] for cid in range(n_clusters)}
#     doc_id_to_index: Dict[str, int] = {}
#
#     for idx, (doc, cid) in enumerate(zip(docs, cluster_ids)):
#         c = int(cid)
#         doc.cluster_id = c
#         cluster_to_doc_indices[c].append(idx)
#         doc_id_to_index[doc.doc_id] = idx
#
#     # 3) 为每个簇构造“目标文档片段集合”文本（cluster_texts）
#     cluster_texts: Dict[int, str] = {}
#     for cid in range(n_clusters):
#         doc_indices = cluster_to_doc_indices[cid]
#         if not doc_indices:
#             cluster_texts[cid] = ""
#             continue
#         texts = [docs[i].text for i in doc_indices]
#         big_text = "\n".join(texts)
#         cluster_texts[cid] = big_text
#
#     # 4) 构造 SimplePIR 数据库：每个簇一条记录，并初始化协议
#     print(f"[{dataset_name}] Initializing SimplePIR protocol ...")
#     t_pir_init_start = time.time()
#     database_records = [cluster_texts[cid] for cid in sorted(cluster_texts.keys())]
#     config = SimplePIRConfig(security_level=SIMPLEPIR_SECURITY_LEVEL)
#     protocol = SimplePIRProtocol(database_records, config=config)
#     t_pir_init_end = time.time()
#
#     # 总离线时间（不含写 pkl 的磁盘 IO）
#     t_total_end = time.time()
#     total_time_sec = t_total_end - t_total_start
#     embed_time_sec = t_embed_end - t_embed_start
#     kmeans_time_sec = t_kmeans_end - t_kmeans_start
#     pir_init_time_sec = t_pir_init_end - t_pir_init_start
#
#     # 5) 序列化离线状态（含 SimplePIRProtocol）
#     ensure_dir(STATE_DIR)
#     out_path = os.path.join(STATE_DIR, f"{dataset_name}_state.pkl")
#
#     # 为减小体积，可以把 embedding 存成 float32
#     doc_emb_float32 = doc_embeddings.astype("float32")
#
#     state = {
#         "dataset_name": dataset_name,
#         "docs": docs,
#         "doc_embeddings": doc_emb_float32,
#         "cluster_ids": cluster_ids.astype("int32"),
#         "cluster_centroids": cluster_centroids.astype("float32"),
#         "cluster_to_doc_indices": cluster_to_doc_indices,
#         "doc_id_to_index": doc_id_to_index,
#         "cluster_texts": cluster_texts,
#         "security_level": SIMPLEPIR_SECURITY_LEVEL.value,
#         "pir_config_dict": config.to_dict(),
#         "pir_database_records": database_records,
#         "pir_protocol": protocol,
#         # 新增：离线阶段耗时统计（秒）
#         "offline_timings": {
#             "embed_time_sec": embed_time_sec,
#             "kmeans_time_sec": kmeans_time_sec,
#             "pir_init_time_sec": pir_init_time_sec,
#             "total_time_sec": total_time_sec,
#         },
#     }
#
#     with open(out_path, "wb") as f:
#         pickle.dump(state, f)
#
#     print(f"[{dataset_name}] Offline state saved to: {out_path}")
#     print(
#         f"[{dataset_name}] Offline timings (sec): "
#         f"embed={embed_time_sec:.2f}, "
#         f"kmeans={kmeans_time_sec:.2f}, "
#         f"pir_init={pir_init_time_sec:.2f}, "
#         f"total={total_time_sec:.2f}"
#     )
#
#
# def main():
#     ensure_dir(STATE_DIR)
#     embedder = build_embedder(EMBEDDING_MODEL_NAME)
#
#     for dataset_name, path in DATASETS.items():
#         if not os.path.exists(path):
#             print(f"[WARN] Dataset {dataset_name} path not found: {path}, skip.")
#             continue
#         n_clusters = N_CLUSTERS_CONFIG[dataset_name]
#         build_offline_state_for_dataset(
#             dataset_name=dataset_name,
#             json_path=path,
#             n_clusters=n_clusters,
#             embedder=embedder,
#         )
#
#
# if __name__ == "__main__":
#     main()

# offline_build.py
# 作用：离线一次性构建嵌入、聚类和 SimplePIR 协议，并保存到磁盘
# 后续在线实验直接加载，不再重复构建

import os
import json
import pickle
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity  # 虽然这里没用到，但保留无妨
from sentence_transformers import SentenceTransformer

import sys
sys.path.append("/root/siton-tmp/SIMPLE-PIR-CODE")

from simple_pir.config.pir_config import SimplePIRConfig, SecurityLevel
from simple_pir.core.pir_protocol import SimplePIRProtocol


# =========================
# 全局配置
# =========================

DATASETS = {
    "enron_1k": "/root/siton-tmp/data/enron_1k.json",
    "enron_5k": "/root/siton-tmp/data/enron_5k.json",
    "enron_10k": "/root/siton-tmp/data/enron_10k.json",

    "wiki_1k": "/root/siton-tmp/data/simplewiki_1k.json",
    "wiki_5k": "/root/siton-tmp/data/simplewiki_5k.json",
    "wiki_10k": "/root/siton-tmp/data/simplewiki_10k.json",
}


# 本地已下载好的 BGE 模型目录
EMBEDDING_MODEL_NAME = "/root/siton-tmp/bge-base-en-v1.5"

# 每个数据集的聚类数
N_CLUSTERS_CONFIG = {
    "enron_1k": 32,
    "enron_5k": 64,
    "enron_10k": 128,

    "wiki_1k": 32,
    "wiki_5k": 64,
    "wiki_10k": 128,
}


# SimplePIR 安全等级
SIMPLEPIR_SECURITY_LEVEL = SecurityLevel.MEDIUM

# 离线状态输出目录（按你的要求：直接保存在 /root/siton-tmp/pir-rag 下）
STATE_DIR = "/root/siton-tmp/pir-rag"

GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)


# =========================
# 数据结构 & 工具函数
# =========================

@dataclass
class Document:
    doc_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_documents_from_json(path: str, dataset_name: str, max_docs: Optional[int] = None) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Document] = []
    empty_cnt = 0

    # 用于避免 enron 的 id 重复导致覆盖/冲突（保留原始 id，同时保证唯一）
    seen_ids = {}

    def _make_unique_id(base_id: str) -> str:
        base_id = str(base_id)
        if base_id not in seen_ids:
            seen_ids[base_id] = 0
            return base_id
        seen_ids[base_id] += 1
        return f"{base_id}__dup{seen_ids[base_id]}"

    def _get_text_from_item(item: dict) -> str:
        # 兼容：你说 enron 是 content；你样例里是 text；wiki 是 content
        text = item.get("content") or item.get("text") or ""
        if isinstance(text, str):
            return text
        return ""

    # -------- ENRON: dict(query -> list[items]) --------
    if dataset_name.startswith("enron") and isinstance(data, dict):
        for q, items in data.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if max_docs is not None and len(docs) >= max_docs:
                    break
                if not isinstance(it, dict):
                    continue

                base_id = it.get("id")  # 按你的要求：用数据集里的 id
                if base_id is None:
                    # 实在没有 id 才退化为顺序号
                    base_id = str(len(docs))

                text = _get_text_from_item(it).strip()
                if not text:
                    empty_cnt += 1
                    continue

                # 如果同一个 id 在不同 query 下重复出现，这里自动去重/防冲突
                doc_id = _make_unique_id(base_id)

                docs.append(Document(doc_id=str(doc_id), text=text))

            if max_docs is not None and len(docs) >= max_docs:
                break

    # -------- WIKI: list[{"title":..., "content":...}] --------
    elif dataset_name.startswith("wiki") and isinstance(data, list):
        for i, it in enumerate(data):
            if max_docs is not None and len(docs) >= max_docs:
                break
            if not isinstance(it, dict):
                continue

            # 按先后顺序给 id（你要求的）
            doc_id = str(i)

            title = it.get("title") or ""
            content = (it.get("content") or "").strip()
            if not content:
                empty_cnt += 1
                continue

            # 可选：把 title 拼到正文头部，通常对检索更友好
            text = f"{title}\n\n{content}".strip() if title else content

            docs.append(Document(doc_id=doc_id, text=text))

    # -------- 其他/兜底：list 或 dict(data/docs/...) --------
    else:
        # 兼容一些常见封装结构
        if isinstance(data, dict):
            for k in ["data", "docs", "documents", "items"]:
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break

        if isinstance(data, list):
            for i, it in enumerate(data):
                if max_docs is not None and len(docs) >= max_docs:
                    break
                if not isinstance(it, dict):
                    continue
                doc_id = str(it.get("id") or it.get("doc_id") or i)
                text = _get_text_from_item(it).strip()
                if not text:
                    empty_cnt += 1
                    continue
                docs.append(Document(doc_id=doc_id, text=text))
        else:
            raise ValueError(f"Unexpected JSON format in {path}: root is {type(data)}")

    if not docs:
        raise ValueError(f"All documents are empty after parsing: {path}. skipped_empty={empty_cnt}")

    print(f"Loaded {len(docs)} non-empty documents (skipped empty={empty_cnt}).")
    return docs



def build_embedder(model_name: str) -> SentenceTransformer:
    # 使用本地目录加载 BGE，不会访问网络
    return SentenceTransformer(model_name)


def embed_documents(
    docs: List[Document], model: SentenceTransformer, batch_size: int = 32
) -> np.ndarray:
    from tqdm import tqdm

    texts = [d.text for d in docs]
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding docs"):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    for doc, emb in zip(docs, embeddings):
        doc.embedding = emb
    return embeddings


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norm


# =========================
# 离线构建主逻辑
# =========================

def build_offline_state_for_dataset(
    dataset_name: str,
    json_path: str,
    n_clusters: int,
    embedder: SentenceTransformer,
):
    print(f"\n====== Offline build for dataset: {dataset_name} ======")
    print(f"Loading docs from: {json_path}")

    t_total_start = time.time()  # 整个离线流程计时起点

    docs = load_documents_from_json(json_path, dataset_name=dataset_name)
    print(f"Loaded {len(docs)} documents.")

    # 1) 文档嵌入
    t_embed_start = time.time()
    doc_embeddings = embed_documents(docs, embedder)
    doc_embeddings = l2_normalize(doc_embeddings)
    t_embed_end = time.time()

    # 在 build_offline_state_for_dataset() 里，doc_embeddings 归一化之后加：
    print(f"[{dataset_name}] Embedding shape: {doc_embeddings.shape}")
    print(f"[{dataset_name}] Embedding std (mean over dims): {doc_embeddings.std(axis=0).mean():.6f}")

    # 统计“近似唯一”的向量个数（用 round 降低浮点噪声）
    approx_unique = np.unique(np.round(doc_embeddings, 4), axis=0).shape[0]
    print(f"[{dataset_name}] Approx unique embeddings (round=4): {approx_unique}/{len(doc_embeddings)}")
    empty_cnt = sum(1 for d in docs if not d.text or not d.text.strip())
    print(f"[{dataset_name}] Empty/blank texts: {empty_cnt}/{len(docs)}")

    # 2) KMeans 聚类，得到 n 个簇和公共质心
    print(f"[{dataset_name}] Running KMeans clustering with n_clusters={n_clusters} ...")
    t_kmeans_start = time.time()
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=GLOBAL_RANDOM_SEED,
        n_init="auto",
    )
    cluster_ids = kmeans.fit_predict(doc_embeddings)
    cluster_centroids = l2_normalize(kmeans.cluster_centers_)
    t_kmeans_end = time.time()

    # 给 Document 标注 cluster_id
    cluster_to_doc_indices: Dict[int, List[int]] = {cid: [] for cid in range(n_clusters)}
    doc_id_to_index: Dict[str, int] = {}

    for idx, (doc, cid) in enumerate(zip(docs, cluster_ids)):
        c = int(cid)
        doc.cluster_id = c
        cluster_to_doc_indices[c].append(idx)
        doc_id_to_index[doc.doc_id] = idx

    # 3) 为每个簇构造“目标文档片段集合”文本（cluster_texts）
    cluster_texts: Dict[int, str] = {}
    for cid in range(n_clusters):
        doc_indices = cluster_to_doc_indices[cid]
        if not doc_indices:
            cluster_texts[cid] = ""
            continue
        texts = [docs[i].text for i in doc_indices]
        big_text = "\n".join(texts)
        cluster_texts[cid] = big_text

    # 4) 构造 SimplePIR 数据库：每个簇一条记录，并初始化协议
    print(f"[{dataset_name}] Initializing SimplePIR protocol ...")
    t_pir_init_start = time.time()
    database_records = [cluster_texts[cid] for cid in sorted(cluster_texts.keys())]
    config = SimplePIRConfig(security_level=SIMPLEPIR_SECURITY_LEVEL)
    protocol = SimplePIRProtocol(database_records, config=config)
    t_pir_init_end = time.time()

    # 总离线时间（不含写 pkl 的磁盘 IO）
    t_total_end = time.time()
    total_time_sec = t_total_end - t_total_start
    embed_time_sec = t_embed_end - t_embed_start
    kmeans_time_sec = t_kmeans_end - t_kmeans_start
    pir_init_time_sec = t_pir_init_end - t_pir_init_start

    # 5) 序列化离线状态（含 SimplePIRProtocol）
    ensure_dir(STATE_DIR)
    out_path = os.path.join(STATE_DIR, f"{dataset_name}_state.pkl")

    # 为减小体积，可以把 embedding 存成 float32
    doc_emb_float32 = doc_embeddings.astype("float32")

    state = {
        "dataset_name": dataset_name,
        "docs": docs,
        "doc_embeddings": doc_emb_float32,
        "cluster_ids": cluster_ids.astype("int32"),
        "cluster_centroids": cluster_centroids.astype("float32"),
        "cluster_to_doc_indices": cluster_to_doc_indices,
        "doc_id_to_index": doc_id_to_index,
        "cluster_texts": cluster_texts,
        "security_level": SIMPLEPIR_SECURITY_LEVEL.value,
        "pir_config_dict": config.to_dict(),
        "pir_database_records": database_records,
        "pir_protocol": protocol,
        # 新增：离线阶段耗时统计（秒）
        "offline_timings": {
            "embed_time_sec": embed_time_sec,
            "kmeans_time_sec": kmeans_time_sec,
            "pir_init_time_sec": pir_init_time_sec,
            "total_time_sec": total_time_sec,
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(state, f)

    print(f"[{dataset_name}] Offline state saved to: {out_path}")
    print(
        f"[{dataset_name}] Offline timings (sec): "
        f"embed={embed_time_sec:.2f}, "
        f"kmeans={kmeans_time_sec:.2f}, "
        f"pir_init={pir_init_time_sec:.2f}, "
        f"total={total_time_sec:.2f}"
    )


def main():
    ensure_dir(STATE_DIR)
    embedder = build_embedder(EMBEDDING_MODEL_NAME)

    for dataset_name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"[WARN] Dataset {dataset_name} path not found: {path}, skip.")
            continue
        n_clusters = N_CLUSTERS_CONFIG[dataset_name]
        build_offline_state_for_dataset(
            dataset_name=dataset_name,
            json_path=path,
            n_clusters=n_clusters,
            embedder=embedder,
        )


if __name__ == "__main__":
    main()
