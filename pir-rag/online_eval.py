# # online_eval.py
# import os
# import json
# import pickle
# from dataclasses import dataclass
# from typing import List, Dict, Tuple
#
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
#
# import sys
# sys.path.append("/root/siton-tmp/SIMPLE-PIR-CODE")
# from simple_pir.config.pir_config import SimplePIRConfig, SecurityLevel
# from simple_pir.core.pir_protocol import SimplePIRProtocol
#
#
# # 与 offline_build.py 中保持一致
# STATE_DIR = "/root/siton-tmp/pir-rag"
# EMBEDDING_MODEL_NAME = "/root/siton-tmp/bge-base-en-v1.5"
# TOP_K = 10
#
# # 明文 RAG 结果目录
# PLAIN_RESULTS_DIR = "/root/siton-tmp/outputs/plain_results"
#
# GLOBAL_RANDOM_SEED = 42
# np.random.seed(GLOBAL_RANDOM_SEED)
#
#
# # =========================
# # 15 条固定查询
# # =========================
#
# PREDEFINED_QUERIES: List[str] = [
#     "How do you use the Stefan-Boltzmann law to calculate the radius of a star such as Rigel from its luminosity and surface temperature?",
#     "What developmental milestones and typical behaviors should you expect from an 8 year old child at home and at school?",
#     "What are the symptoms of a head lice infestation and how can you check for lice, eggs, and nits on a child's scalp?",
#     "What special features does the Burj Khalifa in Dubai have and why was it renamed from Burj Dubai?",
#     "What kinds of homes and land are for sale near La Grange, California, and what are their typical sizes and prices?",
#     "What are the main characteristics, temperament, and exercise needs of the Dogo Argentino dog breed?",
#     "How are custom safety nets used in industry and what kinds of clients and applications does a company like US Netting serve?",
#     "What are effective ways to remove weeds from a garden and prevent them from coming back?",
#     "How common is urinary incontinence in the United States, what can cause it, and is it just a normal part of aging?",
#     "How did President Franklin D. Roosevelt prepare the United States for World War II before Pearl Harbor while the country was still isolationist?",
#     "If you have multiple sclerosis and difficulty swallowing pills, is it safe to crush Valium and other medications to make them easier to swallow?",
#     "What strategies can help you get better results when dealing with customer service representatives at cable companies or airlines?",
#     "In Spanish, what does the word 'machacado' mean and how is the verb 'machacar' used in different contexts?",
#     "When building a concrete path, how should you design and support plywood formwork so that it is strong enough and keeps the concrete in place?",
#     "Why do people join political parties, and which political party did U.S. presidents Woodrow Wilson and Herbert Hoover belong to?",
# ]
#
#
# # =========================
# # 数据结构 & 工具
# # =========================
#
# @dataclass
# class Document:
#     doc_id: str
#     text: str
#     embedding: np.ndarray = None
#     cluster_id: int = None
#
#
# @dataclass
# class QueryResult:
#     query_text: str
#     retrieved_doc_ids: List[str]
#     scores: List[float]
#
#
# def l2_normalize(vecs: np.ndarray) -> np.ndarray:
#     norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
#     return vecs / norm
#
#
# class PIRRAGEvaluator:
#     """
#     执行完整的 PIR-RAG + SimplePIR 流程，并细分时间：
#       - client_build_time: 查询构建（embedding + 选簇）
#       - server_private_time: SimplePIR 私有计算
#       - client_finalize_time: 簇内重排
#       - total_time: 上述三部分之和
#     """
#
#     def __init__(self, state_path: str):
#         with open(state_path, "rb") as f:
#             state = pickle.load(f)
#
#         self.dataset_name: str = state["dataset_name"]
#         self.docs_raw = state["docs"]
#         self.doc_embeddings: np.ndarray = state["doc_embeddings"].astype("float32")
#         self.cluster_ids: np.ndarray = state["cluster_ids"].astype("int32")
#         self.cluster_centroids: np.ndarray = state["cluster_centroids"].astype("float32")
#         self.cluster_to_doc_indices: Dict[int, List[int]] = state["cluster_to_doc_indices"]
#         self.doc_id_to_index: Dict[str, int] = state["doc_id_to_index"]
#         self.cluster_texts: Dict[int, str] = state["cluster_texts"]
#         self.security_level_str: str = state["security_level"]
#         self.pir_config_dict: Dict = state["pir_config_dict"]
#         self.pir_database_records: List[str] = state["pir_database_records"]
#         self.offline_timings: Dict = state.get("offline_timings", {})
#
#         # SimplePIR 协议实例（offline 已初始化）
#         self.protocol: SimplePIRProtocol = state["pir_protocol"]
#
#         # 构造 docs 列表（doc_id / text），以防后续需要
#         self.docs: List[Document] = []
#         for d_raw, cid in zip(self.docs_raw, self.cluster_ids):
#             self.docs.append(
#                 Document(
#                     doc_id=d_raw.doc_id,
#                     text=d_raw.text,
#                     embedding=None,
#                     cluster_id=int(cid),
#                 )
#             )
#
#         self.n_clusters = len(self.cluster_texts)
#
#         print(f"[{self.dataset_name}] Loaded offline state from {state_path}")
#         print(f"[{self.dataset_name}] #docs={len(self.docs)}, #clusters={self.n_clusters}")
#         print(f"[{self.dataset_name}] SimplePIR security level: {self.security_level_str}")
#         if self.offline_timings:
#             print(f"[{self.dataset_name}] Offline timings (sec): {self.offline_timings}")
#
#         self.config = SimplePIRConfig.from_dict(self.pir_config_dict)
#         self.comm_estimate = self.config.estimate_communication_cost(self.n_clusters)
#
#     def _select_best_cluster_for_query(self, query_emb: np.ndarray) -> int:
#         """
#         客户端本地：用查询向量与聚类质心计算相似度，得到最近簇索引 i。
#         """
#         query_emb = query_emb.reshape(1, -1)
#         sims = cosine_similarity(query_emb, self.cluster_centroids)[0]
#         return int(np.argmax(sims))
#
#     def pir_retrieve(
#         self, query_text: str, embedder: SentenceTransformer, top_k: int = 10
#     ) -> Tuple[QueryResult, float, float, float, float, int, int]:
#         """
#         返回：
#           - QueryResult：PIR-RAG Top-K 结果
#           - client_build_time_sec  : 客户端查询构建时间
#           - server_private_time_sec: 服务器端私有计算时间（SimplePIR）
#           - client_finalize_time_sec: 客户端最终化时间（簇内重排）
#           - total_time_sec         : 三部分之和
#           - uplink_bytes_est       : 加密查询大小
#           - downlink_bytes_est     : 加密响应大小
#         """
#         import time
#
#         t_total_start = time.time()
#
#         # (1) 客户端：查询构建（embedding + 选簇）
#         t_build_start = time.time()
#         q_emb = embedder.encode([query_text], convert_to_numpy=True)
#         q_emb = l2_normalize(q_emb)[0]
#         best_cluster = self._select_best_cluster_for_query(q_emb)
#         t_build_end = time.time()
#
#         # (2) 服务器端：SimplePIR 私有计算
#         t_pir_start = time.time()
#         pir_ret = self.protocol.retrieve_item(best_cluster, verify_result=False)
#         t_pir_end = time.time()
#
#         # (3) 客户端：最终化（簇内重排）
#         t_finalize_start = time.time()
#
#         # pir_ret["item_data"] 是目标聚类 i 的明文文档片段集合（本实现中不直接用来排序）
#         _cluster_plain_text = pir_ret.get("item_data", "")
#
#         doc_indices = self.cluster_to_doc_indices.get(best_cluster, [])
#         if not doc_indices:
#             qr = QueryResult(query_text=query_text, retrieved_doc_ids=[], scores=[])
#         else:
#             emb_matrix = self.doc_embeddings[doc_indices]
#             q_emb_vec = q_emb.reshape(1, -1)
#             sims = cosine_similarity(q_emb_vec, emb_matrix)[0]
#             sorted_idx = np.argsort(-sims)
#             top_idx = sorted_idx[:top_k]
#             retrieved_doc_ids = [self.docs[doc_indices[i]].doc_id for i in top_idx]
#             scores = [float(sims[i]) for i in top_idx]
#             qr = QueryResult(
#                 query_text=query_text,
#                 retrieved_doc_ids=retrieved_doc_ids,
#                 scores=scores,
#             )
#
#         t_finalize_end = time.time()
#         t_total_end = time.time()
#
#         client_build_time_sec = t_build_end - t_build_start
#         server_private_time_sec = t_pir_end - t_pir_start
#         client_finalize_time_sec = t_finalize_end - t_finalize_start
#         total_time_sec = t_total_end - t_total_start
#
#         uplink_bytes = self.comm_estimate["query_size_bytes"]
#         downlink_bytes = self.comm_estimate["response_size_bytes"]
#
#         return (
#             qr,
#             client_build_time_sec,
#             server_private_time_sec,
#             client_finalize_time_sec,
#             total_time_sec,
#             uplink_bytes,
#             downlink_bytes,
#         )
#
#
# # =========================
# # 明文 RAG 结果加载 & 准确率计算
# # =========================
#
# def load_plain_results_for_dataset(dataset_name: str) -> Dict[str, List[Dict]]:
#     """
#     从 /root/siton-tmp/outputs/plain_results/{dataset_name}.json 读取明文 RAG 的结果。
#     期望结构： { query_text: [ { "id": doc_id, ... }, ... ], ... }
#     """
#     path = os.path.join(PLAIN_RESULTS_DIR, f"{dataset_name}.json")
#     if not os.path.exists(path):
#         print(f"[WARN] Plain RAG results not found for {dataset_name}: {path}")
#         return {}
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
#
#
# def query_accuracy_from_plain(
#     pir_result: QueryResult, plain_results: Dict[str, List[Dict]]
# ) -> float:
#     """
#     对单个查询计算准确率：
#       - 如果 PIR-RAG 返回的 doc_id 列表与明文 RAG 的 doc_id 集合有交集，则该查询准确率 = 1
#       - 否则 = 0
#     """
#     if pir_result.query_text not in plain_results:
#         # 明文中没有这个查询，视为无法评价，返回 0
#         return 0.0
#
#     pir_ids = set(str(did).strip() for did in pir_result.retrieved_doc_ids)
#     plain_docs = plain_results[pir_result.query_text]
#     plain_ids = set(str(item.get("id")).strip() for item in plain_docs)
#
#     if not pir_ids or not plain_ids:
#         return 0.0
#
#     return 1.0 if pir_ids & plain_ids else 0.0
#
#
# # =========================
# # 运行实验
# # =========================
#
# def run_experiment_for_dataset(dataset_name: str):
#     state_path = os.path.join(STATE_DIR, f"{dataset_name}_state.pkl")
#     if not os.path.exists(state_path):
#         print(f"[ERROR] Offline state not found for {dataset_name}: {state_path}")
#         return
#
#     print(f"\n========== Online PIR-RAG experiment: {dataset_name} ==========")
#
#     evaluator = PIRRAGEvaluator(state_path)
#     embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
#
#     # 载入明文 RAG 结果
#     plain_results = load_plain_results_for_dataset(dataset_name)
#     if not plain_results:
#         print(f"[WARN] No plain RAG results for {dataset_name}, skip accuracy evaluation.")
#         return
#
#     client_build_times = []
#     server_private_times = []
#     client_finalize_times = []
#     total_times = []
#     uplinks = []
#     downlinks = []
#
#     per_query_acc = []
#     valid_query_count = 0
#
#     from tqdm import tqdm
#
#     for q_text in tqdm(PREDEFINED_QUERIES, desc=f"Queries ({dataset_name})"):
#         # PIR-RAG + SimplePIR 一次完整查询
#         (
#             pir_res,
#             t_build,
#             t_pir,
#             t_finalize,
#             t_total,
#             up_bytes,
#             down_bytes,
#         ) = evaluator.pir_retrieve(q_text, embedder, top_k=TOP_K)
#
#         client_build_times.append(t_build)
#         server_private_times.append(t_pir)
#         client_finalize_times.append(t_finalize)
#         total_times.append(t_total)
#         uplinks.append(up_bytes)
#         downlinks.append(down_bytes)
#
#         # 基于明文 RAG 结果计算该查询的准确率（命中为 1，否则 0）
#         acc = query_accuracy_from_plain(pir_res, plain_results)
#         per_query_acc.append(acc)
#         valid_query_count += 1
#
#         print(
#             f"[Query Acc] {q_text[:60]}... -> acc={acc:.1f}, "
#             f"build={t_build:.4f}s, pir={t_pir:.4f}s, finalize={t_finalize:.4f}s, total={t_total:.4f}s"
#         )
#
#     avg_client_build = float(np.mean(client_build_times)) if client_build_times else 0.0
#     avg_server_private = float(np.mean(server_private_times)) if server_private_times else 0.0
#     avg_client_finalize = float(np.mean(client_finalize_times)) if client_finalize_times else 0.0
#     avg_total = float(np.mean(total_times)) if total_times else 0.0
#     avg_uplink_kb = float(np.mean(uplinks) / 1024) if uplinks else 0.0
#     avg_downlink_mb = float(np.mean(downlinks) / (1024 * 1024)) if downlinks else 0.0
#
#     # 最终准确率：准确率为 1 的查询数 / 所有查询数
#     final_accuracy = float(np.mean(per_query_acc)) if per_query_acc else 0.0
#
#     print(f"\n[{dataset_name}] PIR-RAG + SimplePIR vs plaintext RAG:")
#     print(f"  Num queries                     : {valid_query_count}")
#     print(f"  Avg client query build time     : {avg_client_build:.4f} s")
#     print(f"  Avg server private compute time : {avg_server_private:.4f} s")
#     print(f"  Avg client finalize time        : {avg_client_finalize:.4f} s")
#     print(f"  Avg TOTAL query time            : {avg_total:.4f} s")
#     print(f"  Avg uplink                      : {avg_uplink_kb:.2f} KB (estimated)")
#     print(f"  Avg downlink                    : {avg_downlink_mb:.4f} MB (estimated)")
#     print(f"  Accuracy (hit any plain doc)    : {final_accuracy:.3f}")
#
#
# def main():
#     for dataset_name in ["msmarco_1k", "msmarco_5k", "msmarco_10k"]:
#         run_experiment_for_dataset(dataset_name)
#
#
# if __name__ == "__main__":
#     main()

# online_eval.py

import os
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import sys
sys.path.append("/root/siton-tmp/SIMPLE-PIR-CODE")
from simple_pir.config.pir_config import SimplePIRConfig, SecurityLevel
from simple_pir.core.pir_protocol import SimplePIRProtocol


# 与 offline_build.py 中保持一致
STATE_DIR = "/root/siton-tmp/pir-rag"
EMBEDDING_MODEL_NAME = "/root/siton-tmp/bge-base-en-v1.5"
TOP_K = 10

# 明文 RAG 结果目录
PLAIN_RESULTS_DIR = "/root/siton-tmp/outputs/plain_results"

GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)


# =========================
# Enron / Wiki 查询语句
# =========================

ENRON_QUERIES: List[str] = [
    # "What meetings are scheduled?",
    "Tell me about energy trading",
    "What contracts were discussed?",
    "What are the price forecasts?",
    # "What reports need analysis?",
    # "What projects are in development?",
    "What companies are involved?",
    "What emails need attention?",
    "What conference calls are planned?",
    "What financial information is available?",
    "Emails about SEC strategy meetings",
    "Messages mentioning building access or badges",
    "HR newsletters on labor or employment policy",
    # "Forwards with BNA Daily Labor Report content",
    # "Memos on minimum wage or unemployment issues",
    "Emails discussing union negotiations or wage increases",
    "Messages about post-9/11 employment impacts",
    "Notes on federal worker discrimination or whistleblower cases",
    "Emails that list multiple labor news headlines",
    # "Messages sharing external news links with login info",
    "Internal calendar or on-call notification emails",
    # "Emails between facilities or admin staff about office locations",
    "Messages referencing ILO or international labor standards",
    "Forwards about appointments to U.S. labor-related posts",
    "Emails on benefit or donation program changes",
    "Threads with multiple HR recipients in one blast",
    "Messages mentioning airport security or related legislation",
    "Emails summarizing congressional labor actions",
    "Messages about court rulings on workplace drug testing",
    "Long digest-style labor and employment updates",
]

WIKI_QUERIES: List[str] = [
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
    # "Historical events that happened in April",
    "April cultural festivals in Europe or Asia",
    "Sports or major events usually held in April",
    # "August month overview and calendar facts",
    # "Etymology or origin of the name August",
    # "August national or religious holidays",
    # "August historical events in the 20th century",
    # "Definition of art as human creative activity",
    # "Categories of art such as visual or performing",
    # "Discussion of art versus design",
    "Short history outline of art across eras",
    "Examples of everyday objects treated as art",
    "Comparison of April seasons across hemispheres",
    # "August cultural festivals and public holidays",
]


def get_queries_for_dataset(dataset_name: str) -> List[str]:
    """
    根据数据集名称返回对应查询集合：
      - enron_1k / 5k / 10k 使用 ENRON_QUERIES（30 条）
      - wiki_1k / 5k / 10k 使用 WIKI_QUERIES
    """
    name = dataset_name.lower()
    if name.startswith("enron"):
        return ENRON_QUERIES
    elif name.startswith("wiki"):
        return WIKI_QUERIES
    else:
        # 兜底：未知数据集返回空列表（避免跑错）
        print(f"[WARN] Unknown dataset for query selection: {dataset_name}")
        return []


# =========================
# 数据结构 & 工具
# =========================

@dataclass
class Document:
    doc_id: str
    text: str
    embedding: np.ndarray = None
    cluster_id: int = None


@dataclass
class QueryResult:
    query_text: str
    retrieved_doc_ids: List[str]
    scores: List[float]


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norm


class PIRRAGEvaluator:
    """
    执行完整的 PIR-RAG + SimplePIR 流程，并细分时间：
      - client_build_time: 查询构建（embedding + 选簇）
      - server_private_time: SimplePIR 私有计算
      - client_finalize_time: 簇内重排
      - total_time: 上述三部分之和
    """

    def __init__(self, state_path: str):
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        self.dataset_name: str = state["dataset_name"]
        self.docs_raw = state["docs"]
        self.doc_embeddings: np.ndarray = state["doc_embeddings"].astype("float32")
        self.cluster_ids: np.ndarray = state["cluster_ids"].astype("int32")
        self.cluster_centroids: np.ndarray = state["cluster_centroids"].astype("float32")
        self.cluster_to_doc_indices: Dict[int, List[int]] = state["cluster_to_doc_indices"]
        self.doc_id_to_index: Dict[str, int] = state["doc_id_to_index"]
        self.cluster_texts: Dict[int, str] = state["cluster_texts"]
        self.security_level_str: str = state["security_level"]
        self.pir_config_dict: Dict = state["pir_config_dict"]
        self.pir_database_records: List[str] = state["pir_database_records"]
        self.offline_timings: Dict = state.get("offline_timings", {})

        # SimplePIR 协议实例（offline 已初始化）
        self.protocol: SimplePIRProtocol = state["pir_protocol"]

        # 构造 docs 列表（doc_id / text）
        self.docs: List[Document] = []
        for d_raw, cid in zip(self.docs_raw, self.cluster_ids):
            self.docs.append(
                Document(
                    doc_id=d_raw.doc_id,
                    text=d_raw.text,
                    embedding=None,
                    cluster_id=int(cid),
                )
            )

        self.n_clusters = len(self.cluster_texts)

        print(f"[{self.dataset_name}] Loaded offline state from {state_path}")
        print(f"[{self.dataset_name}] #docs={len(self.docs)}, #clusters={self.n_clusters}")
        print(f"[{self.dataset_name}] SimplePIR security level: {self.security_level_str}")
        if self.offline_timings:
            print(f"[{self.dataset_name}] Offline timings (sec): {self.offline_timings}")

        self.config = SimplePIRConfig.from_dict(self.pir_config_dict)
        self.comm_estimate = self.config.estimate_communication_cost(self.n_clusters)

    def _select_best_cluster_for_query(self, query_emb: np.ndarray) -> int:
        """
        客户端本地：用查询向量与聚类质心计算相似度，得到最近簇索引 i。
        """
        query_emb = query_emb.reshape(1, -1)
        sims = cosine_similarity(query_emb, self.cluster_centroids)[0]
        return int(np.argmax(sims))

    def pir_retrieve(
        self, query_text: str, embedder: SentenceTransformer, top_k: int = 10
    ) -> Tuple[QueryResult, float, float, float, float, int, int]:
        """
        返回：
          - QueryResult：PIR-RAG Top-K 结果
          - client_build_time_sec  : 客户端查询构建时间
          - server_private_time_sec: 服务器端私有计算时间（SimplePIR）
          - client_finalize_time_sec: 客户端最终化时间（簇内重排）
          - total_time_sec         : 三部分之和
          - uplink_bytes_est       : 加密查询大小
          - downlink_bytes_est     : 加密响应大小
        """
        import time

        t_total_start = time.time()

        # (1) 客户端：查询构建（embedding + 选簇）
        t_build_start = time.time()
        q_emb = embedder.encode([query_text], convert_to_numpy=True)
        q_emb = l2_normalize(q_emb)[0]
        best_cluster = self._select_best_cluster_for_query(q_emb)
        t_build_end = time.time()

        # (2) 服务器端：SimplePIR 私有计算
        t_pir_start = time.time()
        pir_ret = self.protocol.retrieve_item(best_cluster, verify_result=False)
        t_pir_end = time.time()

        # (3) 客户端：最终化（簇内重排）
        t_finalize_start = time.time()

        # pir_ret["item_data"] 是目标聚类 i 的明文文档片段集合（本实现中不直接用来排序）
        _cluster_plain_text = pir_ret.get("item_data", "")

        doc_indices = self.cluster_to_doc_indices.get(best_cluster, [])
        if not doc_indices:
            qr = QueryResult(query_text=query_text, retrieved_doc_ids=[], scores=[])
        else:
            emb_matrix = self.doc_embeddings[doc_indices]
            q_emb_vec = q_emb.reshape(1, -1)
            sims = cosine_similarity(q_emb_vec, emb_matrix)[0]
            sorted_idx = np.argsort(-sims)
            top_idx = sorted_idx[:top_k]
            retrieved_doc_ids = [self.docs[doc_indices[i]].doc_id for i in top_idx]
            scores = [float(sims[i]) for i in top_idx]
            qr = QueryResult(
                query_text=query_text,
                retrieved_doc_ids=retrieved_doc_ids,
                scores=scores,
            )

        t_finalize_end = time.time()
        t_total_end = time.time()

        client_build_time_sec = t_build_end - t_build_start
        server_private_time_sec = t_pir_end - t_pir_start
        client_finalize_time_sec = t_finalize_end - t_finalize_start
        total_time_sec = t_total_end - t_total_start

        uplink_bytes = self.comm_estimate["query_size_bytes"]
        downlink_bytes = self.comm_estimate["response_size_bytes"]

        return (
            qr,
            client_build_time_sec,
            server_private_time_sec,
            client_finalize_time_sec,
            total_time_sec,
            uplink_bytes,
            downlink_bytes,
        )


# =========================
# 明文 RAG 结果加载 & 准确率计算
# =========================

def load_plain_results_for_dataset(dataset_name: str) -> Dict[str, List[Dict]]:
    """
    从 /root/siton-tmp/outputs/plain_results/{dataset_name}.json 读取明文 RAG 的结果。
    期望结构： { query_text: [ { "id": doc_id, ... }, ... ], ... }
    """
    path = os.path.join(PLAIN_RESULTS_DIR, f"{dataset_name}.json")
    if not os.path.exists(path):
        print(f"[WARN] Plain RAG results not found for {dataset_name}: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_doc_id(doc_id: str) -> str:
    """
    统一 doc_id 形式，避免离线构建为去重添加的后缀影响准确率判定。
    例：
      "58.__dup1" -> "58."
      "58."       -> "58."
    """
    s = str(doc_id).strip()
    if "__dup" in s:
        s = s.split("__dup", 1)[0]
    return s.strip()


def query_accuracy_from_plain(
    pir_result: QueryResult, plain_results: Dict[str, List[Dict]]
) -> float:
    """
    对单个查询计算准确率：
      - 如果 PIR-RAG 返回的 doc_id 列表与明文 RAG 的 doc_id 集合有交集，则该查询准确率 = 1
      - 否则 = 0
    """
    if pir_result.query_text not in plain_results:
        # 明文中没有这个查询，视为无法评价，返回 0
        return 0.0

    pir_ids = set(_normalize_doc_id(did) for did in pir_result.retrieved_doc_ids)

    plain_docs = plain_results[pir_result.query_text]
    plain_ids = set(
        _normalize_doc_id(item.get("id"))
        for item in plain_docs
        if isinstance(item, dict) and item.get("id") is not None
    )

    if not pir_ids or not plain_ids:
        return 0.0

    return 1.0 if pir_ids & plain_ids else 0.0


# =========================
# 运行实验
# =========================

def run_experiment_for_dataset(dataset_name: str):
    state_path = os.path.join(STATE_DIR, f"{dataset_name}_state.pkl")
    if not os.path.exists(state_path):
        print(f"[ERROR] Offline state not found for {dataset_name}: {state_path}")
        return

    print(f"\n========== Online PIR-RAG experiment: {dataset_name} ==========")

    evaluator = PIRRAGEvaluator(state_path)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 载入明文 RAG 结果
    plain_results = load_plain_results_for_dataset(dataset_name)
    if not plain_results:
        print(f"[WARN] No plain RAG results for {dataset_name}, skip accuracy evaluation.")
        return

    # 根据数据集选择对应的查询集合
    queries = get_queries_for_dataset(dataset_name)
    if not queries:
        print(f"[WARN] No queries defined for dataset {dataset_name}, skip.")
        return

    client_build_times = []
    server_private_times = []
    client_finalize_times = []
    total_times = []
    uplinks = []
    downlinks = []

    per_query_acc = []
    valid_query_count = 0

    from tqdm import tqdm

    for q_text in tqdm(queries, desc=f"Queries ({dataset_name})"):
        # PIR-RAG + SimplePIR 一次完整查询
        (
            pir_res,
            t_build,
            t_pir,
            t_finalize,
            t_total,
            up_bytes,
            down_bytes,
        ) = evaluator.pir_retrieve(q_text, embedder, top_k=TOP_K)

        client_build_times.append(t_build)
        server_private_times.append(t_pir)
        client_finalize_times.append(t_finalize)
        total_times.append(t_total)
        uplinks.append(up_bytes)
        downlinks.append(down_bytes)

        # 基于明文 RAG 结果计算该查询的准确率（命中为 1，否则 0）
        acc = query_accuracy_from_plain(pir_res, plain_results)
        per_query_acc.append(acc)
        valid_query_count += 1

        print(
            f"[Query Acc] {q_text[:60]}... -> acc={acc:.1f}, "
            f"build={t_build:.4f}s, pir={t_pir:.4f}s, finalize={t_finalize:.4f}s, total={t_total:.4f}s"
        )

    avg_client_build = float(np.mean(client_build_times)) if client_build_times else 0.0
    avg_server_private = float(np.mean(server_private_times)) if server_private_times else 0.0
    avg_client_finalize = float(np.mean(client_finalize_times)) if client_finalize_times else 0.0
    avg_total = float(np.mean(total_times)) if total_times else 0.0
    avg_uplink_kb = float(np.mean(uplinks) / 1024) if uplinks else 0.0
    avg_downlink_mb = float(np.mean(downlinks) / (1024 * 1024)) if downlinks else 0.0

    # 最终准确率：准确率为 1 的查询数 / 所有查询数
    final_accuracy = float(np.mean(per_query_acc)) if per_query_acc else 0.0

    print(f"\n[{dataset_name}] PIR-RAG + SimplePIR vs plaintext RAG:")
    print(f"  Num queries                     : {valid_query_count}")
    print(f"  Avg client query build time     : {avg_client_build:.4f} s")
    print(f"  Avg server private compute time : {avg_server_private:.4f} s")
    print(f"  Avg client finalize time        : {avg_client_finalize:.4f} s")
    print(f"  Avg TOTAL query time            : {avg_total:.4f} s")
    print(f"  Avg uplink                      : {avg_uplink_kb:.2f} KB (estimated)")
    print(f"  Avg downlink                    : {avg_downlink_mb:.4f} MB (estimated)")
    print(f"  Accuracy (hit any plain doc)    : {final_accuracy:.3f}")


def main():
    # Enron + Wiki 六个数据集
    for dataset_name in [
        "enron_1k",
        "enron_5k",
        "enron_10k",
        "wiki_1k",
        "wiki_5k",
        "wiki_10k",
    ]:
        run_experiment_for_dataset(dataset_name)


if __name__ == "__main__":
    main()


