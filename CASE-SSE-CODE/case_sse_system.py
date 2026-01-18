# import json
# import os
# import time
# import re
# import numpy as np
# from typing import List, Dict, Any
# from gensim.models import Word2Vec
# from sklearn.cluster import KMeans
# import torch
#
# from secure_index import SecureIndexBuilder
#
# enron  wiki
# # ============================================================
# # 文档对象
# # ============================================================
# class Document:
#     def __init__(self, doc_id: int, title: str, content: str):
#         self.doc_id = doc_id
#         self.title = title
#         self.content = content
#         self.tokens: List[str] = []
#
#
# # ============================================================
# # 文本预处理（用于构建 vocab & token 索引）
# # ============================================================
# class DataPreprocessor:
#     def __init__(self, vocab_size=2000):
#         self.vocab_size = vocab_size
#         self.vocabulary: Dict[str, int] = {}     # word -> index(0~1999)
#         self.word_freq: Dict[str, int] = {}
#
#     # --------------------------------
#     # 基础清洗
#     # --------------------------------
#     @staticmethod
#     def tokenize(text: str) -> List[str]:
#         text = text.lower()
#         words = re.findall(r"[a-zA-Z]+", text)
#         return words
#
#     # --------------------------------
#     # 构建词表（top-2000 高频词）
#     # --------------------------------
#     def build_vocabulary(self, docs: List["Document"]):
#         """
#         同时完成两件事：
#         1. 给每个 doc 填好 doc.tokens
#         2. 基于这些 tokens 统计词频，截断成 top-2000 词汇表
#         """
#         freq: Dict[str, int] = {}
#
#         for doc in docs:
#             # 如果 tokens 还没填，就先分词
#             if not getattr(doc, "tokens", None):
#                 doc.tokens = self.tokenize(doc.content)
#
#             for w in doc.tokens:
#                 freq[w] = freq.get(w, 0) + 1
#
#         # 按频率排序，取前 vocab_size 个词
#         sorted_words = sorted(freq.items(), key=lambda x: -x[1])
#         top_words = sorted_words[: self.vocab_size]
#
#         self.vocabulary = {w: i for i, (w, _) in enumerate(top_words)}
#         self.word_freq = {w: c for (w, c) in top_words}
#
#     # --------------------------------
#     # 将文档 tokens 转成词频向量（2000维）
#     # --------------------------------
#     def doc_to_vector(self, doc: Document) -> np.ndarray:
#         # 确保有 tokens
#         if not getattr(doc, "tokens", None):
#             doc.tokens = self.tokenize(doc.content or "")
#
#         vec = np.zeros(self.vocab_size, dtype=float)
#         for w in doc.tokens:
#             if w in self.vocabulary:
#                 vec[self.vocabulary[w]] += 1
#
#         if vec.sum() > 0:
#             vec = vec / vec.sum()
#         return vec
#
#
# # ============================================================
# # Word2Vec Trainer —— 支持 GPU（若可用），epochs=5
# # ============================================================
# class Word2VecTrainer:
#     def __init__(self, vector_size=200, window=5, min_count=1, workers=4, epochs=5):
#         self.vector_size = vector_size
#         self.window = window
#         self.min_count = min_count
#         self.workers = workers
#         self.epochs = epochs
#         self.model: Word2Vec = None
#
#     def train(self, docs: List["Document"]):
#         import re
#         sentences: List[List[str]] = []
#
#         for d in docs:
#             tokens = getattr(d, "tokens", None)
#             if not tokens:
#                 text = (d.content or "").lower()
#                 tokens = re.findall(r"[a-zA-Z]+", text)
#                 d.tokens = tokens
#
#             if tokens:
#                 sentences.append(tokens)
#
#         if not sentences:
#             raise RuntimeError("Word2VecTrainer.train: 数据集中没有任何非空句子，无法训练 Word2Vec")
#
#         self.model = Word2Vec(
#             vector_size=self.vector_size,
#             window=self.window,
#             min_count=self.min_count,
#             workers=self.workers,
#             sg=1,
#         )
#
#         self.model.build_vocab(sentences)
#         self.model.train(
#             sentences,
#             total_examples=len(sentences),
#             epochs=self.epochs,
#         )
#
#     # --------------------------------
#     # 语义扩展
#     # --------------------------------
#     def semantic_extension(self, query_word: str, top_k=5):
#         if self.model is None:
#             return []
#
#         if query_word not in self.model.wv:
#             return [(query_word, 1.0)]
#
#         sims = self.model.wv.most_similar(query_word, top_k)
#         return [(query_word, 1.0)] + sims
#
#
# # ============================================================
# # CaseSSE System
# # ============================================================
# class CaseSSESystem:
#     def __init__(self, config):
#         self.config = config
#         self.docs: List[Document] = []
#         self.train_docs: List[Document] = []
#
#         self.preprocessor = DataPreprocessor(vocab_size=self.config.secure_dim)
#         self.w2v_trainer = Word2VecTrainer(
#             vector_size=config.w2v_dim,
#             window=5,
#             min_count=1,
#             workers=4,
#             epochs=config.w2v_epochs
#         )
#
#         self.k = 8
#         self.document_clusters: Dict[int, List[Document]] = {}
#         self.cluster_keyword_distribution: Dict[int, Dict[str, float]] = {}
#
#         self.index_builder: SecureIndexBuilder = SecureIndexBuilder(config)
#         self.avl_tree = None
#         self.inverted_index = {}
#
#     # ------------------------------------------------------------
#     # 加载数据集 & 预处理
#     # ------------------------------------------------------------
#     def initialize_system(self, dataset_path: str):
#         # 1. 读 JSON
#         with open(dataset_path, "r", encoding="utf-8") as f:
#             raw = json.load(f)
#
#         self.docs = []
#         for i, item in enumerate(raw):
#             content = (
#                     item.get("text")
#                     or item.get("content")
#                     or item.get("body")
#                     or item.get("email")
#                     or item.get("message")
#                     or ""
#             )
#
#             doc = Document(
#                 doc_id=i,
#                 title=item.get("title", ""),
#                 content=content,
#             )
#             self.docs.append(doc)
#
#         # 2. 先确定训练集
#         train_size = max(100, int(len(self.docs) * self.config.train_ratio))
#         self.train_docs = self.docs[:train_size]
#
#         # 3. 在训练集上构建 vocab，并且给每个 train_doc 填好 tokens
#         self.preprocessor.build_vocabulary(self.train_docs)
#
#         # 4. 用 train_docs (已经有 tokens) 训练 Word2Vec
#         self.w2v_trainer.train(self.train_docs)
#
#         # 5. 用 train_docs 的 TF 分布做聚类
#         doc_vecs = np.array([self.preprocessor.doc_to_vector(d) for d in self.train_docs])
#         self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
#         labels = self.kmeans.fit_predict(doc_vecs)
#
#         # 6. 聚类分组
#         self.document_clusters = {c: [] for c in range(self.k)}
#         for doc, c in zip(self.train_docs, labels):
#             self.document_clusters[c].append(doc)
#
#         # 7. 构造每个类别的关键词概率分布
#         self.cluster_keyword_distribution = {}
#         for c in range(self.k):
#             counts = np.zeros(self.config.secure_dim, dtype=float)
#             for doc in self.document_clusters[c]:
#                 v = self.preprocessor.doc_to_vector(doc)
#                 counts += v
#             if counts.sum() > 0:
#                 counts /= counts.sum()
#
#             kw = {}
#             for w, idx in self.preprocessor.vocabulary.items():
#                 if idx < len(counts):
#                     kw[w] = counts[idx]
#             self.cluster_keyword_distribution[c] = kw
#
#     # ------------------------------------------------------------
#     # 构建安全索引（使用 secure_index.py）
#     # ------------------------------------------------------------
#     def build_secure_indices(self):
#         vocab = self.preprocessor.vocabulary
#         self.avl_tree, self.inverted_index = self.index_builder.build_two_layer_index(
#             self.document_clusters,
#             self.cluster_keyword_distribution,
#             vocab
#         )
#
#     # ------------------------------------------------------------
#     # 构建 QueryTd（2000维查询向量）
#     # ------------------------------------------------------------
#     def build_query_vector(self, query_keywords: List[str]) -> np.ndarray:
#         vec = np.zeros(self.config.secure_dim, dtype=float)
#         for w in query_keywords:
#             if w in self.preprocessor.vocabulary:
#                 idx = self.preprocessor.vocabulary[w]
#                 vec[idx] = 1.0
#
#         if vec.sum() > 0:
#             vec /= vec.sum()
#         return vec
#
#     # ------------------------------------------------------------
#     # 执行一次查询（top-k = 20）
#     # ------------------------------------------------------------
#     def test_secure_search(self, query_keywords: List[str], top_k=20):
#         # 构造 2000维向量
#         vec_m = self.build_query_vector(query_keywords)
#
#         # 生成陷门
#         VTD_1, VTD_2 = self.index_builder.generate_query_trapdoor(vec_m)
#
#         # 执行搜索
#         docs = self.index_builder.secure_search(
#             self.avl_tree,
#             self.inverted_index,
#             VTD_1,
#             VTD_2,
#             top_k=top_k
#         )
#         return docs
#     # ------------------------------------------------------------
#     # 语义搜索演示（用于 debug，不用于实验脚本）
#     # ------------------------------------------------------------
#     def demo_semantic_search(self, query_word: str):
#         print(f"\n=== 语义搜索演示: '{query_word}' ===")
#
#         # 1) Word2Vec 语义扩展
#         extended = self.w2v_trainer.semantic_extension(
#             query_word,
#             top_k=self.config.semantic_extension_count
#         )
#         print("语义扩展结果:")
#         for w, sim in extended:
#             print(f"  {w}: {sim:.3f}")
#
#         # 2) 构造查询关键词列表
#         query_keywords = [w for w, _ in extended]
#
#         # 3) 安全搜索
#         docs = self.test_secure_search(query_keywords, top_k=20)
#         print(f"搜索到的文档ID: {docs[:10]}...")
#
#         if docs:
#             print("相关文档标题:")
#             for d in self.train_docs:
#                 if d.doc_id in docs[:5]:
#                     print(f"  [{d.doc_id}] {d.title}")
#
#     # ------------------------------------------------------------
#     # 从句子中提取关键词（实验脚本使用）
#     # ------------------------------------------------------------
#     def extract_base_keyword(self, text: str) -> str:
#         """
#         简单策略：
#         1. 全部转小写
#         2. 用正则取单词
#         3. 依次寻找落在 vocab 内的词
#         """
#         words = re.findall(r"[a-zA-Z]+", text.lower())
#         for w in words:
#             if w in self.preprocessor.vocabulary:
#                 return w
#         # fallback：第一个单词
#         return words[0] if words else "unknown"
#
#     # ------------------------------------------------------------
#     # 公共接口：对一批 queries 执行搜索（实验脚本使用）
#     # ------------------------------------------------------------
#     def run_queries(self, queries: List[str], plain_results: Dict[str, List[int]],
#                     top_k=20):
#         """
#         返回：
#             total_time_sec        -> 所有查询总耗时
#             num_used              -> 实际参与统计的查询数
#             tp_sum                -> 所有查询交集大小之和
#             plain_sum             -> 所有查询明文结果数之和
#             nonzero_count         -> 交集>0 的查询个数
#             per_query_stats       -> 每条查询的详细统计（列表）
#         """
#
#         total_time = 0.0
#         num_used = 0
#         tp_sum = 0
#         plain_sum = 0
#         nonzero_count = 0
#         per_query_stats = []
#
#         for idx, q in enumerate(queries):
#             if q not in plain_results:
#                 continue
#
#             # 提取基础关键词
#             base = self.extract_base_keyword(q)
#
#             # 语义扩展
#             extended = self.w2v_trainer.semantic_extension(
#                 base,
#                 top_k=self.config.semantic_extension_count
#             )
#             query_keywords = [w for w, _ in extended]
#
#             if not query_keywords:
#                 continue
#
#             plain_ids = plain_results[q]
#             if not plain_ids:
#                 continue
#
#             tk = max(top_k, len(plain_ids))
#
#             # 计时
#             t0 = time.time()
#             enc_ids = self.test_secure_search(query_keywords, top_k=tk)
#             dt = time.time() - t0
#
#             total_time += dt
#             num_used += 1
#
#             # 交集 & 单条查询准确率
#             s_enc = set(enc_ids)
#             s_plain = set(plain_ids)
#             inter = len(s_enc & s_plain)
#
#             plain_cnt = len(s_plain)
#             per_acc = inter / plain_cnt if plain_cnt > 0 else 0.0
#             hit = inter > 0
#             if hit:
#                 nonzero_count += 1
#
#             # 打印每条查询的时间和准确率
#             print(f"  [Query {num_used:02d}] time = {dt:.4f} s, "
#                   f"acc = {per_acc:.4f}, hit = {hit}, ",
#                   f"plain={plain_cnt}, enc={len(enc_ids)}")
#
#             # 保存到 per_query_stats
#             per_query_stats.append({
#                 "query_index": num_used,
#                 "query_text": q,
#                 "base_keyword": base,
#                 "query_keywords": query_keywords,
#                 "time_sec": dt,
#                 "plain_count": plain_cnt,
#                 "enc_count": len(enc_ids),
#                 "intersection": inter,
#                 "accuracy_overlap": per_acc,          # 当前查询的 |∩| / |plain|
#                 "hit_nonzero": hit                    # 交集>0?
#             })
#
#         return total_time, num_used, tp_sum, plain_sum, nonzero_count, per_query_stats
#
#
# # ============================================================
# # CaseSSEConfig —— 参数配置
# # ============================================================
# class CaseSSEConfig:
#     def __init__(self):
#         # 向量维度（你要求固定为 2000）
#         self.secure_dim = 2000
#
#         # 扩展维度 ε（论文常用 10~20）
#         self.epsilon = 10
#
#         # Word2Vec 参数
#         self.w2v_dim = 200
#         self.w2v_epochs = 5     # 你指定 epoch=5
#
#         # 训练集比例
#         self.train_ratio = 1
#
#         # 语义扩展数量（QueryTd）
#         self.semantic_extension_count = 5
#
#
# # ============================================================
# # 工具函数：加载 JSON 文本数据
# # ============================================================
# def load_json(path: str):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
#
#
# # ============================================================
# # 主系统类已经在 Part1 + Part2 中定义完毕
# # （CaseSSESystem）
# # ============================================================
#
# # -------------------------
# # 供外部调用的包装接口
# # -------------------------
# def create_case_sse_system():
#     config = CaseSSEConfig()
#     return CaseSSESystem(config)
#
#
# # -------------------------
# # 简单测试入口（可选）
# # -------------------------
# if __name__ == "__main__":
#     print("CaseSSE system core module. Use from experiment script.")
#
#
import json
import os
import time
import re
import numpy as np
from typing import List, Dict, Any
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import torch

from secure_index import SecureIndexBuilder

#msmsmsms
# ============================================================
# 文档对象
# ============================================================
class Document:
    def __init__(self, doc_id, title: str, content: str):
        # 统一存成字符串，方便和明文结果对齐
        self.doc_id = str(doc_id).strip()
        self.title = title
        self.content = content
        self.tokens: List[str] = []


# ============================================================
# 文本预处理（用于构建 vocab & token 索引）
# ============================================================
class DataPreprocessor:
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.vocabulary: Dict[str, int] = {}     # word -> index(0~1999)
        self.word_freq: Dict[str, int] = {}

    # --------------------------------
    # 基础清洗
    # --------------------------------
    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        words = re.findall(r"[a-zA-Z]+", text)
        return words

    # --------------------------------
    # 构建词表（top-2000 高频词）
    # --------------------------------
    def build_vocabulary(self, docs: List["Document"]):
        """
        同时完成两件事：
        1. 给每个 doc 填好 doc.tokens
        2. 基于这些 tokens 统计词频，截断成 top-2000 词汇表
        """
        freq: Dict[str, int] = {}

        for doc in docs:
            # 如果 tokens 还没填，就先分词
            if not getattr(doc, "tokens", None):
                doc.tokens = self.tokenize(doc.content)

            for w in doc.tokens:
                freq[w] = freq.get(w, 0) + 1

        # 按频率排序，取前 vocab_size 个词
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        top_words = sorted_words[: self.vocab_size]

        self.vocabulary = {w: i for i, (w, _) in enumerate(top_words)}
        self.word_freq = {w: c for (w, c) in top_words}

    # --------------------------------
    # 将文档 tokens 转成词频向量（2000维）
    # --------------------------------
    def doc_to_vector(self, doc: Document) -> np.ndarray:
        # 确保有 tokens
        if not getattr(doc, "tokens", None):
            doc.tokens = self.tokenize(doc.content or "")

        vec = np.zeros(self.vocab_size, dtype=float)
        for w in doc.tokens:
            if w in self.vocabulary:
                vec[self.vocabulary[w]] += 1

        if vec.sum() > 0:
            vec = vec / vec.sum()
        return vec


# ============================================================
# Word2Vec Trainer —— 支持 GPU（若可用），epochs=5
# ============================================================
class Word2VecTrainer:
    def __init__(self, vector_size=200, window=5, min_count=1, workers=4, epochs=5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model: Word2Vec = None

    def train(self, docs: List["Document"]):
        import re
        sentences: List[List[str]] = []

        for d in docs:
            tokens = getattr(d, "tokens", None)
            if not tokens:
                text = (d.content or "").lower()
                tokens = re.findall(r"[a-zA-Z]+", text)
                d.tokens = tokens

            if tokens:
                sentences.append(tokens)

        if not sentences:
            raise RuntimeError("Word2VecTrainer.train: 数据集中没有任何非空句子，无法训练 Word2Vec")

        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1,
        )

        self.model.build_vocab(sentences)
        self.model.train(
            sentences,
            total_examples=len(sentences),
            epochs=self.epochs,
        )

    # --------------------------------
    # 语义扩展
    # --------------------------------
    def semantic_extension(self, query_word: str, top_k=5):
        if self.model is None:
            return []

        if query_word not in self.model.wv:
            return [(query_word, 1.0)]

        sims = self.model.wv.most_similar(query_word, top_k)
        return [(query_word, 1.0)] + sims


# ============================================================
# CaseSSE System
# ============================================================
class CaseSSESystem:
    def __init__(self, config):
        self.config = config
        self.docs: List[Document] = []
        self.train_docs: List[Document] = []

        self.preprocessor = DataPreprocessor(vocab_size=self.config.secure_dim)
        self.w2v_trainer = Word2VecTrainer(
            vector_size=config.w2v_dim,
            window=5,
            min_count=1,
            workers=4,
            epochs=config.w2v_epochs
        )

        self.k = 8
        self.document_clusters: Dict[int, List[Document]] = {}
        self.cluster_keyword_distribution: Dict[int, Dict[str, float]] = {}

        self.index_builder: SecureIndexBuilder = SecureIndexBuilder(config)
        self.avl_tree = None
        self.inverted_index = {}

    # ------------------------------------------------------------
    # 加载数据集 & 预处理
    # ------------------------------------------------------------
    def initialize_system(self, dataset_path: str):
        # 1. 读 JSON
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.docs = []
        for i, item in enumerate(raw):
            # 统一抽 content
            content = (
                    item.get("text")
                    or item.get("content")
                    or item.get("body")
                    or item.get("email")
                    or item.get("message")
                    or ""
            )

            # 优先使用 MS MARCO 的 doc_id
            raw_id = item.get("doc_id")
            if raw_id is None:
                # 兼容老 enron / wiki：有 id 就用 id，没有就退回下标
                raw_id = item.get("id", i)

            doc = Document(
                doc_id=raw_id,  # 这里传入真实 doc_id
                title=item.get("title", ""),
                content=content,
            )
            self.docs.append(doc)

        # 2. 先确定训练集
        train_size = max(100, int(len(self.docs) * self.config.train_ratio))
        self.train_docs = self.docs[:train_size]

        # 3. 在训练集上构建 vocab，并且给每个 train_doc 填好 tokens
        self.preprocessor.build_vocabulary(self.train_docs)

        # 4. 用 train_docs (已经有 tokens) 训练 Word2Vec
        self.w2v_trainer.train(self.train_docs)

        # 5. 用 train_docs 的 TF 分布做聚类
        doc_vecs = np.array([self.preprocessor.doc_to_vector(d) for d in self.train_docs])
        self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(doc_vecs)

        # 6. 聚类分组
        self.document_clusters = {c: [] for c in range(self.k)}
        for doc, c in zip(self.train_docs, labels):
            self.document_clusters[c].append(doc)

        # 7. 构造每个类别的关键词概率分布
        self.cluster_keyword_distribution = {}
        for c in range(self.k):
            counts = np.zeros(self.config.secure_dim, dtype=float)
            for doc in self.document_clusters[c]:
                v = self.preprocessor.doc_to_vector(doc)
                counts += v
            if counts.sum() > 0:
                counts /= counts.sum()

            kw = {}
            for w, idx in self.preprocessor.vocabulary.items():
                if idx < len(counts):
                    kw[w] = counts[idx]
            self.cluster_keyword_distribution[c] = kw

    # ------------------------------------------------------------
    # 构建安全索引（使用 secure_index.py）
    # ------------------------------------------------------------
    def build_secure_indices(self):
        vocab = self.preprocessor.vocabulary
        self.avl_tree, self.inverted_index = self.index_builder.build_two_layer_index(
            self.document_clusters,
            self.cluster_keyword_distribution,
            vocab
        )

    # ------------------------------------------------------------
    # 构建 QueryTd（2000维查询向量）
    # ------------------------------------------------------------
    def build_query_vector(self, query_keywords: List[str]) -> np.ndarray:
        vec = np.zeros(self.config.secure_dim, dtype=float)
        for w in query_keywords:
            if w in self.preprocessor.vocabulary:
                idx = self.preprocessor.vocabulary[w]
                vec[idx] = 1.0

        if vec.sum() > 0:
            vec /= vec.sum()
        return vec

    # ------------------------------------------------------------
    # 执行一次查询（top-k = 20）
    # ------------------------------------------------------------
    def test_secure_search(self, query_keywords: List[str], top_k=20):
        # 构造 2000维向量
        vec_m = self.build_query_vector(query_keywords)

        # 生成陷门
        VTD_1, VTD_2 = self.index_builder.generate_query_trapdoor(vec_m)

        # 执行搜索
        docs = self.index_builder.secure_search(
            self.avl_tree,
            self.inverted_index,
            VTD_1,
            VTD_2,
            top_k=top_k
        )
        return docs
    # ------------------------------------------------------------
    # 语义搜索演示（用于 debug，不用于实验脚本）
    # ------------------------------------------------------------
    def demo_semantic_search(self, query_word: str):
        print(f"\n=== 语义搜索演示: '{query_word}' ===")

        # 1) Word2Vec 语义扩展
        extended = self.w2v_trainer.semantic_extension(
            query_word,
            top_k=self.config.semantic_extension_count
        )
        print("语义扩展结果:")
        for w, sim in extended:
            print(f"  {w}: {sim:.3f}")

        # 2) 构造查询关键词列表
        query_keywords = [w for w, _ in extended]

        # 3) 安全搜索
        docs = self.test_secure_search(query_keywords, top_k=20)
        print(f"搜索到的文档ID: {docs[:10]}...")

        if docs:
            print("相关文档标题:")
            for d in self.train_docs:
                if d.doc_id in docs[:5]:
                    print(f"  [{d.doc_id}] {d.title}")

    # ------------------------------------------------------------
    # 从句子中提取关键词（实验脚本使用）
    # ------------------------------------------------------------
    def extract_base_keyword(self, text: str) -> str:
        """
        简单策略：
        1. 全部转小写
        2. 用正则取单词
        3. 依次寻找落在 vocab 内的词
        """
        words = re.findall(r"[a-zA-Z]+", text.lower())
        for w in words:
            if w in self.preprocessor.vocabulary:
                return w
        # fallback：第一个单词
        return words[0] if words else "unknown"

    # ------------------------------------------------------------
    # 公共接口：对一批 queries 执行搜索（实验脚本使用）
    # ------------------------------------------------------------
    def run_queries(self, queries: List[str], plain_results: Dict[str, List[int]],
                    top_k=20):
        """
        返回：
            total_time_sec        -> 所有查询总耗时
            num_used              -> 实际参与统计的查询数
            tp_sum                -> 所有查询交集大小之和
            plain_sum             -> 所有查询明文结果数之和
            nonzero_count         -> 交集>0 的查询个数
            per_query_stats       -> 每条查询的详细统计（列表）
        """

        total_time = 0.0
        num_used = 0
        tp_sum = 0
        plain_sum = 0
        nonzero_count = 0
        per_query_stats = []

        for idx, q in enumerate(queries):
            if q not in plain_results:
                continue

            # 提取基础关键词
            base = self.extract_base_keyword(q)

            # 语义扩展
            extended = self.w2v_trainer.semantic_extension(
                base,
                top_k=self.config.semantic_extension_count
            )
            query_keywords = [w for w, _ in extended]

            if not query_keywords:
                continue

            plain_ids = plain_results[q]
            if not plain_ids:
                continue

            tk = max(top_k, len(plain_ids))

            # 计时
            t0 = time.time()
            enc_ids = self.test_secure_search(query_keywords, top_k=tk)
            dt = time.time() - t0

            total_time += dt
            num_used += 1

            # 交集 & 单条查询准确率
            s_enc = set(enc_ids)
            s_plain = set(plain_ids)
            inter = len(s_enc & s_plain)

            plain_cnt = len(s_plain)
            per_acc = inter / plain_cnt if plain_cnt > 0 else 0.0
            hit = inter > 0
            if hit:
                nonzero_count += 1

            # 打印每条查询的时间和准确率
            print(f"  [Query {num_used:02d}] time = {dt:.4f} s, "
                  f"acc = {per_acc:.4f}, hit = {hit}, ",
                  f"plain={plain_cnt}, enc={len(enc_ids)}")

            # 保存到 per_query_stats
            per_query_stats.append({
                "query_index": num_used,
                "query_text": q,
                "base_keyword": base,
                "query_keywords": query_keywords,
                "time_sec": dt,
                "plain_count": plain_cnt,
                "enc_count": len(enc_ids),
                "intersection": inter,
                "accuracy_overlap": per_acc,          # 当前查询的 |∩| / |plain|
                "hit_nonzero": hit                    # 交集>0?
            })

        return total_time, num_used, tp_sum, plain_sum, nonzero_count, per_query_stats


# ============================================================
# CaseSSEConfig —— 参数配置
# ============================================================
class CaseSSEConfig:
    def __init__(self):
        # 向量维度（你要求固定为 2000）
        self.secure_dim = 2000

        # 扩展维度 ε（论文常用 10~20）
        self.epsilon = 10

        # Word2Vec 参数
        self.w2v_dim = 200
        self.w2v_epochs = 5     # 你指定 epoch=5

        # 训练集比例
        self.train_ratio = 1

        # 语义扩展数量（QueryTd）
        self.semantic_extension_count = 5


# ============================================================
# 工具函数：加载 JSON 文本数据
# ============================================================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 主系统类已经在 Part1 + Part2 中定义完毕
# （CaseSSESystem）
# ============================================================

# -------------------------
# 供外部调用的包装接口
# -------------------------
def create_case_sse_system():
    config = CaseSSEConfig()
    return CaseSSESystem(config)


# -------------------------
# 简单测试入口（可选）
# -------------------------
if __name__ == "__main__":
    print("CaseSSE system core module. Use from experiment script.")


