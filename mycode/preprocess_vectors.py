#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_vectors.py
统一处理 Enron 与 Wiki 数据集的文本向量编码
- 自动识别字段结构:
  Enron: {"id": ..., "content": ...}
  Wiki:  {"title": ..., "content": ...}
- 输出 Sentence-BERT 向量和对应 meta 文件
"""

import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import Timer, save_json

# ====================== 数据路径配置 ======================
DATASET_PATHS = {
    # Enron 数据集
    'enron_1k': r'/root/siton-tmp/data/enron_1k.json',
    'enron_5k': r'/root/siton-tmp/data/enron_5k.json',
    'enron_10k': r'/root/siton-tmp/data/enron_10k.json',

    # Wiki 数据集
    'wiki_1k': r'/root/siton-tmp/data/simplewiki_1k.json',
    'wiki_5k': r'/root/siton-tmp/data/simplewiki_5k.json',
    'wiki_10k': r'/root/siton-tmp/data/simplewiki_10k.json',
}

# ====================== 输出路径 ======================
OUT_VEC_DIR = r'../outputs/vectors'
OUT_META_DIR = r'../outputs/meta'
MODEL_NAME = '/root/siton-tmp/all-MiniLM-L6-v2'

# ====================== 编码函数 ======================
def load_documents(path):
    """根据文件内容结构自动解析 Enron 或 Wiki"""
    with open(path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    texts, ids, titles = [], [], []

    for i, d in enumerate(docs):
        # Enron 格式: {"id": ..., "content": ...}
        if "content" in d and "id" in d:
            ids.append(str(d["id"]))
            titles.append(f"enron_doc_{i}")
            texts.append(str(d["content"]))

        # Wiki 格式: {"title": ..., "content": ...}
        elif "content" in d and "title" in d:
            ids.append(str(i))
            titles.append(str(d["title"]))
            texts.append(str(d["content"]))

        # 其他未知格式：跳过
        else:
            continue

    return ids, titles, texts


def encode_dataset(name: str, model):
    """对单个数据集进行编码"""
    os.makedirs(OUT_VEC_DIR, exist_ok=True)
    os.makedirs(OUT_META_DIR, exist_ok=True)

    path = DATASET_PATHS[name]
    if not os.path.exists(path):
        print(f"[skip] 数据文件不存在: {path}")
        return

    print(f"\n[encode_dataset] 开始编码 {name} ...")

    ids, titles, texts = load_documents(path)
    if len(texts) == 0:
        print(f"[warning] 数据集 {name} 为空或格式错误。")
        return

    timer = Timer(); timer.start()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    elapsed = timer.stop()

    # 保存 npy 向量文件
    np.save(os.path.join(OUT_VEC_DIR, f'{name}_vecs.npy'), embeddings)

    # 保存 meta 文件
    meta = [
        {'id': ids[i], 'title': titles[i], 'text': texts[i]}
        for i in range(len(ids))
    ]
    save_json(os.path.join(OUT_META_DIR, f'{name}_meta.json'), meta)

    print(f"[done] {name}: 向量形状 {embeddings.shape}, 耗时 {elapsed:.2f}s")


# ====================== 主函数 ======================
if __name__ == '__main__':
    # 一次加载模型，提高性能
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ 已加载模型: {MODEL_NAME}\n")

    # 遍历所有存在的数据文件
    for name, path in DATASET_PATHS.items():
        if os.path.exists(path):
            encode_dataset(name, model)
        else:
            print(f"[skip] {name} 不存在，跳过。")

    print("\n✅ 所有数据集编码完成，结果已保存到 ../outputs/")

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# preprocess_vectors.py
# 统一处理 MS MARCO 子集的文本向量编码
#
# - 数据集格式:
#   MS MARCO: {"doc_id": ..., "text": ...}
#
# - 输出:
#   Sentence-BERT 向量 (npy) 和对应 meta 文件 (json)
# """
#
# import os
# import json
# import numpy as np
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from utils import Timer, save_json
#
# # ====================== 数据路径配置 ======================
#
# DATASET_PATHS = {
#     "msmarco_1k": "/root/siton-tmp/data/msmarco_1k.json",
#     "msmarco_5k": "/root/siton-tmp/data/msmarco_5k.json",
#     "msmarco_10k": "/root/siton-tmp/data/msmarco_10k.json",
# }
#
# # ====================== 输出路径 ======================
#
# OUT_VEC_DIR = "/root/siton-tmp/outputs/vectors"
# OUT_META_DIR = "/root/siton-tmp/outputs/meta"
# MODEL_NAME = "/root/siton-tmp/all-MiniLM-L6-v2"
#
#
# # ====================== 编码函数 ======================
#
# def load_documents(path):
#     """
#     解析 MS MARCO 格式:
#       {"doc_id": ..., "text": ...}
#     """
#     with open(path, "r", encoding="utf-8") as f:
#         docs = json.load(f)
#
#     texts, ids, titles = [], [], []
#
#     for i, d in enumerate(docs):
#         # MS MARCO 格式
#         if "text" in d and "doc_id" in d:
#             ids.append(str(d["doc_id"]))
#             titles.append(f"msmarco_doc_{i}")
#             texts.append(str(d["text"]))
#         else:
#             # 其他未知格式：跳过
#             continue
#
#     return ids, titles, texts
#
#
# def encode_dataset(name: str, model):
#     """对单个数据集进行编码并保存向量与 meta"""
#     os.makedirs(OUT_VEC_DIR, exist_ok=True)
#     os.makedirs(OUT_META_DIR, exist_ok=True)
#
#     path = DATASET_PATHS[name]
#     if not os.path.exists(path):
#         print(f"[skip] 数据文件不存在: {path}")
#         return
#
#     print(f"\n[encode_dataset] 开始编码 {name} ...")
#     print(f"[info] 数据路径: {path}")
#
#     ids, titles, texts = load_documents(path)
#     if len(texts) == 0:
#         print(f"[warning] 数据集 {name} 为空或格式错误。")
#         return
#
#     timer = Timer()
#     timer.start()
#     embeddings = model.encode(
#         texts,
#         batch_size=64,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=True,
#     )
#     elapsed = timer.stop()
#
#     # 保存 npy 向量文件
#     vec_path = os.path.join(OUT_VEC_DIR, f"{name}_vecs.npy")
#     np.save(vec_path, embeddings)
#
#     # 保存 meta 文件
#     meta = [
#         {"id": ids[i], "title": titles[i], "text": texts[i]}
#         for i in range(len(ids))
#     ]
#     meta_path = os.path.join(OUT_META_DIR, f"{name}_meta.json")
#     save_json(meta_path, meta)
#
#     print(
#         f"[done] {name}: 向量形状 {embeddings.shape}, 耗时 {elapsed:.2f}s\n"
#         f"       vecs -> {vec_path}\n"
#         f"       meta -> {meta_path}"
#     )
#
#
# # ====================== 主函数 ======================
#
# if __name__ == "__main__":
#     # 一次加载模型，提高性能
#     model = SentenceTransformer(MODEL_NAME)
#     print(f"✅ 已加载模型: {MODEL_NAME}\n")
#
#     # 遍历所有存在的数据文件
#     for name, path in DATASET_PATHS.items():
#         if os.path.exists(path):
#             encode_dataset(name, model)
#         else:
#             print(f"[skip] {name} 数据文件不存在: {path}")
#
#     print("\n✅ 所有 MS MARCO 子集编码完成，结果已保存到 /root/siton-tmp/outputs/")
