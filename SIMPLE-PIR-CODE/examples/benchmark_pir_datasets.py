# #!/usr/bin/env python3
# import sys
# import json
# import time
# import random
# from pathlib import Path
#
# sys.path.append('..')
#
# from simple_pir import SimplePIRProtocol, SimplePIRConfig, SecurityLevel
#
# # 要测试的数据集（你给的 6 个）
# DATASETS = [
#     ("enron_1k",       Path("/root/siton-tmp/data/enron_1k.json")),
#     ("enron_5k",       Path("/root/siton-tmp/data/enron_5k.json")),
#     ("enron_10k",      Path("/root/siton-tmp/data/enron_10k.json")),
#     ("simplewiki_1k",  Path("/root/siton-tmp/data/simplewiki_1k.json")),
#     ("simplewiki_5k",  Path("/root/siton-tmp/data/simplewiki_5k.json")),
#     ("simplewiki_10k", Path("/root/siton-tmp/data/simplewiki_10k.json")),
# ]
#
# # 每个数据集只测 3 个 ID
# NUM_RETRIEVALS_PER_DATASET = 3
#
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
#
#
# def load_documents(json_path: Path):
#     with json_path.open("r", encoding="utf-8") as f:
#         return json.load(f)
#
#
# def build_database_and_mapping(docs):
#     """
#     构建：
#       - database: PIR 用的内容列表
#       - docid_to_index: doc_id -> index
#     """
#     database = []
#     docid_to_index = {}
#
#     for idx, doc in enumerate(docs):
#         # doc_id：如果没给，就用下标
#         doc_id = str(doc.get("id", "")).strip() or str(idx)
#
#         # 内容字段：优先 content，其次 body/text/title
#         content = (
#             doc.get("content")
#             or doc.get("body")
#             or doc.get("text")
#             or doc.get("title")
#             or ""
#         )
#
#         database.append(content)
#         docid_to_index[doc_id] = idx
#
#     return database, docid_to_index
#
#
# def init_pir(database):
#     config = SimplePIRConfig(SecurityLevel.MEDIUM)
#     config.enable_preprocessing = True
#     pir = SimplePIRProtocol(database, config)
#     return pir
#
#
# def choose_doc_ids(docid_to_index, k: int):
#     """选出要测试的 doc_id，这里简单用随机选"""
#     all_ids = list(docid_to_index.keys())
#     if len(all_ids) <= k:
#         return all_ids
#     return random.sample(all_ids, k)
#
#
# def benchmark_dataset(dataset_name: str, json_path: Path):
#     docs = load_documents(json_path)
#     database, docid_to_index = build_database_and_mapping(docs)
#     pir = init_pir(database)
#
#     test_doc_ids = choose_doc_ids(docid_to_index, NUM_RETRIEVALS_PER_DATASET)
#
#     overall_times = []        # doc_id -> index + PIR
#     protocol_total_times = [] # PIR 内部 total_time
#
#     for doc_id in test_doc_ids:
#         # doc_id -> index
#         t0 = time.perf_counter()
#         index = docid_to_index[doc_id]
#         t1 = time.perf_counter()
#
#         # PIR 协议
#         result = pir.retrieve_item(index)
#         t2 = time.perf_counter()
#
#         if not result["retrieval_successful"]:
#             continue
#
#         overall_times.append(t2 - t0)
#         protocol_total_times.append(result["performance_breakdown"]["total_time"])
#
#     if not overall_times:
#         print(f"{dataset_name}: no successful retrievals")
#         return
#
#     avg_overall = sum(overall_times) / len(overall_times)
#     avg_protocol = sum(protocol_total_times) / len(protocol_total_times)
#
#     # 👉 只输出你关心的时间
#     print(
#         f"{dataset_name}: "
#         f"avg_overall_time={avg_overall:.6f}s, "
#         f"avg_protocol_time={avg_protocol:.6f}s, "
#         f"n={len(overall_times)}"
#     )
#
#
# def main():
#     for name, path in DATASETS:
#         benchmark_dataset(name, path)
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import sys
import json
import time
import random
from pathlib import Path

sys.path.append('..')

from simple_pir import SimplePIRProtocol, SimplePIRConfig, SecurityLevel

# 要测试的数据集（新增 msmarco_*）
DATASETS = [
    ("enron_1k",       Path("/root/siton-tmp/data/enron_1k.json")),
    ("enron_5k",       Path("/root/siton-tmp/data/enron_5k.json")),
    ("enron_10k",      Path("/root/siton-tmp/data/enron_10k.json")),
    ("simplewiki_1k",  Path("/root/siton-tmp/data/simplewiki_1k.json")),
    ("simplewiki_5k",  Path("/root/siton-tmp/data/simplewiki_5k.json")),
    ("simplewiki_10k", Path("/root/siton-tmp/data/simplewiki_10k.json")),

    # ✅ 新增 MSMARCO
    ("msmarco_1k",     Path("/root/siton-tmp/data/msmarco_1k.json")),
    ("msmarco_5k",     Path("/root/siton-tmp/data/msmarco_5k.json")),
    ("msmarco_10k",    Path("/root/siton-tmp/data/msmarco_10k.json")),
]

# 每个数据集只测 3 个 ID
NUM_RETRIEVALS_PER_DATASET = 3

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def load_documents(json_path: Path):
    """
    返回一个 list[dict]。
    兼容两类常见结构：
      1) 直接是 list
      2) dict 包一层，例如 {"data":[...]} / {"docs":[...]} / {"documents":[...]} / {"items":[...]}
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for k in ["data", "docs", "documents", "items", "passages"]:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break

    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON format in {json_path}: root is {type(data)}")

    return data


def build_database_and_mapping(docs):
    """
    构建：
      - database: PIR 用的内容列表
      - docid_to_index: doc_id -> index
    """
    database = []
    docid_to_index = {}

    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            # 如果遇到非 dict（极少数情况），跳过或转字符串
            doc_id = str(idx)
            content = str(doc)
        else:
            # doc_id：如果没给，就用下标
            doc_id = str(doc.get("id", "")).strip() or str(idx)

            # 内容字段：优先 content，其次 body/text/title
            content = (
                doc.get("content")
                or doc.get("body")
                or doc.get("text")
                or doc.get("title")
                or ""
            )

        database.append(content)

        # 如遇到重复 doc_id，保留第一次，后续跳过映射，避免覆盖
        if doc_id not in docid_to_index:
            docid_to_index[doc_id] = idx

    return database, docid_to_index


def init_pir(database):
    config = SimplePIRConfig(SecurityLevel.MEDIUM)
    config.enable_preprocessing = True
    pir = SimplePIRProtocol(database, config)
    return pir


def choose_doc_ids(docid_to_index, k: int):
    """选出要测试的 doc_id，这里简单用随机选"""
    all_ids = list(docid_to_index.keys())
    if len(all_ids) <= k:
        return all_ids
    return random.sample(all_ids, k)


def benchmark_dataset(dataset_name: str, json_path: Path):
    docs = load_documents(json_path)
    database, docid_to_index = build_database_and_mapping(docs)
    pir = init_pir(database)

    test_doc_ids = choose_doc_ids(docid_to_index, NUM_RETRIEVALS_PER_DATASET)

    overall_times = []        # doc_id -> index + PIR
    protocol_total_times = [] # PIR 内部 total_time

    for doc_id in test_doc_ids:
        # doc_id -> index
        t0 = time.perf_counter()
        index = docid_to_index[doc_id]
        t1 = time.perf_counter()

        # PIR 协议
        result = pir.retrieve_item(index)
        t2 = time.perf_counter()

        if not result.get("retrieval_successful", False):
            continue

        overall_times.append(t2 - t0)

        perf = result.get("performance_breakdown", {})
        protocol_total_times.append(perf.get("total_time", 0.0))

    if not overall_times:
        print(f"{dataset_name}: no successful retrievals")
        return

    avg_overall = sum(overall_times) / len(overall_times)
    avg_protocol = sum(protocol_total_times) / len(protocol_total_times) if protocol_total_times else 0.0

    # 👉 只输出你关心的时间
    print(
        f"{dataset_name}: "
        f"avg_overall_time={avg_overall:.6f}s, "
        f"avg_protocol_time={avg_protocol:.6f}s, "
        f"n={len(overall_times)}"
    )


def main():
    for name, path in DATASETS:
        benchmark_dataset(name, path)


if __name__ == "__main__":
    main()
