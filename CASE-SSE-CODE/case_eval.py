# import json
# import os
# import time
# import numpy as np
# from typing import Dict, List, Any
#
# from case_sse_system import CaseSSESystem, CaseSSEConfig
#
#
# # ============================================================
# # 加载明文结果（支持多种格式）
# # ============================================================
# import re
#
# def load_plain_results(path: str) -> Dict[str, List[int]]:
#     """
#     解析你提供的明文结果格式（query: [ {id:"58.", ...}, ... ]）
#     或其他格式，自动提取 doc_id。
#     """
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     out = {}
#     if isinstance(data, dict):
#         for q, arr in data.items():
#             ids = []
#             if isinstance(arr, list):
#                 for item in arr:
#                     if isinstance(item, dict) and "id" in item:
#                         sid = str(item["id"])
#                         m = re.search(r"\d+", sid)
#                         if m:
#                             ids.append(int(m.group()))
#                     else:
#                         try:
#                             ids.append(int(item))
#                         except:
#                             pass
#             out[q] = ids
#         return out
#
#     # list 格式 fallback
#     if isinstance(data, list):
#         for obj in data:
#             if not isinstance(obj, dict):
#                 continue
#             q = obj.get("query")
#             raw = obj.get("doc_ids", [])
#             ids = []
#             for r in raw:
#                 m = re.search(r"\d+", str(r))
#                 if m:
#                     ids.append(int(m.group()))
#             if q:
#                 out[q] = ids
#         return out
#
#     raise ValueError("Unsupported plain result format: " + path)
#
#
# # ============================================================
# # Enron / Wiki 查询（按你给定的版本）
# # ============================================================
#
# ENRON_QUERIES = [
#     "What meetings are scheduled?",
#     "Tell me about energy trading",
#     "What contracts were discussed?",
#     "What are the price forecasts?",
#     "What reports need analysis?",
#     "What projects are in development?",
#     "What companies are involved?",
#     "What emails need attention?",
#     "What conference calls are planned?",
#     "What financial information is available?",
#     "Emails about SEC strategy meetings",
#     "Messages mentioning building access or badges",
#     "HR newsletters on labor or employment policy",
#     "Forwards with BNA Daily Labor Report content",
#     "Memos on minimum wage or unemployment issues",
#     "Emails discussing union negotiations or wage increases",
#     "Messages about post-9/11 employment impacts",
#     "Notes on federal worker discrimination or whistleblower cases",
#     "Emails that list multiple labor news headlines",
#     "Messages sharing external news links with login info",
#     "Internal calendar or on-call notification emails",
#     "Emails between facilities or admin staff about office locations",
#     "Messages referencing ILO or international labor standards",
#     "Forwards about appointments to U.S. labor-related posts",
#     "Emails on benefit or donation program changes",
#     "Threads with multiple HR recipients in one blast",
#     "Messages mentioning airport security or related legislation",
#     "Emails summarizing congressional labor actions",
#     "Messages about court rulings on workplace drug testing",
#     "Long digest-style labor and employment updates",
# ]
#
# WIKI_QUERIES = [
#     # "What is the history of artificial intelligence?",
#     # "Tell me about the structure of the human brain.",
#     # "What are the major events of World War II?",
#     # "Explain the theory of evolution by Charles Darwin.",
#     # "What are the moons of Jupiter?",
#     "Describe the process of photosynthesis.",
#     # "Who discovered gravity?",
#     # "What are the causes of climate change?",
#     "Explain quantum mechanics basics.",
#     # "Tell me about the culture of ancient Egypt.",
#     # "April month overview in the Gregorian calendar",
#     # "Etymology or origin of the name April",
#     # "April holidays and observances worldwide",
#     # "Seasonal description of April in both hemispheres",
#     # "Movable Christian feasts that fall in April",
#     # "Sayings or phrases about April weather",
#     # "Historical events that happened in April",
#     # "April cultural festivals in Europe or Asia",
#     # "Sports or major events usually held in April",
#     # "August month overview and calendar facts",
#     # "Etymology or origin of the name August",
#     # "August national or religious holidays",
#     # "August historical events in the 20th century",
#     "Definition of art as human creative activity",
#     # "Categories of art such as visual or performing",
#     # "Discussion of art versus design",
#     # "Short history outline of art across eras",
#     # "Examples of everyday objects treated as art",
#     # "Comparison of April seasons across hemispheres",
#     # "August cultural festivals and public holidays",
# ]
#
#
# # ============================================================
# # 数据集配置（你提供的路径）
# # ============================================================
#
# DATASETS = [
#     {
#         "name": "enron_1k",
#         "data_path": "/root/siton-tmp/data/enron_1k.json",
#         "plain_path": "/root/siton-tmp/outputs/plain_results/enron_1k.json",
#         "queries": ENRON_QUERIES,
#     },
#     {
#         "name": "enron_5k",
#         "data_path": "/root/siton-tmp/data/enron_5k.json",
#         "plain_path": "/root/siton-tmp/outputs/plain_results/enron_5k.json",
#         "queries": ENRON_QUERIES,
#     },
#     {
#         "name": "enron_10k",
#         "data_path": "/root/siton-tmp/data/enron_10k.json",
#         "plain_path": "/root/siton-tmp/outputs/plain_results/enron_10k.json",
#         "queries": ENRON_QUERIES,
#     },
#     {
#         "name": "wiki_1k",
#         "data_path": "/root/siton-tmp/data/simplewiki_1k.json",
#         "plain_path": "/root/siton-tmp/outputs/plain_results/wiki_1k.json",
#         "queries": WIKI_QUERIES,
#     },
#     {
#         "name": "wiki_5k",
#         "data_path": "/root/siton-tmp/data/simplewiki_5k.json",
#         "plain_path": "/root/siton-tmp/outputs/plain_results/wiki_5k.json",
#         "queries": WIKI_QUERIES,
#     },
#     {
#         "name": "wiki_10k",
#         "data_path": "/root/siton-tmp/data/simplewiki_10k.json",
#         "plain_path": "/root/siton-tmp/outputs/plain_results/wiki_10k.json",
#         "queries": WIKI_QUERIES,
#     },
# ]
#
#
# # ============================================================
# # 主实验流程
# # ============================================================
#
# def run_single_experiment(cfg) -> Dict[str, Any]:
#     name = cfg["name"]
#     data_path = cfg["data_path"]
#     plain_path = cfg["plain_path"]
#     queries = cfg["queries"]
#
#     print("\n" + "=" * 80)
#     print(f"[数据集] {name}")
#     print("=" * 80)
#
#     system = CaseSSESystem(CaseSSEConfig())
#
#     # -------------------------------------
#     # 1. 初始化系统
#     # -------------------------------------
#     t0 = time.time()
#     system.initialize_system(data_path)
#     init_time = time.time() - t0
#     print(f"[阶段 1] 初始化时间: {init_time:.3f} 秒")
#
#     # -------------------------------------
#     # 2. 构建安全索引
#     # -------------------------------------
#     t1 = time.time()
#     system.build_secure_indices()
#     index_time = time.time() - t1
#     print(f"[阶段 2] 索引构建时间: {index_time:.3f} 秒")
#
#     # -------------------------------------
#     # 3. 加载明文结果
#     # -------------------------------------
#     plain_results = load_plain_results(plain_path)
#
#     # -------------------------------------
#     # 4. 搜索 + 准确率
#     # -------------------------------------
#     (total_time,
#      num_used,
#      tp_sum,
#      plain_sum,
#      nonzero_count,
#      per_query_stats) = system.run_queries(
#         queries,
#         plain_results,
#         top_k=40
#     )
#
#     avg_time = total_time / num_used if num_used > 0 else 0.0
#     accuracy_overlap = tp_sum / plain_sum if plain_sum > 0 else 0.0
#     accuracy_nonzero = nonzero_count / num_used if num_used > 0 else 0.0
#
#     print(f"[阶段 3] 平均搜索时间: {avg_time:.4f} 秒/查询")
#     print(f"[阶段 3] 准确率1 (|∩|/|plain|): {accuracy_overlap:.4f}")
#     print(f"[阶段 3] 准确率2 (命中查询数/总查询数): {accuracy_nonzero:.4f}")
#     print(f"[阶段 3] 有效查询个数: {num_used}")
#
#     return {
#         "dataset": name,
#         "init_time": init_time,
#         "index_time": index_time,
#         "avg_search_time": avg_time,
#         "accuracy_overlap": accuracy_overlap,
#         "accuracy_nonzero": accuracy_nonzero,
#         "num_queries": num_used,
#         "num_nonzero": nonzero_count,
#         "tp_sum": tp_sum,
#         "plain_sum": plain_sum,
#         "queries": per_query_stats,  # 每条查询的详细结果
#     }
#
#
# # ============================================================
# # 主入口
# # ============================================================
#
# def main():
#     all_results = []
#
#     for cfg in DATASETS:
#         res = run_single_experiment(cfg)
#         all_results.append(res)
#
#     print("\n" + "=" * 80)
#     print("全部实验完成（6 个数据集）")
#     print("=" * 80)
#
#     for r in all_results:
#         print(f"\n数据集: {r['dataset']}")
#         print(f"  初始化时间       : {r['init_time']:.3f} 秒")
#         print(f"  索引构建时间     : {r['index_time']:.3f} 秒")
#         print(f"  平均搜索时间     : {r['avg_search_time']:.4f} 秒/查询")
#         # print(f"  准确率1 (|∩|/|plain|) : {r['accuracy_overlap']:.4f} "
#         #       f"(={r['tp_sum']}/{r['plain_sum']})")
#         print(f"  准确率: {r['accuracy_nonzero']:.4f} "
#               f"(={r['num_nonzero']}/{r['num_queries']})")
#         print(f"  有效查询数       : {r['num_queries']}")
#
#     # ===== 保存 JSON 结果 =====
#     output_dir = "/root/siton-tmp/CASE-SSE-CODE/case_sse_output"
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, "case_sse_results.json")
#
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(all_results, f, ensure_ascii=False, indent=2)
#
#     print(f"\n所有结果已保存到: {output_path}")
#
#
# if __name__ == "__main__":
#     main()
import json
import os
import time
import numpy as np
from typing import Dict, List, Any

from case_sse_system import CaseSSESystem, CaseSSEConfig

# ============================================================
# 加载明文结果（支持多种格式）
# ============================================================
import re

def load_plain_results(path: str) -> Dict[str, List[str]]:
    """
    解析明文结果格式（query: [ {id:"xxx", ...}, ... ]）
    直接使用 item["id"] 的字符串形式作为 doc_id。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, List[str]] = {}

    # dict 格式：{ query: [ { "id": ... }, ... ], ... }
    if isinstance(data, dict):
        for q, arr in data.items():
            ids: List[str] = []
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict) and "id" in item:
                        # 直接保留原始 id 字符串
                        ids.append(str(item["id"]).strip())
                    else:
                        # 如果是简单列表，就也转成字符串
                        ids.append(str(item).strip())
            out[q] = ids
        return out

    # list 格式 fallback：[{ "query": ..., "doc_ids": [...] }, ...]
    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            q = obj.get("query")
            raw = obj.get("doc_ids", [])
            ids: List[str] = [str(r).strip() for r in raw]
            if q:
                out[q] = ids
        return out

    raise ValueError("Unsupported plain result format: " + path)



# ============================================================
# MS MARCO 查询（15 条）
# ============================================================

MSMARCO_QUERIES = [
    "How do you use the Stefan-Boltzmann law to calculate the radius of a star such as Rigel from its luminosity and surface temperature?",
    "What developmental milestones and typical behaviors should you expect from an 8 year old child at home and at school?",
    "What are the symptoms of a head lice infestation and how can you check for lice, eggs, and nits on a child's scalp?",
    "What special features does the Burj Khalifa in Dubai have and why was it renamed from Burj Dubai?",
    "What kinds of homes and land are for sale near La Grange, California, and what are their typical sizes and prices?",
    "What are the main characteristics, temperament, and exercise needs of the Dogo Argentino dog breed?",
    "How are custom safety nets used in industry and what kinds of clients and applications does a company like US Netting serve?",
    "What are effective ways to remove weeds from a garden and prevent them from coming back?",
    "How common is urinary incontinence in the United States, what can cause it, and is it just a normal part of aging?",
    "How did President Franklin D. Roosevelt prepare the United States for World War II before Pearl Harbor while the country was still isolationist?",
    "If you have multiple sclerosis and difficulty swallowing pills, is it safe to crush Valium and other medications to make them easier to swallow?",
    "What strategies can help you get better results when dealing with customer service representatives at cable companies or airlines?",
    "In Spanish, what does the word 'machacado' mean and how is the verb 'machacar' used in different contexts?",
    "When building a concrete path, how should you design and support plywood formwork so that it is strong enough and keeps the concrete in place?",
    "Why do people join political parties, and which political party did U.S. presidents Woodrow Wilson and Herbert Hoover belong to?",
]


# ============================================================
# 数据集配置：MS MARCO 1k / 5k / 10k
# ============================================================

DATASETS = [
    {
        "name": "msmarco_1k",
        "data_path": "/root/siton-tmp/data/msmarco_1k.json",
        "plain_path": "/root/siton-tmp/outputs/plain_results/msmarco_1k.json",
        "queries": MSMARCO_QUERIES,
    },
    {
        "name": "msmarco_5k",
        "data_path": "/root/siton-tmp/data/msmarco_5k.json",
        "plain_path": "/root/siton-tmp/outputs/plain_results/msmarco_5k.json",
        "queries": MSMARCO_QUERIES,
    },
    {
        "name": "msmarco_10k",
        "data_path": "/root/siton-tmp/data/msmarco_10k.json",
        "plain_path": "/root/siton-tmp/outputs/plain_results/msmarco_10k.json",
        "queries": MSMARCO_QUERIES,
    },
]


# ============================================================
# 主实验流程
# ============================================================

def run_single_experiment(cfg) -> Dict[str, Any]:
    name = cfg["name"]
    data_path = cfg["data_path"]
    plain_path = cfg["plain_path"]
    queries = cfg["queries"]

    print("\n" + "=" * 80)
    print(f"[数据集] {name}")
    print("=" * 80)

    system = CaseSSESystem(CaseSSEConfig())

    # -------------------------------------
    # 1. 初始化系统
    # -------------------------------------
    t0 = time.time()
    system.initialize_system(data_path)
    init_time = time.time() - t0
    print(f"[阶段 1] 初始化时间: {init_time:.3f} 秒")

    # -------------------------------------
    # 2. 构建安全索引
    # -------------------------------------
    t1 = time.time()
    system.build_secure_indices()
    index_time = time.time() - t1
    print(f"[阶段 2] 索引构建时间: {index_time:.3f} 秒")

    # -------------------------------------
    # 3. 加载明文结果
    # -------------------------------------
    plain_results = load_plain_results(plain_path)

    # -------------------------------------
    # 4. 搜索 + 准确率
    # -------------------------------------
    (
        total_time,
        num_used,
        tp_sum,
        plain_sum,
        nonzero_count,
        per_query_stats,
    ) = system.run_queries(
        queries,
        plain_results,
        top_k=40,
    )

    avg_time = total_time / num_used if num_used > 0 else 0.0
    accuracy_overlap = tp_sum / plain_sum if plain_sum > 0 else 0.0
    accuracy_nonzero = nonzero_count / num_used if num_used > 0 else 0.0

    print(f"[阶段 3] 平均搜索时间: {avg_time:.4f} 秒/查询")
    print(f"[阶段 3] 准确率1 (|∩|/|plain|): {accuracy_overlap:.4f}")
    print(f"[阶段 3] 准确率2 (命中查询数/总查询数): {accuracy_nonzero:.4f}")
    print(f"[阶段 3] 有效查询个数: {num_used}")

    return {
        "dataset": name,
        "init_time": init_time,
        "index_time": index_time,
        "avg_search_time": avg_time,
        "accuracy_overlap": accuracy_overlap,
        "accuracy_nonzero": accuracy_nonzero,
        "num_queries": num_used,
        "num_nonzero": nonzero_count,
        "tp_sum": tp_sum,
        "plain_sum": plain_sum,
        "queries": per_query_stats,  # 每条查询的详细结果
    }


# ============================================================
# 主入口
# ============================================================

def main():
    all_results = []

    for cfg in DATASETS:
        res = run_single_experiment(cfg)
        all_results.append(res)

    print("\n" + "=" * 80)
    print("全部实验完成（3 个数据集）")
    print("=" * 80)

    for r in all_results:
        print(f"\n数据集: {r['dataset']}")
        print(f"  初始化时间       : {r['init_time']:.3f} 秒")
        print(f"  索引构建时间     : {r['index_time']:.3f} 秒")
        print(f"  平均搜索时间     : {r['avg_search_time']:.4f} 秒/查询")
        print(f"  准确率: {r['accuracy_nonzero']:.4f} "
              f"(={r['num_nonzero']}/{r['num_queries']})")
        print(f"  有效查询数       : {r['num_queries']}")



if __name__ == "__main__":
    main()
