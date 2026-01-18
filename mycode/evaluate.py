# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import os
# import time
# import json
#
# from generate_trapdoor import generate_trapdoor
# from secure_search import (
#     secure_search,
#     get_cluster_docs,
#     merge_and_decrypt_docs,
# )
#
# # ===================== 配置 =====================
# # DATASETS = ["enron_1k", "enron_5k", "enron_10k"]
# DATASETS = ["wiki_1k", "wiki_5k", "wiki_10k"]
#
# # QUERIES = [
# #     # "What meetings are scheduled?",
# #     # "Tell me about energy trading",
# #     # "What contracts were discussed?",
# #     # "What are the price forecasts?",
# #     # "What reports need analysis?",
# #     # "What projects are in development?",
# #     # "What companies are involved?",
# #     "What emails need attention?",
# #     "What conference calls are planned?",
# #     # "What financial information is available?",
# #     # "Emails about SEC strategy meetings",
# #     # "Messages mentioning building access or badges",
# #     # "HR newsletters on labor or employment policy",
# #     # "Forwards with BNA Daily Labor Report content",
# #     # "Memos on minimum wage or unemployment issues",
# #     # "Emails discussing union negotiations or wage increases",
# #     "Messages about post-9/11 employment impacts",
# #     # "Notes on federal worker discrimination or whistleblower cases",
# #     # "Emails that list multiple labor news headlines",
# #     # "Messages sharing external news links with login info",
# #     "Internal calendar or on-call notification emails",
# #     "Emails between facilities or admin staff about office locations",
# #     # "Messages referencing ILO or international labor standards",
# #     # "Forwards about appointments to U.S. labor-related posts",
# #     # "Emails on benefit or donation program changes",
# #     # "Threads with multiple HR recipients in one blast",
# #     # "Messages mentioning airport security or related legislation",
# #     # "Emails summarizing congressional labor actions",
# #     # "Messages about court rulings on workplace drug testing",
# #     # "Long digest-style labor and employment updates",
# # ]
# QUERIES = [
#     # "What is the history of artificial intelligence?",
#     # "Tell me about the structure of the human brain.",
#     # "What are the major events of World War II?",
#     # "Explain the theory of evolution by Charles Darwin.",
#     # "What are the moons of Jupiter?",
#     # "Describe the process of photosynthesis.",
#     # "Who discovered gravity?",
#     # "What are the causes of climate change?",
#     "Explain quantum mechanics basics.",
#     # "Tell me about the culture of ancient Egypt.",
#     # "April month overview in the Gregorian calendar",
#     # "Etymology or origin of the name April",
#     # "April holidays and observances worldwide",
#     # "Seasonal description of April in both hemispheres",
#     # "Movable Christian feasts that fall in April",
#     "Sayings or phrases about April weather",
#     # "Historical events that happened in April",
#     # "April cultural festivals in Europe or Asia",
#     "Sports or major events usually held in April",
#     # "August month overview and calendar facts",
#     # "Etymology or origin of the name August",
#     # "August national or religious holidays",
#     # "August historical events in the 20th century",
#     # "Definition of art as human creative activity",
#     "Categories of art such as visual or performing",
#     # "Discussion of art versus design",
#     # "Short history outline of art across eras",
#     # "Examples of everyday objects treated as art",
#     # "Comparison of April seasons across hemispheres",
#     # "August cultural festivals and public holidays"
# ]
#
# RESULTS_DIR = r"../outputs/plain_results"
# OUTPUT_DIR = r"../outputs/eval"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#
# # ===================== 工具函数 =====================
# def load_plain_results(dataset):
#     path = os.path.join(RESULTS_DIR, f"{dataset}.json")
#     if not os.path.exists(path):
#         print(f"[WARN] 明文结果文件未找到: {path}")
#         return {}
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
#
#
# def compute_accuracy(secure_ids, plain_ids):
#     if not plain_ids:
#         return 0.0
#     secure_set = set(str(s).strip() for s in secure_ids)
#     plain_set = set(str(p).strip() for p in plain_ids)
#     inter = len(secure_set & plain_set)
#     return inter / len(secure_set) if secure_set else 0.0
#
#
# # ===================== 主流程 =====================
# def run_experiment():
#     final_output = {}
#
#     for dataset in DATASETS:
#         print("\n" + "=" * 100)
#         print(f"🔹 Evaluating dataset: {dataset}")
#         print("=" * 100)
#
#         plain_results = load_plain_results(dataset)
#         if not plain_results:
#             print(f"[WARN] 跳过 {dataset}，未找到明文结果文件。")
#             continue
#         total_trap = 0.0
#         total_time = 0.0
#         total_acc = 0.0
#         valid_queries = 0
#         perfect_count = 0
#         per_query_results = {}
#
#         for query in QUERIES:
#             if query not in plain_results:
#                 print(f"[WARN] {query} 未在明文结果中找到，跳过。")
#                 continue
#
#             print(f"\n[QUERY] {query}")
#
#             try:
#                 # 1) 生成陷门
#                 t_trap = time.time()
#                 t1, t2, q_piece = generate_trapdoor(query, dataset)
#                 t_trap1 = time.time() - t_trap
#                 total_trap += t_trap1
#
#                 # 2) 安全搜索
#                 t_start = time.time()
#                 best_cluster, best_piece, part_a_hex, part_b_hex = secure_search(
#                     t1, t2, dataset, q_piece=q_piece, debug=False
#                 )
#                 t_cost = time.time() - t_start
#                 total_time += t_cost
#
#                 # 3) 合并并解密文档集合
#                 #    优先用 secure_search 里的官方解密逻辑，失败再回退明文
#                 if part_a_hex or part_b_hex:
#                     try:
#                         secure_ids = merge_and_decrypt_docs(
#                             dataset,
#                             part_a_hex or "",
#                             part_b_hex or "",
#                         )
#                     except Exception as dec_e:
#                         print(f"[WARN] 解密文档集合失败，使用明文回退: {dec_e}")
#                         secure_ids = get_cluster_docs(best_cluster, dataset)
#                 else:
#                     secure_ids = get_cluster_docs(best_cluster, dataset)
#
#                 # 4) 明文期望
#                 plain_top_docs = plain_results[query]
#                 plain_ids = [str(d["id"]).strip() for d in plain_top_docs]
#
#                 # 5) 准确率
#                 acc = compute_accuracy(secure_ids, plain_ids)
#                 total_acc += acc
#                 valid_queries += 1
#
#                 is_perfect = acc >= 0.999999
#                 if is_perfect:
#                     perfect_count += 1
#
#                 print(
#                     f" -> 搜索耗时: {t_cost:.3f}s | q_piece: {q_piece} | "
#                     f"best_piece: {best_piece} | 准确率: {acc:.3f}"
#                 )
#
#                 per_query_results[query] = {
#                     "secure_ids": [str(x).strip() for x in secure_ids],
#                     "plain_ids": plain_ids,
#                     "time": t_cost,
#                     "accuracy": acc,
#                     "q_piece": q_piece,
#                     "best_piece": best_piece,
#                     "cluster": best_cluster,
#                     "has_cipher_parts": bool(part_a_hex or part_b_hex),
#                 }
#
#             except KeyboardInterrupt:
#                 raise
#             except Exception as e:
#                 print(f"[ERROR] 查询失败: {query}, 错误: {e}")
#         avg_trap = total_trap / valid_queries if valid_queries > 0 else 0.0
#         avg_time = total_time / valid_queries if valid_queries > 0 else 0.0
#         avg_acc = total_acc / valid_queries if valid_queries > 0 else 0.0
#         perfect_ratio = (perfect_count / valid_queries) if valid_queries > 0 else 0.0
#
#         print(f"\n✅ 数据集 {dataset} 平均搜索时间: {avg_time:.3f}s"f", 平均准确率: {avg_acc:.3f}, 完全正确查询占比: {perfect_ratio:.3f}")
#         print(f"\n 平均陷门生成时间：{avg_trap:.3f}s")
#         final_output[dataset] = {
#             "avg_time": avg_time,
#             "avg_acc": avg_acc,
#             "perfect_query_ratio": perfect_ratio,
#             "num_queries": valid_queries,
#             "queries": per_query_results,
#         }
#
#     out_path = os.path.join(OUTPUT_DIR, "wiki_f2_k20.json")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(final_output, f, ensure_ascii=False, indent=2)
#
#     print("\n=== 实验完成 ✅ ===")
#     print(f"所有结果已写入: {out_path}")
#
#
# if __name__ == "__main__":
#     run_experiment()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import json

from generate_trapdoor import generate_trapdoor
from secure_search import (
    secure_search,
    get_cluster_docs,
    merge_and_decrypt_docs,
)

# ===================== 配置 =====================

# 只对 MS MARCO 三个子集做实验
DATASETS = ["msmarco_1k", "msmarco_5k", "msmarco_10k"]

# 15 条查询语句（与你离线/在线实验保持一致）
QUERIES = [
    # "How do you use the Stefan-Boltzmann law to calculate the radius of a star such as Rigel from its luminosity and surface temperature?",
    # "What developmental milestones and typical behaviors should you expect from an 8 year old child at home and at school?",
    # "What are the symptoms of a head lice infestation and how can you check for lice, eggs, and nits on a child's scalp?",
    "What special features does the Burj Khalifa in Dubai have and why was it renamed from Burj Dubai?",
    "What kinds of homes and land are for sale near La Grange, California, and what are their typical sizes and prices?",
    "What are the main characteristics, temperament, and exercise needs of the Dogo Argentino dog breed?",
    # "How are custom safety nets used in industry and what kinds of clients and applications does a company like US Netting serve?",
    "What are effective ways to remove weeds from a garden and prevent them from coming back?",
    # "How common is urinary incontinence in the United States, what can cause it, and is it just a normal part of aging?",
    "How did President Franklin D. Roosevelt prepare the United States for World War II before Pearl Harbor while the country was still isolationist?",
    # "If you have multiple sclerosis and difficulty swallowing pills, is it safe to crush Valium and other medications to make them easier to swallow?",
    # "What strategies can help you get better results when dealing with customer service representatives at cable companies or airlines?",
    # "In Spanish, what does the word 'machacado' mean and how is the verb 'machacar' used in different contexts?",
    "When building a concrete path, how should you design and support plywood formwork so that it is strong enough and keeps the concrete in place?",
    "Why do people join political parties, and which political party did U.S. presidents Woodrow Wilson and Herbert Hoover belong to?",
]

# 明文 baseline 结果目录（与前面脚本一致）
RESULTS_DIR = "/root/siton-tmp/outputs/plain_results"

# 评估结果输出目录
OUTPUT_DIR = "/root/siton-tmp/outputs/eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 工具函数 =====================

def load_plain_results(dataset):
    path = os.path.join(RESULTS_DIR, f"{dataset}.json")
    if not os.path.exists(path):
        print(f"[WARN] 明文结果文件未找到: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy(secure_ids, plain_ids):
    if not plain_ids:
        return 0.0
    secure_set = set(str(s).strip() for s in secure_ids)
    plain_set = set(str(p).strip() for p in plain_ids)
    inter = len(secure_set & plain_set)
    return inter / len(secure_set) if secure_set else 0.0


# ===================== 主流程 =====================

def run_experiment():
    final_output = {}

    for dataset in DATASETS:
        print("\n" + "=" * 100)
        print(f"🔹 Evaluating dataset: {dataset}")
        print("=" * 100)

        plain_results = load_plain_results(dataset)
        if not plain_results:
            print(f"[WARN] 跳过 {dataset}，未找到明文结果文件。")
            continue

        total_trap = 0.0
        total_time = 0.0
        total_acc = 0.0
        valid_queries = 0
        perfect_count = 0
        per_query_results = {}

        for query in QUERIES:
            if query not in plain_results:
                print(f"[WARN] {query} 未在明文结果中找到，跳过。")
                continue

            print(f"\n[QUERY] {query}")

            try:
                # 1) 生成陷门（客户端查询构建）
                t_trap = time.time()
                t1, t2, q_piece = generate_trapdoor(query, dataset)
                t_trap1 = time.time() - t_trap
                total_trap += t_trap1

                # 2) 安全搜索（服务器端私有计算）
                t_start = time.time()
                best_cluster, best_piece, part_a_hex, part_b_hex = secure_search(
                    t1, t2, dataset, q_piece=q_piece, debug=False
                )
                t_cost = time.time() - t_start
                total_time += t_cost

                # 3) 合并并解密文档集合
                #    优先用 secure_search 的解密逻辑，失败再回退明文簇文档
                if part_a_hex or part_b_hex:
                    try:
                        secure_ids = merge_and_decrypt_docs(
                            dataset,
                            part_a_hex or "",
                            part_b_hex or "",
                        )
                    except Exception as dec_e:
                        print(f"[WARN] 解密文档集合失败，使用明文回退: {dec_e}")
                        secure_ids = get_cluster_docs(best_cluster, dataset)
                else:
                    secure_ids = get_cluster_docs(best_cluster, dataset)

                # 4) 明文基线 “正确答案”
                plain_top_docs = plain_results[query]
                plain_ids = [str(d["id"]).strip() for d in plain_top_docs]

                # 5) 准确率（secure 结果 vs 明文结果）
                acc = compute_accuracy(secure_ids, plain_ids)
                total_acc += acc
                valid_queries += 1

                is_perfect = acc >= 0.999999
                if is_perfect:
                    perfect_count += 1

                print(
                    f" -> 搜索耗时: {t_cost:.3f}s | q_piece: {q_piece} | "
                    f"best_piece: {best_piece} | 准确率: {acc:.3f}"
                )
                print("  secure_ids(sample):", [str(x) for x in secure_ids[:5]])
                print("  plain_ids(sample) :", plain_ids[:5])

                per_query_results[query] = {
                    "secure_ids": [str(x).strip() for x in secure_ids],
                    "plain_ids": plain_ids,
                    "time": t_cost,
                    "accuracy": acc,
                    "q_piece": q_piece,
                    "best_piece": best_piece,
                    "cluster": best_cluster,
                    "has_cipher_parts": bool(part_a_hex or part_b_hex),
                    "trapdoor_time": t_trap1,
                }

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[ERROR] 查询失败: {query}, 错误: {e}")

        avg_trap = total_trap / valid_queries if valid_queries > 0 else 0.0
        avg_time = total_time / valid_queries if valid_queries > 0 else 0.0
        avg_acc = total_acc / valid_queries if valid_queries > 0 else 0.0
        perfect_ratio = (perfect_count / valid_queries) if valid_queries > 0 else 0.0

        print(
            f"\n✅ 数据集 {dataset} 平均搜索时间: {avg_time:.3f}s"
            f", 平均准确率: {avg_acc:.3f}, 完全正确查询占比: {perfect_ratio:.3f}"
        )
        print(f"\n 平均陷门生成时间：{avg_trap:.3f}s")

        final_output[dataset] = {
            "avg_time": avg_time,
            "avg_acc": avg_acc,
            "perfect_query_ratio": perfect_ratio,
            "avg_trapdoor_time": avg_trap,
            "num_queries": valid_queries,
            "queries": per_query_results,
        }

    # 输出文件名改为 MS MARCO 版本
    out_path = os.path.join(OUTPUT_DIR, "msmarco_f3_k20.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\n=== 实验完成 ✅ ===")
    print(f"所有结果已写入: {out_path}")


if __name__ == "__main__":
    run_experiment()
