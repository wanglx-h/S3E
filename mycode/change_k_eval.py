#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_enron_k.py
按当前双服务器 + AES-GCM 拆分的流程做评估

在新建的三个索引结构（k=10,20,40）下分别进行实验，
并与 /root/siton-tmp/outputs/plain_results_k 下的明文结果计算准确率。
"""

import os
import time
import json
import shutil

from generate_trapdoor import generate_trapdoor
from secure_search import (
    secure_search,
    get_cluster_docs,
    merge_and_decrypt_docs,
)

# ===================== 路径配置 =====================
CANON_INDEX_DIR = "/root/siton-tmp/outputs/index"

# 我们新建的 k 分别索引存放位置
K_INDEX_ROOT = "/root/siton-tmp/outputs/index_k"

# 明文 baseline 结果目录
RESULTS_DIR = r"/root/siton-tmp/outputs/plain_results_k"

# 评估输出目录
OUTPUT_DIR = r"../outputs/eval_k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 数据集 & 查询 =====================

DATASETS = ["enron_5k"]

QUERIES = [
    # "What meetings are scheduled?",
    "Tell me about energy trading",
    # "What contracts were discussed?",
    # "What are the price forecasts?",
    # "What reports need analysis?",
    # "What projects are in development?",
    # "What companies are involved?",
    "What emails need attention?",
    "What conference calls are planned?",
    # "What financial information is available?",
    # "Emails about SEC strategy meetings",
    # "Messages mentioning building access or badges",
    # "HR newsletters on labor or employment policy",
    # "Forwards with BNA Daily Labor Report content",
    # "Memos on minimum wage or unemployment issues",
    # "Emails discussing union negotiations or wage increases",
    "Messages about post-9/11 employment impacts",
    # "Notes on federal worker discrimination or whistleblower cases",
    # "Emails that list multiple labor news headlines",
    # "Messages sharing external news links with login info",
    "Internal calendar or on-call notification emails",
    "Emails between facilities or admin staff about office locations",
    # "Messages referencing ILO or international labor standards",
    # "Forwards about appointments to U.S. labor-related posts",
    # "Emails on benefit or donation program changes",
    # "Threads with multiple HR recipients in one blast",
    # "Messages mentioning airport security or related legislation",
    # "Emails summarizing congressional labor actions",
    # "Messages about court rulings on workplace drug testing",
    # "Long digest-style labor and employment updates",
]


# ===================== 工具函数 =====================

def load_plain_results(dataset, k_value):
    """
    从 plain_results_k 目录加载对应 k 的明文结果。
    文件名: enron_5k_k10.json / enron_5k_k20.json / enron_5k_k40.json
    """
    filename = f"{dataset}_k{k_value}.json"
    path = os.path.join(RESULTS_DIR, filename)
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


def switch_index_for_k(dataset, k_value):
    src = os.path.join(K_INDEX_ROOT, f"k{k_value}", dataset)
    dst_root = CANON_INDEX_DIR
    dst = os.path.join(dst_root, dataset)

    if not os.path.isdir(src):
        raise FileNotFoundError(f"索引目录不存在: {src}")

    os.makedirs(dst_root, exist_ok=True)

    # 如果目标已存在，先删掉（可能是目录或者符号链接）
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)

    # 优先创建符号链接；若失败，则复制目录
    try:
        os.symlink(src, dst)
        print(f"[INFO] 已将 {dst} 软链接到 {src}")
    except (OSError, NotImplementedError) as e:
        print(f"[WARN] 创建软链接失败 ({e})，改为复制目录")
        shutil.copytree(src, dst)
        print(f"[INFO] 已将 {src} 复制到 {dst}")


# ===================== 主流程 =====================

def run_experiment():
    # 依次对 k=10,20,40 做三组实验
    for k_value in [10, 20, 40]:
        final_output = {}

        for dataset in DATASETS:
            print("\n" + "=" * 100)
            print(f"🔹 Evaluating dataset: {dataset} (k={k_value})")
            print("=" * 100)

            # 1) 切换 index：让密文代码使用 index_k/k{k}/{dataset}
            try:
                switch_index_for_k(dataset, k_value)
            except Exception as e:
                print(f"[ERROR] 切换索引失败: {e}")
                continue

            # 2) 加载对应 k 的明文结果
            plain_results = load_plain_results(dataset, k_value)
            if not plain_results:
                print(f"[WARN] 跳过 {dataset}，未找到对应 k={k_value} 的明文结果文件。")
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
                    # 3) 生成陷门（内部仍然用 CANON_INDEX_DIR）
                    t_trap = time.time()
                    t1, t2, q_piece = generate_trapdoor(query, dataset)
                    t_trap1 = time.time() - t_trap
                    total_trap += t_trap1

                    # 4) 安全搜索
                    t_start = time.time()
                    best_cluster, best_piece, part_a_hex, part_b_hex = secure_search(
                        t1, t2, dataset, q_piece=q_piece, debug=False
                    )
                    t_cost = time.time() - t_start
                    total_time += t_cost

                    # 5) 合并并解密文档集合
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

                    # 6) 明文期望
                    plain_top_docs = plain_results[query]
                    plain_ids = [str(d["id"]).strip() for d in plain_top_docs]

                    # 7) 准确率
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

                    per_query_results[query] = {
                        "secure_ids": [str(x).strip() for x in secure_ids],
                        "plain_ids": plain_ids,
                        "time": t_cost,
                        "accuracy": acc,
                        "q_piece": q_piece,
                        "best_piece": best_piece,
                        "cluster": best_cluster,
                        "has_cipher_parts": bool(part_a_hex or part_b_hex),
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
                f"\n✅ 数据集 {dataset} (k={k_value}) 平均搜索时间: {avg_time:.3f}s"
                f", 平均准确率: {avg_acc:.3f}, 完全正确查询占比: {perfect_ratio:.3f}"
            )
            print(f"\n 平均陷门生成时间：{avg_trap:.3f}s")

            final_output[dataset] = {
                "k": k_value,
                "avg_time": avg_time,
                "avg_acc": avg_acc,
                "perfect_query_ratio": perfect_ratio,
                "num_queries": valid_queries,
                "avg_trap_time": avg_trap,
                "queries": per_query_results,
            }

        # 8) 保存当前 k 的总结结果
        out_filename = f"enron_f2_k{k_value}.json"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print("\n=== 本轮实验完成 ✅ ===")
        print(f"结果已写入: {out_path}")


if __name__ == "__main__":
    run_experiment()
