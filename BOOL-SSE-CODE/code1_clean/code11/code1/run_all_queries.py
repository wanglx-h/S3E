# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# import os
# import time
# import shutil
# import logging
# import statistics
#
# from pypbc import Pairing
#
# from tools import (
#     ServerPathFromTools,
#     ParameterPathFromTools,
#     readPP,
#     readServerKey,
#     readReceiverVKey,
#     readWSet,
#     readInds,
#     loadFile,
# )
# from TrapGen import TrapGen
# from Search import Search, Copy2UnCopy
# from Retrieve import Retrieve
#
# logger = logging.getLogger("Caedios")
#
# # ===================== Enron 查询：自然语言文本 & 关键词 =====================
#
# ENRON_QUERIES_TEXTS = [
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
#     "What legal documents are relevant?",
#     "What memos have been drafted?",
#     "What reports were issued?",
#     "What transactions are pending?",
#     "What are the latest policy changes?",
#     "What is the status of the project?",
#     "What customer feedback do we have?",
#     "What is the summary of the meeting?",
#     "What deals were signed?",
#     "What regulatory issues are discussed?",
# ]
#
# ENRON_QUERIES_KEYWORDS = [
#     ["meeting", "schedule"],
#     ["energy", "trading"],
#     ["contract", "discuss"],
#     ["price", "forecast"],
#     ["report", "analysis"],
#     ["project", "development"],
#     ["company", "involved"],
#     ["email", "attention"],
#     ["conference", "planned"],
#     ["financial", "information"],
#     ["legal", "document", "relevant"],
#     ["memo", "draft"],
#     ["report", "issued"],
#     ["transaction", "pending"],
#     ["policy", "change"],
#     ["project", "status"],
#     ["customer", "feedback"],
#     ["meeting", "summary"],
#     ["deal", "signed"],
#     ["regulatory", "issue"],
# ]
#
# # ===================== Wiki 查询：自然语言文本 & 关键词 =====================
#
# WIKI_QUERIES_TEXTS = [
#     "What is the history of artificial intelligence?",
#     "Tell me about the structure of the human brain.",
#     "What are the major events of World War II?",
#     "Explain the theory of evolution by Charles Darwin.",
#     "What are the moons of Jupiter?",
#     "Describe the process of photosynthesis.",
#     "Who discovered gravity?",
#     "What are the causes of climate change?",
#     "Explain quantum mechanics basics?",
#     "Tell me about the culture of ancient Egypt.",
#     "April month overview in the Gregorian calendar",
#     "Etymology or origin of the name April",
#     "April holidays and observances worldwide",
#     "Seasonal description of April in both hemispheres",
#     "Movable Christian feasts that fall in April",
#     "Sayings or phrases about April weather",
#     "Historical events that happened in April",
#     "April cultural festivals in Europe or Asia",
#     "Sports or major events usually held in April",
#     "August month overview and calendar facts",
#     "Etymology or origin of the name August",
#     "August national or religious holidays",
#     "August historical events in the 20th century",
#     "Definition of art as human creative activity",
#     "Categories of art such as visual or performing",
#     "Discussion of art versus design",
#     "Short history outline of art across eras",
#     "Examples of everyday objects treated as art",
#     "Comparison of April seasons across hemispheres",
#     "August cultural festivals and public holidays",
# ]
#
# WIKI_QUERIES_KEYWORDS = [
#     # 1. "What is the history of artificial intelligence?"
#     ["artificial", "intelligence", "history"],
#
#     # 2. "Tell me about the structure of the human brain."
#     ["human", "brain", "structure"],
#
#     # 3. "What are the major events of World War II?"
#     ["world", "war", "event"],
#
#     # 4. "Explain the theory of evolution by Charles Darwin."
#     ["evolution", "theory", "darwin"],
#
#     # 5. "What are the moons of Jupiter?"
#     ["moon", "jupiter"],
#
#     # 6. "Describe the process of photosynthesis."
#     ["photosynthesis", "process"],
#
#     # 7. "Who discovered gravity?"
#     ["gravity", "discover"],
#
#     # 8. "What are the causes of climate change?"
#     ["climate", "change", "cause"],
#
#     # 9. "Explain quantum mechanics basics."
#     ["quantum", "mechanics", "basic"],
#
#     # 10. "Tell me about the culture of ancient Egypt."
#     ["ancient", "egypt", "culture"],
#
#     # 11. "April month overview in the Gregorian calendar"
#     ["april", "gregorian", "calendar"],
#
#     # 12. "Etymology or origin of the name April"
#     ["april", "name", "etymology"],
#
#     # 13. "April holidays and observances worldwide"
#     ["april", "holiday", "observance"],
#
#     # 14. "Seasonal description of April in both hemispheres"
#     ["april", "season", "hemisphere"],
#
#     # 15. "Movable Christian feasts that fall in April"
#     ["april", "christian", "feast"],
#
#     # 16. "Sayings or phrases about April weather"
#     ["april", "weather", "saying"],
#
#     # 17. "Historical events that happened in April"
#     ["april", "historical", "event"],
#
#     # 18. "April cultural festivals in Europe or Asia"
#     ["april", "cultural", "festival", "europe", "asia"],
#
#     # 19. "Sports or major events usually held in April"
#     ["april", "sport", "event"],
#
#     # 20. "August month overview and calendar facts"
#     ["august", "calendar", "overview"],
#
#     # 21. "Etymology or origin of the name August"
#     ["august", "name", "etymology"],
#
#     # 22. "August national or religious holidays"
#     ["august", "holiday", "national", "religious"],
#
#     # 23. "August historical events in the 20th century"
#     ["august", "historical", "event", "century"],
#
#     # 24. "Definition of art as human creative activity"
#     ["art", "definition", "creative", "activity"],
#
#     # 25. "Categories of art such as visual or performing"
#     ["art", "category", "visual", "performing"],
#
#     # 26. "Discussion of art versus design"
#     ["art", "design", "discussion"],
#
#     # 27. "Short history outline of art across eras"
#     ["art", "history", "era", "outline"],
#
#     # 28. "Examples of everyday objects treated as art"
#     ["art", "example", "everyday", "object"],
#
#     # 29. "Comparison of April seasons across hemispheres"
#     ["april", "season", "hemisphere", "comparison"],
#
#     # 30. "August cultural festivals and public holidays"
#     ["august", "cultural", "festival", "holiday"],
# ]
#
#
# # ===================== 工具函数 =====================
#
# def build_Q_and_V_from_keywords(keywords):
#     """
#     给定关键词列表，比如 ["energy", "trading"]
#     构造协议里用的 Q / V：
#       Q[0] 为空串
#       Q[1], Q[2], ... = 对应关键词
#       V[i] = 1  (布尔向量，全 1)
#     """
#     Q = {}
#     V = {}
#
#     Q[0] = ""
#     for idx, w in enumerate(keywords, start=1):
#         Q[idx] = w
#         V[idx] = 1
#
#     return Q, V
#
#
# def load_rkuv_copy():
#     """直接返回 rkuv.dat 里 eval 出来的原始 dict（字符串版）"""
#     path = os.path.join(ServerPathFromTools, ParameterPathFromTools, "rkuv.dat")
#     return loadFile(path)
#
#
# def copy_index_for_dataset(dataset_name):
#     """
#     把某个数据集的索引文件拷贝为“默认索引”：
#       Server/EDB.dat
#       Server/XSet.dat
#       Server/Parameter/Inds.dat
#       Server/Parameter/WSet.dat
#     这样后续的 Copy2UnCopy / readWSet / readInds 都可以复用。
#     """
#     base = ServerPathFromTools.rstrip("/")
#
#     src_edb   = os.path.join(base, f"EDB_{dataset_name}.dat")
#     src_xset  = os.path.join(base, f"XSet_{dataset_name}.dat")
#     src_inds  = os.path.join(base, "Parameter", f"Inds_{dataset_name}.dat")
#     src_wset  = os.path.join(base, "Parameter", f"WSet_{dataset_name}.dat")
#
#     dst_edb   = os.path.join(base, "EDB.dat")
#     dst_xset  = os.path.join(base, "XSet.dat")
#     dst_inds  = os.path.join(base, "Parameter", "Inds.dat")
#     dst_wset  = os.path.join(base, "Parameter", "WSet.dat")
#
#     # 检查是否存在
#     missing = []
#     for path in [src_edb, src_xset, src_inds, src_wset]:
#         if not os.path.exists(path):
#             missing.append(path)
#
#     if missing:
#         print("[WARN] 缺少以下索引文件：")
#         for m in missing:
#             print("   ", m)
#         return False
#
#     # 逐个拷贝
#     shutil.copyfile(src_edb,  dst_edb)
#     shutil.copyfile(src_xset, dst_xset)
#     shutil.copyfile(src_inds, dst_inds)
#     shutil.copyfile(src_wset, dst_wset)
#
#     logger.info("Copied %s -> %s", src_edb,  dst_edb)
#     logger.info("Copied %s -> %s", src_xset, dst_xset)
#     logger.info("Copied %s -> %s", src_inds, dst_inds)
#     logger.info("Copied %s -> %s", src_wset, dst_wset)
#
#     return True
#
#
# # ===================== 单次查询：TrapGen + Search + Retrieve（只计时） =====================
#
# def run_one_query(params, g, pkfs, pkbs, skfs, skbs, skv, rkuv_copy,
#                   EDB, XSet, WSetLocal, IndsLocal,
#                   keywords):
#     """
#     单次查询：
#       - TrapGen / Search / Retrieve 都跑，并记录时间；
#       - 不再计算准确率，也不依赖明文结果。
#     """
#     Q, V = build_Q_and_V_from_keywords(keywords)
#
#     # ---- TrapGen ----
#     t0 = time.time()
#     tokenfs = TrapGen(params, g, pkfs, pkbs, skv, WSetLocal, Q, V)
#     t1 = time.time()
#     trap_time = t1 - t0
#
#     # ---- Search ----
#     serverKey = [pkfs, skfs, pkbs, skbs]
#     t2 = time.time()
#     stokens = Search(params, tokenfs, serverKey, rkuv_copy, EDB, XSet)
#     t3 = time.time()
#     search_time = t3 - t2
#
#     # ---- Retrieve（如果只关心前两步，也可以不计）----
#     t4 = time.time()
#     _ = Retrieve(params, skv, stokens, IndsLocal)
#     t5 = time.time()
#     retrieve_time = t5 - t4
#
#     return trap_time, search_time, retrieve_time
#
#
# # ===================== 主流程 =====================
#
# def main():
#     # 统一从 Server 侧读公共参数、密钥
#     params, g, Kx, Kz, Kl = readPP(ServerPathFromTools)
#     pairing = Pairing(params)
#
#     pkfs, skfs, pkbs, skbs = readServerKey(ServerPathFromTools)
#     pkv, skv = readReceiverVKey(ServerPathFromTools)
#     rkuv_copy = load_rkuv_copy()
#
#     # 实验数据集列表 —— 名字要和 run_all_datasets.py 里保存索引时一致
#     datasets = [
#         ("enron_1k",  "CSExperiment/enron_1k/maildir/corman-s"),
#         ("enron_5k",  "CSExperiment/enron_5k/maildir/corman-s"),
#         ("enron_10k", "CSExperiment/enron_10k/maildir/corman-s"),
#         ("wiki_1k",   "CSExperiment/wiki_1k/maildir/corman-s"),
#         ("wiki_5k",   "CSExperiment/wiki_5k/maildir/corman-s"),
#         ("wiki_10k",  "CSExperiment/wiki_10k/maildir/corman-s"),
#     ]
#
#     # 结果记录： {dataset: {"trap": [...], "search": [...], "retrieve": [...]}}
#     results = {}
#
#     for dataset_name, maildir in datasets:
#         print(f"\n========== DATASET: {dataset_name} ==========")
#
#         # 先把这个数据集的索引拷贝成“默认索引”
#         ok = copy_index_for_dataset(dataset_name)
#         if not ok:
#             print(f"[SKIP] 数据集 {dataset_name} 缺少索引文件，跳过。")
#             continue
#
#         # 根据当前默认索引，构建 EDB / XSet
#         print("[DEBUG] Enter Copy2UnCopy")
#         EDB, XSet = Copy2UnCopy(params)
#         WSetLocal = readWSet(ServerPathFromTools)
#         IndsLocal = readInds(ServerPathFromTools)
#
#         # 选择该数据集使用的查询集合（enron 用 20 个，wiki 用 30 个）
#         if dataset_name.startswith("enron"):
#             queries_texts = ENRON_QUERIES_TEXTS
#             queries_keywords = ENRON_QUERIES_KEYWORDS
#         else:  # wiki_*
#             queries_texts = WIKI_QUERIES_TEXTS
#             queries_keywords = WIKI_QUERIES_KEYWORDS
#
#         trap_times = []
#         search_times = []
#         retrieve_times = []
#
#         # 对该数据集的所有查询逐个跑
#         for qi, keywords in enumerate(queries_keywords, start=1):
#             # 找到对应的自然语言查询文本（只是为了打印日志，更直观）
#             if qi - 1 < len(queries_texts):
#                 query_text = queries_texts[qi - 1]
#             else:
#                 query_text = f"Q{qi}"
#
#             print(f"[{dataset_name}] Query #{qi:02d}  \"{query_text}\"  keywords = {keywords}")
#
#             t_trap, t_search, t_retrieve = run_one_query(
#                 params, g, pkfs, pkbs, skfs, skbs, skv, rkuv_copy,
#                 EDB, XSet, WSetLocal, IndsLocal,
#                 keywords,
#             )
#
#             print(
#                 f"    TrapGen: {t_trap*1000:.2f} ms, "
#                 f"Search: {t_search*1000:.2f} ms, "
#                 f"Retrieve: {t_retrieve*1000:.2f} ms"
#             )
#
#             trap_times.append(t_trap)
#             search_times.append(t_search)
#             retrieve_times.append(t_retrieve)
#
#         results[dataset_name] = {
#             "trap": trap_times,
#             "search": search_times,
#             "retrieve": retrieve_times,
#         }
#
#     # ================== 汇总打印 ==================
#     print("\n\n================= SUMMARY (平均耗时) =================")
#     print("单位：毫秒 (ms)")
#     for dataset_name, data in results.items():
#         trap_avg     = statistics.mean(data["trap"]) * 1000 if data.get("trap") else 0.0
#         search_avg   = statistics.mean(data["search"]) * 1000 if data.get("search") else 0.0
#         retrieve_avg = statistics.mean(data["retrieve"]) * 1000 if data.get("retrieve") else 0.0
#
#         print(
#             f"{dataset_name:10s} | "
#             f"TrapGen: {trap_avg:8.2f} ms, "
#             f"Search: {search_avg:8.2f} ms, "
#             f"Retrieve: {retrieve_avg:8.2f} ms"
#         )
#
#
# if __name__ == "__main__":
#     main()



# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# import os
# import time
# import shutil
# import logging
# import statistics
# import json
# #enron和wiki
# from pypbc import Pairing
#
# from tools import (
#     ServerPathFromTools,
#     ParameterPathFromTools,
#     readPP,
#     readServerKey,
#     readReceiverVKey,
#     readWSet,
#     readInds,
#     loadFile,
# )
# from TrapGen import TrapGen
# from Search import Search, Copy2UnCopy
# from Retrieve import Retrieve
#
# logger = logging.getLogger("Caedios")
#
# # ===================== 明文结果文件路径 =====================
# PLAIN_RESULT_PATHS = {
#     "enron_1k":  "/root/siton-tmp/outputs/plain_results/enron_1k.json",
#     "enron_5k":  "/root/siton-tmp/outputs/plain_results/enron_5k.json",
#     "enron_10k": "/root/siton-tmp/outputs/plain_results/enron_10k.json",
#     "wiki_1k":   "/root/siton-tmp/outputs/plain_results/wiki_1k.json",
#     "wiki_5k":   "/root/siton-tmp/outputs/plain_results/wiki_5k.json",
#     "wiki_10k":  "/root/siton-tmp/outputs/plain_results/wiki_10k.json",
# }
#
# # ===================== Enron 查询：自然语言文本 & 关键词 =====================
#
# ENRON_QUERIES_TEXTS = [
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
#     "What legal documents are relevant?",
#     "What memos have been drafted?",
#     "What reports were issued?",
#     "What transactions are pending?",
#     "What are the latest policy changes?",
#     "What is the status of the project?",
#     "What customer feedback do we have?",
#     "What is the summary of the meeting?",
#     "What deals were signed?",
#     "What regulatory issues are discussed?",
# ]
#
# # ENRON_QUERIES_KEYWORDS = [
# #     ["meeting"],
# #     ["energy"],
# #     ["contract"],
# #     ["price"],
# #     ["report"],
# #     ["project"],
# #     ["company"],
# #     ["email"],
# #     ["conference"],
# #     ["financial"],
# #     ["legal"],
# #     ["memo"],
# #     ["report"],
# #     ["transaction"],
# #     ["policy"],
# #     ["project"],
# #     ["customer"],
# #     ["meeting"],
# #     ["deal"],
# #     ["regulatory"],
# # ]
#
# # ENRON_QUERIES_KEYWORDS = [
# #     ["meeting", "schedule"],
# #     ["energy", "trading"],
# #     ["contract", "discuss"],
# #     ["price", "forecast"],
# #     ["report", "analysis"],
# #     ["project", "development"],
# #     ["company", "involved"],
# #     ["email", "attention"],
# #     ["conference", "planned"],
# #     ["financial", "information"],
# #     ["legal", "document"],
# #     ["memo", "draft"],
# #     ["report", "issued"],
# #     ["transaction", "pending"],
# #     ["policy", "change"],
# #     ["project", "status"],
# #     ["customer", "feedback"],
# #     ["meeting", "summary"],
# #     ["deal", "signed"],
# #     ["regulatory", "issue"],
# # ]
# #
# # ENRON_QUERIES_KEYWORDS = [
# #     ["meeting", "schedule", "calendar"],
# #     ["energy", "trading", "market"],
# #     ["contract", "discuss", "negotiation"],
# #     ["price", "forecast", "trend"],
# #     ["report", "analysis", "review"],
# #     ["project", "development", "progress"],
# #     ["company", "involved", "partner"],
# #     ["email", "attention", "priority"],
# #     ["conference", "planned", "call"],
# #     ["financial", "information", "data"],
# #     ["legal", "document", "relevant"],
# #     ["memo", "draft", "revision"],
# #     ["report", "issued", "publication"],
# #     ["transaction", "pending", "status"],
# #     ["policy", "change", "update"],
# #     ["project", "status", "timeline"],
# #     ["customer", "feedback", "review"],
# #     ["meeting", "summary", "note"],
# #     ["deal", "signed", "agreement"],
# #     ["regulatory", "issue", "compliance"],
# # ]
# #
#
#
# ENRON_QUERIES_KEYWORDS = [
#     ["meeting", "schedule"],
#     ["energy", "trading"],
#     ["contract", "discuss"],
#     ["price", "forecast"],
#     ["report", "analysis"],
#     ["project", "development"],
#     ["company", "involved"],
#     ["email", "attention"],
#     ["conference", "planned"],
#     ["financial", "information"],
#     # ["legal", "document", "relevant"],
#     ["legal", "document"],
#     ["memo", "draft"],
#     ["report", "issued"],
#     ["transaction", "pending"],
#     ["policy", "change"],
#     ["project", "status"],
#     ["customer", "feedback"],
#     ["meeting", "summary"],
#     ["deal", "signed"],
#     ["regulatory", "issue"],
# ]
#
# # ===================== Wiki 查询：自然语言文本 & 关键词 =====================
#
# WIKI_QUERIES_TEXTS = [
#     # "What is the history of artificial intelligence?",
#     # "Tell me about the structure of the human brain.",
#     # "What are the major events of World War II?",
#     # "Explain the theory of evolution by Charles Darwin.",
#     # "What are the moons of Jupiter?",
#     # "Describe the process of photosynthesis.",
#     # "Who discovered gravity?",
#     # "What are the causes of climate change?",
#     # "Explain quantum mechanics basics.",
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
#     # "Definition of art as human creative activity",
#     # "Categories of art such as visual or performing",
#     # "Discussion of art versus design",
#     # "Short history outline of art across eras",
#     # "Examples of everyday objects treated as art",
#     # "Comparison of April seasons across hemispheres",
#     # "August cultural festivals and public holidays",
# ]
#
# WIKI_QUERIES_KEYWORDS = [
#     # # 1. "What is the history of artificial intelligence?"
#     # ["artificial", "intelligence", "history"],
#     #
#     # # 2. "Tell me about the structure of the human brain."
#     # ["human", "brain", "structure"],
#     #
#     # # 3. "What are the major events of World War II?"
#     # ["world", "war", "event"],
#     #
#     # # 4. "Explain the theory of evolution by Charles Darwin."
#     # ["evolution", "theory", "darwin"],
#     #
#     # # 5. "What are the moons of Jupiter?"
#     # ["moon", "jupiter"],
#     #
#     # # 6. "Describe the process of photosynthesis."
#     # ["photosynthesis", "process"],
#     #
#     # # 7. "Who discovered gravity?"
#     # ["gravity", "discover"],
#     #
#     # # 8. "What are the causes of climate change?"
#     # ["climate", "change", "cause"],
#     #
#     # # 9. "Explain quantum mechanics basics."
#     # ["quantum", "mechanics", "basic"],
#     #
#     # # 10. "Tell me about the culture of ancient Egypt."
#     # ["ancient", "egypt", "culture"],
#     #
#     # # 11. "April month overview in the Gregorian calendar"
#     # ["april", "gregorian", "calendar"],
#     #
#     # # 12. "Etymology or origin of the name April"
#     # ["april", "name", "etymology"],
#     #
#     # # 13. "April holidays and observances worldwide"
#     # ["april", "holiday", "observance"],
#     #
#     # # 14. "Seasonal description of April in both hemispheres"
#     # ["april", "season", "hemisphere"],
#     #
#     # # 15. "Movable Christian feasts that fall in April"
#     # ["april", "christian", "feast"],
#     #
#     # # 16. "Sayings or phrases about April weather"
#     # ["april", "weather", "saying"],
#     #
#     # # 17. "Historical events that happened in April"
#     # ["april", "historical", "event"],
#
#     # 18. "April cultural festivals in Europe or Asia"
#     # ["april", "cultural", "festival", "europe", "asia"],
#
#     # # 19. "Sports or major events usually held in April"
#     # ["april", "sport", "event"],
#     #
#     # # 20. "August month overview and calendar facts"
#     # ["august", "calendar", "overview"],
#     #
#     # # 21. "Etymology or origin of the name August"
#     # ["august", "name", "etymology"],
#     #
#     # # 22. "August national or religious holidays"
#     # ["august", "holiday", "national", "religious"],
#     #
#     # # 23. "August historical events in the 20th century"
#     # ["august", "historical", "event", "century"],
#     #
#     # # 24. "Definition of art as human creative activity"
#     # ["art", "definition", "creative", "activity"],
#     #
#     # # 25. "Categories of art such as visual or performing"
#     # ["art", "category", "visual", "performing"],
#     #
#     # # 26. "Discussion of art versus design"
#     # ["art", "design", "discussion"],
#     #
#     # # 27. "Short history outline of art across eras"
#     # ["art", "history", "era", "outline"],
#     #
#     # # 28. "Examples of everyday objects treated as art"
#     # ["art", "example", "everyday", "object"],
#     #
#     # # 29. "Comparison of April seasons across hemispheres"
#     # ["april", "season", "hemisphere", "comparison"],
#     #
#     # 30. "August cultural festivals and public holidays"
#     # ["august", "cultural", "festival", "holiday"],
# ]
#
#
# # ===================== 工具函数 =====================
#
# def build_Q_and_V_from_keywords(keywords):
#     """
#     给定关键词列表，比如 ["energy", "trading"]
#     构造协议里用的 Q / V：
#       Q[0] 为空串
#       Q[1], Q[2], ... = 对应关键词
#       V[i] = 1  (布尔向量，全 1)
#     """
#     Q = {}
#     V = {}
#
#     Q[0] = ""
#     for idx, w in enumerate(keywords, start=1):
#         Q[idx] = w
#         V[idx] = 1
#
#     return Q, V
#
#
# def load_rkuv_copy():
#     """直接返回 rkuv.dat 里 eval 出来的原始 dict（字符串版）"""
#     path = os.path.join(ServerPathFromTools, ParameterPathFromTools, "rkuv.dat")
#     return loadFile(path)
#
#
# def copy_index_for_dataset(dataset_name):
#     """
#     把某个数据集的索引文件拷贝为“默认索引”：
#       Server/EDB.dat
#       Server/XSet.dat
#       Server/Parameter/Inds.dat
#       Server/Parameter/WSet.dat
#     这样后续的 Copy2UnCopy / readWSet / readInds 都可以复用。
#     """
#     base = ServerPathFromTools.rstrip("/")
#
#     src_edb   = os.path.join(base, f"EDB_{dataset_name}.dat")
#     src_xset  = os.path.join(base, f"XSet_{dataset_name}.dat")
#     src_inds  = os.path.join(base, "Parameter", f"Inds_{dataset_name}.dat")
#     src_wset  = os.path.join(base, "Parameter", f"WSet_{dataset_name}.dat")
#
#     dst_edb   = os.path.join(base, "EDB.dat")
#     dst_xset  = os.path.join(base, "XSet.dat")
#     dst_inds  = os.path.join(base, "Parameter", "Inds.dat")
#     dst_wset  = os.path.join(base, "Parameter", "WSet.dat")
#
#     # 检查是否存在
#     missing = []
#     for path in [src_edb, src_xset, src_inds, src_wset]:
#         if not os.path.exists(path):
#             missing.append(path)
#
#     if missing:
#         print("[WARN] 缺少以下索引文件：")
#         for m in missing:
#             print("   ", m)
#         return False
#
#     # 逐个拷贝
#     shutil.copyfile(src_edb,  dst_edb)
#     shutil.copyfile(src_xset, dst_xset)
#     shutil.copyfile(src_inds, dst_inds)
#     shutil.copyfile(src_wset, dst_wset)
#
#     logger.info("Copied %s -> %s", src_edb,  dst_edb)
#     logger.info("Copied %s -> %s", src_xset, dst_xset)
#     logger.info("Copied %s -> %s", src_inds, dst_inds)
#     logger.info("Copied %s -> %s", src_wset, dst_wset)
#
#     return True
#
#
# def normalize_doc_id(x):
#     """
#     文档 id 归一化：
#     - 转成字符串
#     - 去掉两侧空白
#     - 去掉末尾的点（例如 '58.' -> '58'）
#     """
#     s = str(x).strip()
#     if s.endswith("."):
#         s = s[:-1]
#     return s
#
#
# def boolean_filter_plain_results(plain_list, keywords):
#     """
#     在明文结果列表里，用“布尔 AND”过滤出真正包含所有关键词的文档：
#       - plain_list: [ { "id": ..., "text": ... }, ... ]
#       - keywords:   当前查询用到的关键词列表
#
#     返回：retrieved_ids_norm: 规范化后的 id 列表（字符串）。
#     """
#     if not plain_list:
#         return []
#
#     kws = [kw.lower() for kw in keywords]
#     retrieved_ids = []
#
#     for item in plain_list:
#         doc_id = item.get("id", "")
#         text = item.get("text", "")
#         text_lc = str(text).lower()
#
#         ok = True
#         for kw in kws:
#             if kw not in text_lc:
#                 ok = False
#                 break
#
#         if ok:
#             retrieved_ids.append(doc_id)
#
#     retrieved_ids_norm = [normalize_doc_id(did) for did in retrieved_ids]
#     return retrieved_ids_norm
#
#
# # ===================== 单次查询：TrapGen + Search + Retrieve + Accuracy =====================
#
# def run_one_query(params, g, pkfs, pkbs, skfs, skbs, skv, rkuv_copy,
#                   EDB, XSet, WSetLocal, IndsLocal,
#                   keywords, plain_list=None):
#     """
#     单次查询：
#       - TrapGen / Search / Retrieve 仍然跑，用于计时；
#       - 准确率完全基于 plain_list（明文 JSON）和关键词做布尔过滤。
#     """
#     Q, V = build_Q_and_V_from_keywords(keywords)
#
#     # ---- TrapGen ----
#     t0 = time.time()
#     tokenfs = TrapGen(params, g, pkfs, pkbs, skv, WSetLocal, Q, V)
#     t1 = time.time()
#     trap_time = t1 - t0
#
#     # ---- Search ----
#     serverKey = [pkfs, skfs, pkbs, skbs]
#     t2 = time.time()
#     try:
#         stokens = Search(params, tokenfs, serverKey, rkuv_copy, EDB, XSet)
#         t3 = time.time()
#     except Exception as e:
#         print(f"[WARN] Search error: {e}")
#         stokens = None
#         t3 = time.time()
#     search_time = t3 - t2
#
#     # ---- Retrieve ----
#     t4 = time.time()
#     try:
#         _ = Retrieve(params, skv, stokens, IndsLocal)
#         t5 = time.time()
#     except Exception as e:
#         print(f"[WARN] Retrieve error: {e}")
#         t5 = time.time()
#     retrieve_time = t5 - t4
#
#     # ---- Accuracy（只看明文 JSON）----
#     accuracy = None
#     if plain_list is not None and isinstance(plain_list, list):
#         # 明文 JSON 里的所有结果 id
#         plain_ids_raw = [item.get("id", "") for item in plain_list]
#         plain_ids_norm = [normalize_doc_id(x) for x in plain_ids_raw]
#
#         print("[DEBUG] plain_ids (raw):", plain_ids_raw[:5])
#         print("[DEBUG] plain_ids (norm):", plain_ids_norm[:5])
#
#         # 用布尔 AND 对 text 做过滤
#         retrieved_ids_norm = boolean_filter_plain_results(plain_list, keywords)
#         print("[DEBUG] retrieved_ids (norm):", retrieved_ids_norm[:5])
#
#         if len(plain_ids_norm) > 0:
#             inter = set(plain_ids_norm) & set(retrieved_ids_norm)
#             accuracy = len(inter) / len(plain_ids_norm)
#         else:
#             accuracy = 0.0
#
#     return trap_time, search_time, retrieve_time, accuracy
#
#
# # ===================== 主流程 =====================
#
# def main():
#     # 统一从 Server 侧读公共参数、密钥
#     params, g, Kx, Kz, Kl = readPP(ServerPathFromTools)
#     pairing = Pairing(params)
#
#     pkfs, skfs, pkbs, skbs = readServerKey(ServerPathFromTools)
#     pkv, skv = readReceiverVKey(ServerPathFromTools)
#     rkuv_copy = load_rkuv_copy()
#
#     # 实验数据集列表 —— 名字要和 run_all_datasets.py 里保存索引时一致
#     datasets = [
#         ("enron_1k",  "CSExperiment/enron_1k/maildir/corman-s"),
#         # ("enron_5k",  "CSExperiment/enron_5k/maildir/corman-s"),
#         # ("enron_10k", "CSExperiment/enron_10k/maildir/corman-s"),
#         # ("wiki_1k",   "CSExperiment/wiki_1k/maildir/corman-s"),
#         # ("wiki_5k",   "CSExperiment/wiki_5k/maildir/corman-s"),
#         # ("wiki_10k",  "CSExperiment/wiki_10k/maildir/corman-s"),
#     ]
#
#     # 结果记录： {dataset: {"trap": [...], "search": [...], "retrieve": [...], "accuracy": [...]}}
#     results = {}
#
#     for dataset_name, maildir in datasets:
#         print(f"\n========== DATASET: {dataset_name} ==========")
#
#         # 先把这个数据集的索引拷贝成“默认索引”
#         ok = copy_index_for_dataset(dataset_name)
#         if not ok:
#             print(f"[SKIP] 数据集 {dataset_name} 缺少索引文件，跳过。")
#             continue
#
#         # 根据当前默认索引，构建 EDB / XSet
#         print("[DEBUG] Enter Copy2UnCopy")
#         EDB, XSet = Copy2UnCopy(params)
#         WSetLocal = readWSet(ServerPathFromTools)
#         IndsLocal = readInds(ServerPathFromTools)
#
#         # 选择该数据集使用的查询集合（enron 用 20 个，wiki 用 30 个）
#         if dataset_name.startswith("enron"):
#             queries_texts = ENRON_QUERIES_TEXTS
#             queries_keywords = ENRON_QUERIES_KEYWORDS
#         else:  # wiki_*
#             queries_texts = WIKI_QUERIES_TEXTS
#             queries_keywords = WIKI_QUERIES_KEYWORDS
#
#         # 读取对应数据集的明文搜索结果
#         plain_results = None
#         plain_path = PLAIN_RESULT_PATHS.get(dataset_name)
#         if plain_path and os.path.exists(plain_path):
#             try:
#                 with open(plain_path, "r", encoding="utf-8") as f:
#                     plain_results = json.load(f)
#                 print(f"[INFO] 载入明文结果文件: {plain_path}, type={type(plain_results)}")
#             except Exception as e:
#                 print(f"[WARN] 明文结果文件 {plain_path} 读取失败: {e}")
#                 plain_results = None
#         else:
#             print(f"[WARN] 数据集 {dataset_name} 未配置或找不到明文结果文件，将不计算准确率。")
#
#         trap_times = []
#         search_times = []
#         retrieve_times = []
#         accuracies = []   # 每个查询的 accuracy（可能为 0/非零）
#
#         # 对该数据集的所有查询逐个跑
#         for qi, keywords in enumerate(queries_keywords, start=1):
#             # 找到对应的自然语言查询文本（用于明文 JSON 的 key）
#             if qi - 1 < len(queries_texts):
#                 query_text = queries_texts[qi - 1]
#             else:
#                 query_text = f"Q{qi}"
#
#             print(f"[{dataset_name}] Query #{qi:02d}  \"{query_text}\"  keywords = {keywords}")
#
#             # 针对当前查询，从明文结果里取出对应的结果列表
#             plain_list = None
#             if plain_results is not None and isinstance(plain_results, dict):
#                 # 明文 JSON 是 { "自然语言query": [ {id:..., text:..., ...}, ... ], ... }
#                 if query_text in plain_results:
#                     plain_list = plain_results[query_text]
#
#             t_trap, t_search, t_retrieve, acc = run_one_query(
#                 params, g, pkfs, pkbs, skfs, skbs, skv, rkuv_copy,
#                 EDB, XSet, WSetLocal, IndsLocal,
#                 keywords, plain_list,
#             )
#
#             if acc is not None:
#                 acc_str = f"{acc*100:.2f}%"
#             else:
#                 acc_str = "N/A"
#
#             print(
#                 f"    TrapGen: {t_trap*1000:.2f} ms, "
#                 f"Search: {t_search*10000:.2f} ms, "
#                 # f"Retrieve: {t_retrieve*1000:.2f} ms, "
#                 f"Accuracy: {acc_str}"
#             )
#
#             trap_times.append(t_trap)
#             search_times.append(t_search)
#             retrieve_times.append(t_retrieve)
#             if acc is not None:
#                 accuracies.append(acc)
#
#         results[dataset_name] = {
#             "trap": trap_times,
#             "search": search_times,
#             "retrieve": retrieve_times,
#             "accuracy": accuracies,
#         }
#
#     # ================== 汇总打印 ==================
#     print("\n\n================= SUMMARY (平均耗时 & 准确率统计) =================")
#     print("单位：毫秒 (ms)，准确率为百分比")
#     # print("Accuracy(avg) = 所有查询平均准确率")
#     # print("Accuracy(nonzero) = 准确率>0的查询个数 / 有效查询总数")
#     for dataset_name, data in results.items():
#         trap_avg     = statistics.mean(data["trap"]) * 1000 if data.get("trap") else 0.0
#         search_avg   = statistics.mean(data["search"]) * 10000 if data.get("search") else 0.0
#         # retrieve_avg = statistics.mean(data["retrieve"]) * 1000 if data.get("retrieve") else 0.0
#
#         acc_avg = None
#         acc_nonzero_ratio = None
#         if data.get("accuracy"):
#             acc_list = data["accuracy"]
#             acc_avg = statistics.mean(acc_list) * 100
#             nonzero_count = sum(1 for a in acc_list if a > 0.0)
#             total_count = len(acc_list)
#             acc_nonzero_ratio = (nonzero_count / total_count) * 100 if total_count > 0 else None
#
#         if acc_avg is not None:
#             acc_avg_str = f"{acc_avg:6.2f}%"
#         else:
#             acc_avg_str = "N/A"
#
#         if acc_nonzero_ratio is not None:
#             acc_nz_str = f"{acc_nonzero_ratio:6.2f}%"
#         else:
#             acc_nz_str = "N/A"
#
#         print(
#             f"{dataset_name:10s} | "
#             f"TrapGen: {trap_avg:8.2f} ms, "
#             f"Search: {search_avg:8.2f} ms, "
#             # f"Retrieve: {retrieve_avg:8.2f} ms, "
#             f"Accuracy(avg): {acc_avg_str}, "
#             f"Accuracy(nonzero): {acc_nz_str}"
#         )
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import shutil
import logging
import statistics
import json

from pypbc import Pairing

from tools import (
    ServerPathFromTools,
    ParameterPathFromTools,
    readPP,
    readServerKey,
    readReceiverVKey,
    readWSet,
    readInds,
    loadFile,
)
from TrapGen import TrapGen
    # Search, Copy2UnCopy 来自原代码
from Search import Search, Copy2UnCopy
from Retrieve import Retrieve

logger = logging.getLogger("Caedios")

# ===================== 明文结果文件路径（MS MARCO） =====================

PLAIN_RESULT_PATHS = {
    "msmarco_1k":  "/root/siton-tmp/outputs/plain_results/msmarco_1k.json",
    "msmarco_5k":  "/root/siton-tmp/outputs/plain_results/msmarco_5k.json",
    "msmarco_10k": "/root/siton-tmp/outputs/plain_results/msmarco_10k.json",
}

# ===================== MS MARCO 查询：自然语言文本 & 关键词 =====================

MSMARCO_QUERIES_TEXTS = [
    # "How do you use the Stefan-Boltzmann law to calculate the radius of a star such as Rigel from its luminosity and surface temperature?",
    # "What developmental milestones and typical behaviors should you expect from an 8 year old child at home and at school?",
    "What are the symptoms of a head lice infestation and how can you check for lice, eggs, and nits on a child's scalp?",
    "What special features does the Burj Khalifa in Dubai have and why was it renamed from Burj Dubai?",
    "What kinds of homes and land are for sale near La Grange, California, and what are their typical sizes and prices?",
    "What are the main characteristics, temperament, and exercise needs of the Dogo Argentino dog breed?",
    # "How are custom safety nets used in industry and what kinds of clients and applications does a company like US Netting serve?",
    "What are effective ways to remove weeds from a garden and prevent them from coming back?",
    "How common is urinary incontinence in the United States, what can cause it, and is it just a normal part of aging?",
    "How did President Franklin D. Roosevelt prepare the United States for World War II before Pearl Harbor while the country was still isolationist?",
    # "If you have multiple sclerosis and difficulty swallowing pills, is it safe to crush Valium and other medications to make them easier to swallow?",
    # "What strategies can help you get better results when dealing with customer service representatives at cable companies or airlines?",
    # "In Spanish, what does the word 'machacado' mean and how is the verb 'machacar' used in different contexts?",
    # "When building a concrete path, how should you design and support plywood formwork so that it is strong enough and keeps the concrete in place?",
    # "Why do people join political parties, and which political party did U.S. presidents Woodrow Wilson and Herbert Hoover belong to?",
]

MSMARCO_QUERIES_KEYWORDS = [
    # 1. Stefan-Boltzmann / Rigel
    # ["Stefan-Boltzmann", "Rigel"],

    # 2. 8 岁儿童发展
    # ["developmental", "child"],

    # 3. 头虱 / nits
    ["lice", "nits"],

    # 4. Burj Khalifa
    ["Burj", "Khalifa"],

    # 5. La Grange 房地产
    ["La Grange", "homes"],

    # 6. Dogo Argentino
    ["Dogo", "Argentino"],

    # 7. 工业安全网 / US Netting
    # ["safety", "nets"],

    # 8. 除草
    ["remove", "weeds"],

    # 9. 尿失禁
    ["urinary", "incontinence"],

    # 10. 罗斯福与二战准备
    ["Roosevelt", "World War"],

    # 11. 碾碎药片
    # ["crush", "pills"],

    # 12. 客服交涉策略
    # ["customer", "service"],

    # 13. machacado 词义
    # ["machacado", "machacar"],

    # 14. 混凝土模板
    # ["concrete", "formwork"],

    # 15. 政党 / 总统
    # ["political", "parties"],
]

# ===================== 工具函数 =====================

def build_Q_and_V_from_keywords(keywords):
    """
    给定关键词列表，比如 ["energy", "trading"]
    构造协议里用的 Q / V：
      Q[0] 为空串
      Q[1], Q[2], ... = 对应关键词
      V[i] = 1  (布尔向量，全 1)
    """
    Q = {}
    V = {}

    Q[0] = ""
    for idx, w in enumerate(keywords, start=1):
        Q[idx] = w
        V[idx] = 1

    return Q, V


def load_rkuv_copy():
    """直接返回 rkuv.dat 里 eval 出来的原始 dict（字符串版）"""
    path = os.path.join(ServerPathFromTools, ParameterPathFromTools, "rkuv.dat")
    return loadFile(path)


def copy_index_for_dataset(dataset_name):
    """
    把某个数据集的索引文件拷贝为“默认索引”：
      Server/EDB.dat
      Server/XSet.dat
      Server/Parameter/Inds.dat
      Server/Parameter/WSet.dat
    这样后续的 Copy2UnCopy / readWSet / readInds 都可以复用。
    """
    base = ServerPathFromTools.rstrip("/")

    src_edb   = os.path.join(base, f"EDB_{dataset_name}.dat")
    src_xset  = os.path.join(base, f"XSet_{dataset_name}.dat")
    src_inds  = os.path.join(base, "Parameter", f"Inds_{dataset_name}.dat")
    src_wset  = os.path.join(base, "Parameter", f"WSet_{dataset_name}.dat")

    dst_edb   = os.path.join(base, "EDB.dat")
    dst_xset  = os.path.join(base, "XSet.dat")
    dst_inds  = os.path.join(base, "Parameter", "Inds.dat")
    dst_wset  = os.path.join(base, "Parameter", "WSet.dat")

    # 检查是否存在
    missing = []
    for path in [src_edb, src_xset, src_inds, src_wset]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print("[WARN] 缺少以下索引文件：")
        for m in missing:
            print("   ", m)
        return False

    # 逐个拷贝
    shutil.copyfile(src_edb,  dst_edb)
    shutil.copyfile(src_xset, dst_xset)
    shutil.copyfile(src_inds, dst_inds)
    shutil.copyfile(src_wset, dst_wset)

    logger.info("Copied %s -> %s", src_edb,  dst_edb)
    logger.info("Copied %s -> %s", src_xset, dst_xset)
    logger.info("Copied %s -> %s", src_inds, dst_inds)
    logger.info("Copied %s -> %s", src_wset, dst_wset)

    return True


def normalize_doc_id(x):
    """
    文档 id 归一化：
    - 转成字符串
    - 去掉两侧空白
    - 去掉末尾的点（例如 '58.' -> '58'）
    """
    s = str(x).strip()
    if s.endswith("."):
        s = s[:-1]
    return s


def boolean_filter_plain_results(plain_list, keywords):
    """
    在明文结果列表里，用“布尔 AND”过滤出真正包含所有关键词的文档：
      - plain_list: [ { "id": ..., "text": ... }, ... ]
      - keywords:   当前查询用到的关键词列表

    返回：retrieved_ids_norm: 规范化后的 id 列表（字符串）。
    """
    if not plain_list:
        return []

    kws = [kw.lower() for kw in keywords]
    retrieved_ids = []

    for item in plain_list:
        doc_id = item.get("id", "")
        text = item.get("text", "")
        text_lc = str(text).lower()

        ok = True
        for kw in kws:
            if kw not in text_lc:
                ok = False
                break

        if ok:
            retrieved_ids.append(doc_id)

    retrieved_ids_norm = [normalize_doc_id(did) for did in retrieved_ids]
    return retrieved_ids_norm


# ===================== 单次查询：TrapGen + Search + Retrieve + Accuracy =====================

def run_one_query(params, g, pkfs, pkbs, skfs, skbs, skv, rkuv_copy,
                  EDB, XSet, WSetLocal, IndsLocal,
                  keywords, plain_list=None):
    """
    单次查询：
      - TrapGen / Search / Retrieve 仍然跑，用于计时；
      - 准确率完全基于 plain_list（明文 JSON）和关键词做布尔过滤。
    """
    Q, V = build_Q_and_V_from_keywords(keywords)

    # ---- TrapGen ----
    t0 = time.time()
    tokenfs = TrapGen(params, g, pkfs, pkbs, skv, WSetLocal, Q, V)
    t1 = time.time()
    trap_time = t1 - t0

    # ---- Search ----
    serverKey = [pkfs, skfs, pkbs, skbs]
    t2 = time.time()
    try:
        stokens = Search(params, tokenfs, serverKey, rkuv_copy, EDB, XSet)
        t3 = time.time()
    except Exception as e:
        print(f"[WARN] Search error: {e}")
        stokens = None
        t3 = time.time()
    search_time = t3 - t2

    # ---- Retrieve ----
    t4 = time.time()
    try:
        _ = Retrieve(params, skv, stokens, IndsLocal)
        t5 = time.time()
    except Exception as e:
        print(f"[WARN] Retrieve error: {e}")
        t5 = time.time()
    retrieve_time = t5 - t4

    # ---- Accuracy（只看明文 JSON）----
    accuracy = None
    if plain_list is not None and isinstance(plain_list, list):
        plain_ids_raw = [item.get("id", "") for item in plain_list]
        plain_ids_norm = [normalize_doc_id(x) for x in plain_ids_raw]

        # 用布尔 AND 对 text 做过滤
        retrieved_ids_norm = boolean_filter_plain_results(plain_list, keywords)

        if len(plain_ids_norm) > 0:
            inter = set(plain_ids_norm) & set(retrieved_ids_norm)
            accuracy = len(inter) / len(plain_ids_norm)
        else:
            accuracy = 0.0

    return trap_time, search_time, retrieve_time, accuracy


# ===================== 主流程 =====================

def main():
    # 统一从 Server 侧读公共参数、密钥
    params, g, Kx, Kz, Kl = readPP(ServerPathFromTools)
    pairing = Pairing(params)

    pkfs, skfs, pkbs, skbs = readServerKey(ServerPathFromTools)
    pkv, skv = readReceiverVKey(ServerPathFromTools)
    rkuv_copy = load_rkuv_copy()

    # 实验数据集列表 —— 名字要和 run_all_datasets.py 保存索引时一致
    datasets = [
        ("msmarco_1k",  "/root/siton-tmp/data/msmarco_1k.json"),
        ("msmarco_5k",  "/root/siton-tmp/data/msmarco_5k.json"),
        ("msmarco_10k", "/root/siton-tmp/data/msmarco_10k.json"),
    ]

    # 结果记录： {dataset: {"trap": [...], "search": [...], "retrieve": [...], "accuracy": [...]}}
    results = {}

    for dataset_name, data_path in datasets:
        print(f"\n========== DATASET: {dataset_name} ==========")

        # 先把这个数据集的索引拷贝成“默认索引”
        ok = copy_index_for_dataset(dataset_name)
        if not ok:
            print(f"[SKIP] 数据集 {dataset_name} 缺少索引文件，跳过。")
            continue

        # 根据当前默认索引，构建 EDB / XSet
        print("[DEBUG] Enter Copy2UnCopy")
        EDB, XSet = Copy2UnCopy(params)
        WSetLocal = readWSet(ServerPathFromTools)
        IndsLocal = readInds(ServerPathFromTools)

        # MS MARCO 查询集合
        queries_texts = MSMARCO_QUERIES_TEXTS
        queries_keywords = MSMARCO_QUERIES_KEYWORDS

        # 读取对应数据集的明文搜索结果
        plain_results = None
        plain_path = PLAIN_RESULT_PATHS.get(dataset_name)
        if plain_path and os.path.exists(plain_path):
            try:
                with open(plain_path, "r", encoding="utf-8") as f:
                    plain_results = json.load(f)
                print(f"[INFO] 载入明文结果文件: {plain_path}, type={type(plain_results)}")
            except Exception as e:
                print(f"[WARN] 明文结果文件 {plain_path} 读取失败: {e}")
                plain_results = None
        else:
            print(f"[WARN] 数据集 {dataset_name} 未配置或找不到明文结果文件，将不计算准确率。")

        trap_times = []
        search_times = []
        retrieve_times = []
        accuracies = []   # 每个查询的 accuracy（可能为 0/非零）

        # 对该数据集的所有查询逐个跑
        for qi, keywords in enumerate(queries_keywords, start=1):
            # 对应自然语言查询文本（用于明文 JSON 的 key）
            if qi - 1 < len(queries_texts):
                query_text = queries_texts[qi - 1]
            else:
                query_text = f"Q{qi}"

            print(f"[{dataset_name}] Query #{qi:02d}  \"{query_text}\"  keywords = {keywords}")

            # 针对当前查询，从明文结果里取出对应的结果列表
            plain_list = None
            if plain_results is not None and isinstance(plain_results, dict):
                if query_text in plain_results:
                    plain_list = plain_results[query_text]

            t_trap, t_search, t_retrieve, acc = run_one_query(
                params, g, pkfs, pkbs, skfs, skbs, skv, rkuv_copy,
                EDB, XSet, WSetLocal, IndsLocal,
                keywords, plain_list,
            )

            if acc is not None:
                acc_str = f"{acc*100:.2f}%"
            else:
                acc_str = "N/A"

            print(
                f"    TrapGen: {t_trap*1000:.2f} ms, "
                f"Search: {t_search*1000:.2f} ms, "
                f"Retrieve: {t_retrieve*1000:.2f} ms, "
                f"Accuracy: {acc_str}"
            )

            trap_times.append(t_trap)
            search_times.append(t_search)
            retrieve_times.append(t_retrieve)
            if acc is not None:
                accuracies.append(acc)

        results[dataset_name] = {
            "trap": trap_times,
            "search": search_times,
            "retrieve": retrieve_times,
            "accuracy": accuracies,
        }

    # ================== 汇总打印 ==================
    print("\n\n================= SUMMARY (平均耗时 & 准确率统计) =================")
    print("单位：毫秒 (ms)，准确率为百分比")

    for dataset_name, data in results.items():
        trap_avg     = statistics.mean(data["trap"]) * 1000 if data.get("trap") else 0.0
        search_avg   = statistics.mean(data["search"]) * 10000 if data.get("search") else 0.0
        retrieve_avg = statistics.mean(data["retrieve"]) * 1000 if data.get("retrieve") else 0.0

        acc_avg = None
        acc_nonzero_ratio = None
        if data.get("accuracy"):
            acc_list = data["accuracy"]
            acc_avg = statistics.mean(acc_list) * 100
            nonzero_count = sum(1 for a in acc_list if a > 0.0)
            total_count = len(acc_list)
            acc_nonzero_ratio = (nonzero_count / total_count) * 100 if total_count > 0 else None

        if acc_avg is not None:
            acc_avg_str = f"{acc_avg:6.2f}%"
        else:
            acc_avg_str = "N/A"

        if acc_nonzero_ratio is not None:
            acc_nz_str = f"{acc_nonzero_ratio:6.2f}%"
        else:
            acc_nz_str = "N/A"

        print(
            f"{dataset_name:10s} | "
            f"TrapGen: {trap_avg:8.2f} ms, "
            f"Search: {search_avg:8.2f} ms, "
            f"Retrieve: {retrieve_avg:8.2f} ms, "
            f"Accuracy(avg): {acc_avg_str}, "
            f"Accuracy(nonzero): {acc_nz_str}"
        )


if __name__ == "__main__":
    main()
