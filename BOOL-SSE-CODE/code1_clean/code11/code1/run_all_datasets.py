# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# import os
# import time
# import shutil
# import logging
#
# from GlobalSetup import GlobalSetupThroughFile
# from KeyGenServer import KeyGenServerThroughFile
# from KeyGenReceiverU import KeyGenReceiverUThroughFile
# from KeyGenReceiverV import KeyGenReceiverVThroughFile
# # from ReKeyGen import RetrieveThroughFile
# from ReKeyGen import ReKeyGenThroughFile
# from EDBSetup import AllSetup
#
# logger = logging.getLogger("Caedios")
#
#
# def safe_copy(src, dst):
#     """如果 src 存在就拷贝到 dst。"""
#     if os.path.exists(src):
#         dst_dir = os.path.dirname(dst)
#         if dst_dir and not os.path.exists(dst_dir):
#             os.makedirs(dst_dir, exist_ok=True)
#         shutil.copyfile(src, dst)
#         logger.info("Copied %s -> %s", src, dst)
#
#
# def build_one_dataset(dataset_name, maildir):
#     """对单个数据集构建索引并计时。"""
#     print(f"\n[DATASET {dataset_name}] 开始构建索引，maildir = {maildir}")
#
#     t0 = time.time()
#     AllSetup(maildir)          # ★ 构建索引的真正耗时在这里
#     t1 = time.time()
#     elapsed = t1 - t0
#
#     print(f"[DATASET {dataset_name}] 构建完成，耗时 {elapsed:.3f} 秒")
#
#     # 为每个数据集保存单独的索引文件
#     safe_copy("Server/EDB.dat",               f"Server/EDB_{dataset_name}.dat")
#     safe_copy("Server/XSet.dat",              f"Server/XSet_{dataset_name}.dat")
#     safe_copy("Server/Parameter/Inds.dat",    f"Server/Parameter/Inds_{dataset_name}.dat")
#     safe_copy("Server/Parameter/WSet.dat",    f"Server/Parameter/WSet_{dataset_name}.dat")
#
#
# def main():
#     # 1. 系统初始化和密钥生成（只做一次）
#     print("[GLOBAL] GlobalSetup & KeyGen 阶段开始")
#     GlobalSetupThroughFile()
#     KeyGenServerThroughFile()
#     KeyGenReceiverUThroughFile()
#     KeyGenReceiverVThroughFile()
#     ReKeyGenThroughFile()
#     # ★ 这里原来调用 RetrieveThroughFile()，会去读 Client/Res.dat，建索引阶段没有这个文件
#     # RetrieveThroughFile()
#
#     print("[GLOBAL] GlobalSetup & KeyGen 阶段结束\n")
#
#     # 2. 各个数据集索引构建
#     datasets = [
#         # 根据你真实拥有的数据集来开/关注释
#         # ("enron_1k",  "CSExperiment/enron_1k/maildir/corman-s"),
#         # ("enron_5k",  "CSExperiment/enron_5k/maildir/corman-s"),
#         # ("enron_10k", "CSExperiment/enron_10k/maildir/corman-s"),
#         ("wiki_1k",   "CSExperiment/wiki_1k/maildir/corman-s"),
#         ("wiki_5k",   "CSExperiment/wiki_5k/maildir/corman-s"),
#         ("wiki_10k",  "CSExperiment/wiki_10k/maildir/corman-s"),
#     ]
#
#     for name, maildir in datasets:
#         if not os.path.isdir(maildir):
#             print(f"[WARN] 数据集 {name} 被跳过：目录 {maildir} 不存在")
#             continue
#         build_one_dataset(name, maildir)
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

from GlobalSetup import GlobalSetupThroughFile
from KeyGenServer import KeyGenServerThroughFile
from KeyGenReceiverU import KeyGenReceiverUThroughFile
from KeyGenReceiverV import KeyGenReceiverVThroughFile
# from ReKeyGen import RetrieveThroughFile
from ReKeyGen import ReKeyGenThroughFile
from EDBSetup import AllSetup

logger = logging.getLogger("Caedios")
logger.setLevel(logging.WARNING)

print("当前 Caedios 日志级别 =", logger.level)  # 这里应该打印 30



def safe_copy(src, dst):
    """如果 src 存在就拷贝到 dst。"""
    if os.path.exists(src):
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src, dst)
        logger.info("Copied %s -> %s", src, dst)


def build_one_dataset(dataset_name, mail_or_data_path):
    """
    对单个数据集构建索引并计时。

    注：第二个参数在原版中是 maildir 目录，这里直接传入你配置的
    /root/siton-tmp/data/msmarco_xx.json 路径，AllSetup 内部如果仍假设为目录，
    你需要在 EDBSetup.AllSetup 中做相应适配。
    """
    print(f"\n[DATASET {dataset_name}] 开始构建索引，path = {mail_or_data_path}")

    t0 = time.time()
    AllSetup(mail_or_data_path)      # ★ 构建索引的真正耗时在这里
    t1 = time.time()
    elapsed = t1 - t0

    print(f"[DATASET {dataset_name}] 构建完成，耗时 {elapsed:.3f} 秒")

    # 为每个数据集保存单独的索引文件（文件名中带上 msmarco_xxx）
    safe_copy("Server/EDB.dat",               f"Server/EDB_{dataset_name}.dat")
    safe_copy("Server/XSet.dat",              f"Server/XSet_{dataset_name}.dat")
    safe_copy("Server/Parameter/Inds.dat",    f"Server/Parameter/Inds_{dataset_name}.dat")
    safe_copy("Server/Parameter/WSet.dat",    f"Server/Parameter/WSet_{dataset_name}.dat")


def main():
    # 1. 系统初始化和密钥生成（只做一次）
    print("[GLOBAL] GlobalSetup & KeyGen 阶段开始")
    GlobalSetupThroughFile()
    KeyGenServerThroughFile()
    KeyGenReceiverUThroughFile()
    KeyGenReceiverVThroughFile()
    ReKeyGenThroughFile()
    # ★ 这里原来调用 RetrieveThroughFile()，会去读 Client/Res.dat，建索引阶段没有这个文件
    # RetrieveThroughFile()

    print("[GLOBAL] GlobalSetup & KeyGen 阶段结束\n")

    # 2. MS MARCO 三个数据集索引构建
    datasets = [
        ("msmarco_1k",  "/root/siton-tmp/data/msmarco_1k.json"),
        ("msmarco_5k",  "/root/siton-tmp/data/msmarco_5k.json"),
        ("msmarco_10k", "/root/siton-tmp/data/msmarco_10k.json"),
    ]

    for name, data_path in datasets:
        if not os.path.exists(data_path):
            print(f"[WARN] 数据集 {name} 被跳过：文件 {data_path} 不存在")
            continue
        build_one_dataset(name, data_path)


if __name__ == "__main__":
    main()
