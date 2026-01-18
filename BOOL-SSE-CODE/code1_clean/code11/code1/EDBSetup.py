from tools import *
from pypbc import *
from pathlib import Path
from email.parser import Parser
from nltk.tokenize import *
from collections import Counter
import re
import json
import os
import spacy
import pytextrank

ParameterPath = ParameterPathFromTools

# ====== 自定义一个简单的分词函数，避免依赖 nltk 的 punkt ======
def word_tokenize(text):
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    # 仅保留 a-z0-9 的连续串作为 token
    return re.findall(r"[a-z0-9]+", text)

_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with",
    "is","are","was","were","be","been","being","am",
    "this","that","these","those","it","its","as","from","into","about","over","under","between",
    "i","you","he","she","we","they","me","him","her","us","them","my","your","his","their","our",
    "what","which","who","whom","why","how",
    "do","does","did","doing","done",
    "can","could","should","would","may","might","must","will","shall",
    "not","no","nor","so","too","very",
}

def msmarco_top_keywords(text: str, k: int = 15):
    """
    从 MS MARCO 的 text 中抽取 Top-K 关键词（10-15个建议）。
    - 规范化：只保留 [a-z0-9]+
    - 过滤：停用词 / 纯数字 / 过短过长
    - 选择：按词频取 Top-K（稳定可复现实验规模）
    """
    toks = word_tokenize(text)

    filtered = []
    for t in toks:
        if len(t) < 3 or len(t) > 24:
            continue
        if t.isdigit():
            continue
        if t in _STOPWORDS:
            continue
        filtered.append(t)

    if not filtered:
        return []

    cnt = Counter(filtered)

    # 词频降序，词典序作为 tie-break，保证稳定
    ranked = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))

    # 取 Top-K
    top = [w for w, _ in ranked[:k]]

    return top

# ============================================================

# 全局容器（每次 AllSetup 会重置）
WSet = {}
IndsCopy = {}
EDBCopy = {}
XSetCopy = {}


def Get(delta, w):
    """WSet 中关键字 w 的计数"""
    c = 0
    if w in WSet.keys():
        c = WSet[w]
    else:
        WSet[w] = 0
    return c


def Update(delta, w, c):
    """更新 WSet[w] 的计数"""
    state = 0
    if w in WSet.keys():
        WSet[w] = c
        state = 1
    return state


def EDBSetup(params, g, pkfs, pkbs, pku, sku, D, EDB, XSet):
    """
    User u 加密文件，生成 EDB 和 XSet

    params: pairing 参数
    g:      G1 生成元（Element）
    pkfs:   前端服务器公钥
    pkbs:   后端服务器公钥
    pku:    用户 u 公钥
    sku:    用户 u 私钥
    D:      (filePath, WindSet)
    EDB:    现有 EDB 字典
    XSet:   现有 XSet 字典
    """

    pairing = Pairing(params)
    alpha = sku["sk"]  # Element(Zr)

    [filePath, WindSet] = D

    ind = generate_random_str(32)
    try:
        fileOrigin = open(filePath, "r")
        fileData = fileOrigin.read()
    except Exception as e:
        logger.info(e)
        return [EDB, XSet]

    Kind = generate_random_str(32)
    iv = generate_random_str(16)
    fileEncrypted = encrypt(fileData, Kind, iv)

    path = ServerPathFromTools + MailEncPathFromTools
    createFile(path + ind, fileEncrypted, "wb")

    # 更新 WSet 计数
    for keyword, content in WindSet.items():
        c = Get(0, content)
        c = c + 1
        Update(0, content, c)

    # 计算 xind ∈ Zr
    indByte = str(ind).encode()
    xind = PRF_Fp(params, sku["Kx"], indByte)  # Element(Zr)

    for keyword, content in WindSet.items():
        r1 = Element.random(pairing, Zr)  # 随机 Zr
        c = Get(0, content)

        cw = str(c).encode("utf-8")
        w = str(content).encode("utf-8")

        # l 是 PRF 输出（用作 EDB/XSet 的索引）
        l = PRF_F(sku["Kl"], w + cw)
        z = PRF_Fp(params, sku["Kz"], w + cw)  # Element(Zr)

        # m ∈ GT
        m = pairing.apply(Element.random(pairing, G1), Element.random(pairing, G1))
        IndsCopy[str(m)] = [str(filePath), str(ind), str(Kind), str(iv)]

        # a 直接随机取一个 G1 元素即可（和 g^r1 分布一样）
        a = Element.random(pairing, G1)

        # hashE0 = H(g^alpha) ∈ G1
        hashE0 = Element.from_hash(
            pairing, G1, Hash1(str(pku).encode()).digest()
        )

        # alpha 用作指数时，转成 int
        alpha_int = int(alpha)
        temp = pairing.apply(a, hashE0 ** alpha_int)

        # b = m * temp ∈ GT
        b = m * temp

        # e0 = (a, b)
        e0 = {"a": a, "b": b}

        # ===== e1：先在整数域里相乘，再放回 Zr =====
        e1_int = int(xind) * int(z) * int(alpha)
        e1 = Element(pairing, Zr, str(e1_int))
        # ==========================================

        EDB[l] = {"e0": e0, "e1": e1}
        EDBCopy[l] = {"e0": {"a": str(a), "b": str(b)}, "e1": str(e1)}

        # hash_value = H1(w) ∈ G1
        hash_value = Element.from_hash(
            pairing, G1, Hash1(w).digest()
        )

        # ===== xtag：把底数统一“投”到 G1 再做 pairing =====
        xind_int = int(xind)
        base1 = Element(pairing, G1, str(pkfs / pkbs))
        base2 = Element(pairing, G1, str(hash_value ** xind_int))
        xtag = pairing.apply(base1, base2)
        # ===============================================

        XSet[l] = xtag
        XSetCopy[l] = str(xtag)

    return [EDB, XSet]

def GetDList_from_msmarco_json(json_path: Path):
    """
    针对 MS MARCO 格式的 JSON:
      [
        {"doc_id": "...", "text": "..."},
        ...
      ]

    - 为每条记录生成一个 .txt 文件:  {json_dir}/{json_stem}/{doc_id}.txt
    - 用 text 分词生成关键字字典 D
    - 返回 {file_path: D} 的字典
    """
    logger.info("========== GetDList_from_msmarco_json: %s ==========", json_path)

    DList = {}

    # 例如: /root/.../msmarco_1k.json -> /root/.../msmarco_1k/
    out_dir = json_path.parent / json_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读 JSON（你当前给的是 list 格式）
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cnt = 0
    for item in data:
        doc_id = str(item.get("doc_id", "")).strip()
        text = item.get("text", "")
        if not doc_id or not text:
            continue

        # 为每条记录创建一个 txt 文件
        file_path = out_dir / f"{doc_id}.txt"
        with open(file_path, "w", encoding="utf-8") as fw:
            fw.write(text)

        # 构造关键字字典 D
        D = {}
        D["DOC_ID"] = doc_id  # 你需要的话可以在搜索阶段用这个字段

        keywords = msmarco_top_keywords(text, k=10)
        for w in keywords:
            D[w] = w

        DList[file_path] = D
        cnt += 1

    logger.info("GetDList_from_msmarco_json: total %d docs", cnt)
    logger.info("========== GetDList_from_msmarco_json End ==========")
    return DList



# def GetDList(dir):
#     """递归读取邮件文件，构造每个文件的属性字典 D"""
#
#     logger.info("==================GetDList Start==================")
#
#     p = Path(dir)
#     DList = {}
#     # 递归匹配所有普通文件（包括没有点号/后缀的）
#     FileList = [f for f in p.glob("**/*") if f.is_file()]
#
#     for filepath in FileList:
#         logger.info("Reading %s", filepath)
#
#         D = {}
#
#         f = open(filepath, "rb+")
#         byt = f.read()
#         data = byt.decode("ISO-8859-1")
#         email = Parser().parsestr(data)
#
#         D["Message-ID"] = email["Message-ID"]
#         D["Date"] = email["Date"]
#         D["From"] = email["From"]
#         D["X-FileName"] = email["X-FileName"]
#         D["X-Origin"] = email["X-Origin"]
#         D["X-From"] = email["X-From"]
#         D["X-Folder"] = email["X-Folder"]
#         toMails = email["To"]
#         toMailsList = re.split("[,\\s]", str(toMails))
#         for mail in toMailsList:
#             if mail:
#                 D[mail] = mail
#
#         # 针对 subject 做分词
#         subject = email["subject"]
#         words = word_tokenize(subject)
#         for word in words:
#             D[word] = word
#
#         DList[filepath] = D
#
#     logger.info("==================GetDList End==================")
#     return DList
def GetDList(dir):
    """递归读取邮件文件，或从 MS MARCO JSON 读取，构造每个文件的属性字典 D"""

    logger.info("==================GetDList Start==================")

    p = Path(dir)

    # ---------- 分支 1：如果是 MS MARCO JSON 文件 ----------
    if p.is_file() and p.suffix.lower() == ".json":
        DList = GetDList_from_msmarco_json(p)
        logger.info("==================GetDList End (json)==================")
        return DList

    # ---------- 分支 2：原始 Enron maildir 逻辑 ----------
    DList = {}
    # 递归匹配所有普通文件（包括没有点号/后缀的）
    FileList = [f for f in p.glob("**/*") if f.is_file()]

    for filepath in FileList:
        logger.info("Reading %s", filepath)

        D = {}

        f = open(filepath, "rb+")
        byt = f.read()
        data = byt.decode("ISO-8859-1")
        email = Parser().parsestr(data)

        D["Message-ID"] = email["Message-ID"]
        D["Date"] = email["Date"]
        D["From"] = email["From"]
        D["X-FileName"] = email["X-FileName"]
        D["X-Origin"] = email["X-Origin"]
        D["X-From"] = email["X-From"]
        D["X-Folder"] = email["X-Folder"]
        toMails = email["To"]
        toMailsList = re.split("[,\\s]", str(toMails))
        for mail in toMailsList:
            if mail:
                D[mail] = mail

        # 针对 subject 做分词
        subject = email["subject"]
        words = word_tokenize(subject)
        for word in words:
            D[word] = word

        DList[filepath] = D

    logger.info("==================GetDList End==================")
    return DList



def AllSetup(maildir="CSExperiment/maildir/corman-s"):
    global WSet, IndsCopy, EDBCopy, XSetCopy

    # 每次构建一个数据集前，重置这几个全局容器
    WSet = {}
    IndsCopy = {}
    EDBCopy = {}
    XSetCopy = {}

    """
    构建一个数据集的 EDB / XSet，并把索引写到：
      Server/EDB.dat
      Server/XSet.dat
      Server/Parameter/Inds.dat
      Server/Parameter/WSet.dat

    maildir：邮件所在的目录，比如 "CSExperiment/enron_5k/maildir/corman-s"
    """
    DList = GetDList(maildir)

    [params, g, Kx, Kz, Kl] = readPP(ServerPathFromTools)

    serverKey = readServerKey(ServerPathFromTools)
    [pkfs, skfs, pkbs, skbs] = serverKey

    uKey = readReceiverUKey(ServerPathFromTools)
    [pku, sku] = uKey

    EDB = {}
    XSet = {}
    logger.info("==================EDBSetup Start==================")
    for D in DList.items():
        [EDB, XSet] = EDBSetup(params, g, pkfs, pkbs, pku, sku, D, EDB, XSet)
    logger.info("==================EDBSetup End==================")

    createFile(ServerPathFromTools + "EDB.dat", str(EDBCopy), "w")
    createFile(ServerPathFromTools + "XSet.dat", str(XSetCopy), "w")
    createFile(
        ServerPathFromTools + ParameterPath + "Inds.dat", str(IndsCopy), "w"
    )
    createFile(
        ServerPathFromTools + ParameterPath + "WSet.dat", str(WSet), "w"
    )


if __name__ == "__main__":
    AllSetup()
