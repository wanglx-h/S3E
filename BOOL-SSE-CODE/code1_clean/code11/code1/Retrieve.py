from tools import *
from pypbc import *

ParameterPath = ParameterPathFromTools

# inds = readInds(ClientPathFromTools)


def RetrieveThroughFile():
    """从 Client/Res.dat 中读取搜索结果，生成需要解密的文件列表，并返回文档 ID 列表"""

    logger.info("==================Retrieve Start==================")
    [params, g, Kx, Kz, Kl] = readPP(ClientPathFromTools)
    pairing = Pairing(params)

    vKey = readReceiverVKey(ClientPathFromTools)
    [pkv, skv] = vKey
    beta = skv["sk"]

    filesToReceive = []

    # 加密搜索结果
    ResCopy = loadFile(ClientPathFromTools + "Res.dat")
    Res = ResCopy

    inds = readInds(ClientPathFromTools)

    if Res:
        for item in Res:
            # item = [c1_str, c2_str, [a_str, b_str]]
            c1 = Element(pairing, G1, str(item[0]))
            c2 = Element(pairing, GT, str(item[1]))
            c3 = item[2]

            a = Element(pairing, G1, str(c3[0]))
            b = Element(pairing, G1, str(c3[1]))

            # X = b / a^beta
            X = b / (a ** beta)

            # hash_value = H1(X)
            hash_value = Element.from_hash(
                pairing,
                G1,
                Hash1(str(X).encode("utf-8")).digest()
            )

            # temp = e(c1, H(X))
            temp = pairing.apply(c1, hash_value)

            # m = c2 / temp
            m = c2 / temp

            # 利用 m 在 Inds 中查回 ind, Kind, iv
            ind = inds[str(m)][1]
            Kind = inds[str(m)][2]
            iv = inds[str(m)][3]

            filesToReceive.append([ind, Kind, iv])
    else:
        logger.info("Res is null")

    # 记录到 passInd.dat
    createFile(ClientPathFromTools + "passInd.dat", str(filesToReceive), "w")

    # 返回文档 ID 列表，供准确率计算使用
    doc_ids = [ind for (ind, Kind, iv) in filesToReceive]
    logger.info("==================Retrieve End==================")
    return doc_ids


def RetrieveDecFiles():
    """根据 passInd.dat，把加密邮件解密成明文文件"""

    srcpath = ClientPathFromTools + MailEncPathFromTools
    dstpath = ClientPathFromTools + MailDecPathFromTools
    copy = loadFile(ClientPathFromTools + "passInd.dat")

    for t in copy:
        ind = t[0]
        Kind = t[1]
        iv = t[2]
        file = open(srcpath + ind, "r")
        dataEnc = file.read()
        dataDec = decrypt(dataEnc, Kind, iv)
        createFile(dstpath + ind, dataDec, "w")


def Retrieve(params, skv, ResCopy, inds):
    """
    实验用接口：直接用内存中的 ResCopy 和 IndsLocal，
    返回文档 ID 列表 doc_ids，供 run_all_queries.py 做准确率统计。
    """

    pairing = Pairing(params)
    beta = skv["sk"]

    filesToReceive = []
    Res = ResCopy

    if Res:
        for item in Res:
            # item = [c1_str, c2_str, [a_str, b_str]]
            c1 = Element(pairing, G1, str(item[0]))
            c2 = Element(pairing, GT, str(item[1]))
            c3 = item[2]

            a = Element(pairing, G1, str(c3[0]))
            b = Element(pairing, G1, str(c3[1]))

            # X = b / a^beta
            X = b / (a ** beta)

            # hash_value = H1(X)
            hash_value = Element.from_hash(
                pairing,
                G1,
                Hash1(str(X).encode("utf-8")).digest()
            )

            # temp = e(c1, H(X))
            temp = pairing.apply(c1, hash_value)

            # m = c2 / temp
            m = c2 / temp

            ind = inds[str(m)][1]
            Kind = inds[str(m)][2]
            iv = inds[str(m)][3]

            filesToReceive.append([ind, Kind, iv])
    else:
        logger.info("Res is null")

    # 仍然可以顺便写一份 passInd.dat（看你是否需要）
    createFile(ClientPathFromTools + "passInd.dat", str(filesToReceive), "w")

    logger.info("==================Retrieve End==================")

    # 返回文档ID列表
    doc_ids = [ind for (ind, Kind, iv) in filesToReceive]
    return doc_ids


if __name__ == '__main__':
    RetrieveThroughFile()
    RetrieveDecFiles()
