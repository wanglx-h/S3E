from tools import *
from pypbc import *

ParameterPath = ParameterPathFromTools

# inds = readInds(ServerPathFromTools)


def SearchThroughFile():
    """后端 Server 用 tokenbs 在加密索引上搜索（从文件读入所有数据）"""

    logger.info("==================Search Start==================")

    [params, g, Kx, Kz, Kl] = readPP(ServerPathFromTools)
    pairing = Pairing(params)
    [pkfs, skfs, pkbs, skbs] = readServerKey(ServerPathFromTools)

    # EDB / XSet 从文件版
    EDBCopy = loadFile(ServerPathFromTools + "EDB.dat")
    XSetCopy = loadFile(ServerPathFromTools + "XSet.dat")

    # Zr 中的 1 和 2
    zone = Element.one(pairing, Zr)
    # ztwo = zone + zone

    vector = {}
    count = 0  # 记录有几个文件符合

    # tokenbs（老接口，默认是 4 元结构）
    tokenbsCopy = loadFile(ServerPathFromTools + "tokenbs.dat")
    # 如果只有 3 个元素，则兼容处理（构造一个假的 StatusCopy）
    if len(tokenbsCopy) == 4:
        l, TrapCopy, StatusCopy, booleanVector = tokenbsCopy
    else:
        l, TrapCopy, booleanVector = tokenbsCopy
        StatusCopy = {}
        for i in TrapCopy:
            StatusCopy[i] = {}
            for j in TrapCopy[i]:
                # 没有单独的 Tg，就用 T2 作为占位（保证结构不炸，语义会有偏差）
                StatusCopy[i][j] = TrapCopy[i][j][1]

    # 读 rkuv
    rkuvCopy = loadFile(ServerPathFromTools + ParameterPath + "rkuv.dat")
    rkuv = {}
    rkuv[1] = Element(pairing, G1, str(rkuvCopy[1]))
    rkuv[2] = Element(pairing, G1, str(rkuvCopy[2]))
    rkuv[3] = Element(pairing, G1, str(rkuvCopy[3]))
    rkuv[4] = Element(pairing, G1, str(rkuvCopy[4]))

    eta_elem = skbs["sk"]
    eta_int = int(str(eta_elem))

    # 由于 l 是从 1 开始存的，这里用 <=
    i = 1
    ResCopy = []
    while i <= len(l):
        print("==============")
        if l[i] in EDBCopy.keys():
            logger.info("Judge Trap[%s] UgUe is exist for j in XSet", i)

            e0Copy = EDBCopy[l[i]]["e0"]
            e0 = {}
            e0["a"] = Element(pairing, G1, str(e0Copy["a"]))
            e0["b"] = Element(pairing, GT, str(e0Copy["b"]))

            e1Copy = EDBCopy[l[i]]["e1"]
            e1_int = int(str(e1Copy), 16)
            e1 = Element(pairing, Zr, str(e1_int))

            j = 1
            while j <= len(TrapCopy[i]):
                [T1Copy, T2Copy] = TrapCopy[i][j]
                T1 = Element(pairing, G1, str(T1Copy))
                T2 = Element(pairing, G1, str(T2Copy))

                TgCopy = StatusCopy[i][j]
                Tg = Element(pairing, G1, str(TgCopy))

                # T2^eta
                T2e = T2 ** eta_int

                # eta^2 在 Zr 中：先算 Element，再取整数
                # 直接用整数 2 做指数
                ztwo = 2
                eta2_elem = eta_elem ** ztwo
                eta2_int = int(str(eta2_elem))
                T1e = T1 ** eta2_int

                Te = T2e / T1e  # T_eta = T2^{eta} / T1^{eta^2}

                # rkuv[4]^{e1}
                e1_scalar = int(str(e1))
                part1 = rkuv[4] ** e1_scalar
                part2 = Tg / Te
                UgUe = pairing.apply(part1, part2)

                flag = 0  # 判断本次关键字是否在 XSet 中
                for itemCopy in XSetCopy.values():
                    item = Element(pairing, GT, str(itemCopy))
                    if UgUe == item:
                        flag = 1

                vector[j] = flag
                j += 1

            if vector == booleanVector:
                a = e0["a"]
                b = e0["b"]
                c1 = a
                temp = pairing.apply(a, rkuv[3])
                c2 = b * temp
                c3 = [rkuv[1], rkuv[2]]
                ResCopy.append(
                    [str(c1), str(c2), [str(rkuv[1]), str(rkuv[2])]]
                )
                count += 1
                logger.info("Check Trap[%s] UgUe success", i)
            else:
                logger.info("Check Trap[%s] UgUe fail", i)

        i += 1

    createFile(ServerPathFromTools + "Res.dat", str(ResCopy), "w")
    logger.info("==================Search End==================")
    return count


def Search(params, tokenbsCopy, serverKey, rkuvCopy, EDB, XSet):
    """后端 Server 用 tokenfs/tokenbs 在内存中的 EDB/XSet 上搜索"""

    logger.info("==================Search Start==================")

    pairing = Pairing(params)
    [pkfs, skfs, pkbs, skbs] = serverKey

    zone = Element.one(pairing, Zr)
    # ztwo = zone + zone
    ztwo = 2
    vector = {}
    count = 0  # 记录有几个文件符合

    # 这里 tokenbsCopy 实际上传入的是 TrapGen 返回的 tokenfs：
    #   tokenfs = [l, Trap, boolVector]
    # 老版本 Search 期望的是 [l, TrapCopy, StatusCopy, booleanVector]
    # 所以我们做一个兼容层：
    if len(tokenbsCopy) == 4:
        l, TrapCopy, StatusCopy, booleanVector = tokenbsCopy
    else:
        l, TrapCopy, booleanVector = tokenbsCopy
        StatusCopy = {}
        for i in TrapCopy:
            StatusCopy[i] = {}
            for j in TrapCopy[i]:
                # 用 T2 作为 Tg 的占位，至少保证结构完整不崩
                StatusCopy[i][j] = TrapCopy[i][j][1]

    rkuv = {}
    rkuv[1] = Element(pairing, G1, str(rkuvCopy[1]))
    rkuv[2] = Element(pairing, G1, str(rkuvCopy[2]))
    rkuv[3] = Element(pairing, G1, str(rkuvCopy[3]))
    rkuv[4] = Element(pairing, G1, str(rkuvCopy[4]))

    eta_elem = skbs["sk"]
    eta_int = int(str(eta_elem))

    i = 1
    ResCopy = []
    # 由于 l 是从 1 开始存的，这里用 <=
    while i <= len(l):
        print("==============")
        if l[i] in EDB.keys():
            logger.info("Judge Trap[%s] UgUe is exist for j in XSet", i)
            e0 = EDB[l[i]]["e0"]
            e1 = EDB[l[i]]["e1"]  # 这里本来就是 Zr 元素

            j = 1
            while j <= len(TrapCopy[i]):
                [T1Copy, T2Copy] = TrapCopy[i][j]
                T1 = Element(pairing, G1, str(T1Copy))
                T2 = Element(pairing, G1, str(T2Copy))
                TgCopy = StatusCopy[i][j]
                Tg = Element(pairing, G1, str(TgCopy))

                # T2^eta
                T2e = T2 ** eta_int

                # eta^2
                eta2_elem = eta_elem ** ztwo
                eta2_int = int(str(eta2_elem))
                T1e = T1 ** eta2_int

                Te = T2e / T1e

                # rkuv[4]^{e1}
                e1_scalar = int(str(e1))
                part1 = rkuv[4] ** e1_scalar
                part2 = Tg / Te
                UgUe = pairing.apply(part1, part2)

                flag = 0  # 判断本次关键字是否在 XSet 中
                for item in XSet.values():
                    if UgUe == item:
                        flag = 1

                vector[j] = flag
                j += 1

            if vector == booleanVector:
                a = e0["a"]
                b = e0["b"]
                c1 = a
                temp = pairing.apply(a, rkuv[3])
                c2 = b * temp
                c3 = [rkuv[1], rkuv[2]]
                e = [c1, c2, c3]
                count += 1
                ResCopy.append(
                    [str(c1), str(c2), [str(rkuv[1]), str(rkuv[2])]]
                )
                logger.info("Check Trap[%s] UgUe success", i)
            else:
                logger.info("Check Trap[%s] UgUe fail", i)

        i += 1

    createFile(ServerPathFromTools + "Res.dat", str(ResCopy), "w")
    logger.info("==================Search End==================")
    return ResCopy



def Copy2UnCopy(params):
    """从文件版 EDB/XSet 还原为内存版（Element 形式）"""
    EDB = {}
    XSet = {}
    pairing = Pairing(params)
    EDBCopy = loadFile(ServerPathFromTools + "EDB.dat")
    XSetCopy = loadFile(ServerPathFromTools + "XSet.dat")
    for key in EDBCopy.keys():
        e0Copy = EDBCopy[key]["e0"]
        e0 = {}
        e0["a"] = Element(pairing, G1, str(e0Copy["a"]))
        e0["b"] = Element(pairing, GT, str(e0Copy["b"]))
        e1Copy = EDBCopy[key]["e1"]
        e1_int = int(str(e1Copy), 16)
        e1 = Element(pairing, Zr, str(e1_int))
        EDB[key] = {"e0": e0, "e1": e1}

        itemCopy = XSetCopy[key]
        item = Element(pairing, GT, str(itemCopy))
        XSet[key] = item
    return [EDB, XSet]


if __name__ == "__main__":
    SearchThroughFile()


