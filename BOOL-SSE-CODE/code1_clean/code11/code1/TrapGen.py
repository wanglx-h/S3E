from tools import *
from pypbc import *

ParameterPath = ParameterPathFromTools
TrapCopy = {}
WSet = {}  # readWSet(ClientPathFromTools)


def Get(delta, w, WSet):
    """The number of keyword w in DB"""
    c = 0
    if w in WSet.keys():
        c = WSet[w]
    else:
        WSet[w] = 0
    return c


def TrapGenThroughFile():
    """User v generate token to server (写文件版)"""
    logger.info("==================TrapGen Start==================")
    Copy = loadFile(ClientPathFromTools + "para.dat")
    [Q, V] = Copy
    logTime.info("Generate Trap for %s", Q)
    print(Q)

    # 仅用于做实验：所有关键字都置为 1
    i = 1
    V = {}
    V[0] = 1
    while i < len(Q):
        V[i] = 1
        i += 1

    [params, g, Kx, Kz, Kl] = readPP(ClientPathFromTools)
    pairing = Pairing(params)

    # 把 g 投到 G1
    g_g1 = Element(pairing, G1, str(g))

    serverKey = readServerKey(ClientPathFromTools)
    [pkfs, skfs, pkbs, skbs] = serverKey

    # 把 pkfs、pkbs 也统一投到 G1
    pkfs_g1 = Element(pairing, G1, str(pkfs))
    pkbs_g1 = Element(pairing, G1, str(pkbs))

    vKey = readReceiverVKey(ClientPathFromTools)
    [pkv, skv] = vKey

    WSetLocal = readWSet(ClientPathFromTools)

    zone = Element.one(pairing, Zr)
    beta = skv["sk"]  # 理论上是 Zr

    i = 1
    n = Get(0, Q[i], WSetLocal)

    logger.info("Trap length is %s", n)
    q = len(Q)
    logger.info("Send to Front Server for %s times", n)
    logger.info("Query Keywords = %s", Q)
    logger.info("Query keyword number = %d", q - 1)

    l = {}
    Trap = {}
    boolVector = {}

    i = 1
    # 因为有个 Q[0] 是空，所以 <
    while i < q:
        boolVector[i] = V[i]
        i += 1

    i = 1
    while i <= n:
        Trap[i] = {}
        TrapCopy[i] = {}
        i += 1

    i = 1
    while i <= n:
        w1 = str(Q[1]).encode("utf-8")
        c = str(i).encode("utf-8")
        l[i] = PRF_F(skv["Kl"], w1 + c)
        z = PRF_Fp(params, skv["Kz"], w1 + c)  # 按设定应在 Zr

        j = 1
        # 因为有个 Q[0] 是空，所以 <
        while j < q:
            wj = str(Q[j]).encode("utf-8")
            r2 = Element.random(pairing, Zr)
            r2_int = int(r2)  # 作为指数使用的整数

            # T1 = g^r2  ->  g_g1 ** int(r2)
            T1 = g_g1 ** r2_int

            hash_value = Element.from_hash(
                pairing, G1, Hash1(wj).digest()
            )  # H1(wj)

            # === 关键修改：把 beta / z 统一“投到” Zr 再做乘除 ===
            beta_zr = Element(pairing, Zr, str(beta))
            z_zr = Element(pairing, Zr, str(z))
            inv = zone / (beta_zr * z_zr)  # Element(Zr)
            inv_int = int(inv)
            # ==================================================

            # pkfs、pkbs 用 pkfs_g1 / pkbs_g1
            T2 = (hash_value ** inv_int) * ((pkfs_g1 * pkbs_g1) ** r2_int)

            trap = [T1, T2]
            Trap[i][j] = trap
            TrapCopy[i][j] = [str(T1), str(T2)]
            j += 1
        i += 1

    tokenfs = [l, Trap, boolVector]
    tokenfsCopy = [l, TrapCopy, boolVector]
    createFile(ClientPathFromTools + "tokenfs.dat", str(tokenfsCopy), "w")
    logger.info("==================TrapGen End==================")
    return tokenfs


def TrapGen(params, g, pkfs, pkbs, skv, WSetLocal, Q, V):
    """User v generate token to server (内存版，run_all_queries 里调用)"""
    pairing = Pairing(params)
    zone = Element.one(pairing, Zr)
    beta = skv["sk"]

    # 把 g, pkfs, pkbs 统一还原到 G1
    g_g1 = Element(pairing, G1, str(g))
    pkfs_g1 = Element(pairing, G1, str(pkfs))
    pkbs_g1 = Element(pairing, G1, str(pkbs))

    i = 1
    n = Get(0, Q[i], WSetLocal)

    logger.info("Trap length is %s", n)
    q = len(Q)
    logger.info("Send to Front Server for %s times", n)
    logger.info("Query Keywords = %s", Q)
    logger.info("Query keyword number = %d", q - 1)

    l = {}
    Trap = {}
    boolVector = {}

    i = 1
    # 因为有个 Q[0] 是空，所以 <
    while i < q:
        boolVector[i] = V[i]
        i += 1

    i = 1
    while i <= n:
        Trap[i] = {}
        TrapCopy[i] = {}
        i += 1

    i = 1
    while i <= n:
        w1 = str(Q[1]).encode("utf-8")
        c = str(i).encode("utf-8")
        l[i] = PRF_F(skv["Kl"], w1 + c)
        z = PRF_Fp(params, skv["Kz"], w1 + c)

        j = 1
        # 因为有个 Q[0] 是空，所以 <
        while j < q:
            wj = str(Q[j]).encode("utf-8")
            r2 = Element.random(pairing, Zr)
            r2_int = int(r2)

            T1 = g_g1 ** r2_int

            hash_value = Element.from_hash(
                pairing, G1, Hash1(wj).digest()
            )

            beta_zr = Element(pairing, Zr, str(beta))
            z_zr = Element(pairing, Zr, str(z))
            inv = zone / (beta_zr * z_zr)
            inv_int = int(inv)

            T2 = (hash_value ** inv_int) * ((pkfs_g1 * pkbs_g1) ** r2_int)

            trap = [T1, T2]
            Trap[i][j] = trap
            TrapCopy[i][j] = [str(T1), str(T2)]
            j += 1
        i += 1

    tokenfs = [l, Trap, boolVector]
    tokenfsCopy = [l, TrapCopy, boolVector]
    createFile(ClientPathFromTools + "tokenfs.dat", str(tokenfsCopy), "w")
    logger.info("==================TrapGen End==================")
    return tokenfs


if __name__ == "__main__":
    TrapGenThroughFile()
