from tools import *
from pypbc import *

ParameterPath = ParameterPathFromTools

def ReKeyGenThroughFile():
    """User u 生成重加密密钥 rkuv，并写入 Server/Parameter/rkuv.dat"""

    logger.info("==================ReKeyGen Start==================")

    # 1. 读取公共参数，用这一个 pairing 作为全局基准
    [params, g, Kx, Kz, Kl] = readPP(ServerPathFromTools)
    pairing = Pairing(params)

    # 2. 用同一个 pairing 读取 uKey.dat
    uKeyCopy = loadFile(ServerPathFromTools + ParameterPath + "uKey.dat")
    [pkuCopy, skuCopy] = uKeyCopy
    # 公钥 pku：G1 元素
    pku = Element(pairing, G1, str(pkuCopy))
    # 私钥 alpha：Zr 元素（文件里存的是十六进制串）
    alpha_int = int(skuCopy["sk"], 16)
    alpha_elem = Element(pairing, Zr, str(alpha_int))

    # 3. 用同一个 pairing 读取 vKey.dat
    vKeyCopy = loadFile(ServerPathFromTools + ParameterPath + "vKey.dat")
    [pkvCopy, skvCopy] = vKeyCopy
    pkv = Element(pairing, G1, str(pkvCopy))
    beta_int = int(skvCopy["sk"], 16)
    beta_elem = Element(pairing, Zr, str(beta_int))  # 虽然这里暂时用不上，先构造好

    rkuv = {}

    # Zr 中的 1
    zone = Element.one(pairing, Zr)

    # 4. 重加密随机性：在 Zr 上随机一个元素 r3，然后用对应的整数做指数
    r3_elem = Element.random(pairing, Zr)
    r3_int = int(str(r3_elem))

    # X 是 G1 上的随机元素（和 pkv 属于同一 pairing）
    X = Element.random(pairing, G1)

    # alpha 的整数形式
    alpha_int = int(str(alpha_elem))

    # -------- H(g^alpha) --------
    hashE0 = Element.from_hash(
        pairing,
        G1,
        Hash1(str(pku).encode()).digest()   # 第三个参数必须是 bytes
    )

    # temp = H(g^alpha)^(-alpha)
    temp = hashE0 ** (-alpha_int)

    # -------- H(X) --------
    hash_value = Element.from_hash(
        pairing,
        G1,
        Hash1(str(X).encode("utf-8")).digest()
    )

    # 5. 计算 rkuv 的 4 个分量

    # rkuv[1] = g^r3
    rkuv[1] = g ** r3_int

    # rkuv[2] = X * pkv^r3   —— 都在 G1 上，用的是同一个 pairing
    rkuv[2] = X * (pkv ** r3_int)

    # rkuv[3] = H(g^alpha)^(-alpha) * H(X)
    rkuv[3] = temp * hash_value

    # rkuv[4] = pkv^{1/alpha}
    # 先在 Zr 中算 1/alpha，再把对应整数拿出来当指数用
    inv_alpha_elem = zone / alpha_elem          # Zr 中的 1/alpha
    inv_alpha_int = int(str(inv_alpha_elem))    # 用这个整数做指数
    rkuv[4] = pkv ** inv_alpha_int

    # 6. 写入文件（全部转成字符串）
    rkuvCopy = {
        1: str(rkuv[1]),
        2: str(rkuv[2]),
        3: str(rkuv[3]),
        4: str(rkuv[4]),
    }

    createFile(
        ServerPathFromTools + ParameterPath + "rkuv.dat",
        str(rkuvCopy),
        "w"
    )

    logger.info("==================ReKeyGen End==================")
    return rkuv


if __name__ == '__main__':
    ReKeyGenThroughFile()
