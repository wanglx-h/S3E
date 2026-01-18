from tools import *
from pypbc import *

ParameterPath = ParameterPathFromTools

def KeyGenReceiverUThroughFile():
    logger.info("==================KeyGenReceiver Start==================")

    # 1. 读取公共参数
    [params, g, Kx, Kz, Kl] = readPP(ServerPathFromTools)
    pairing = Pairing(params)

    # 2. 在 Zr 上随机一个 alpha（Element）
    alpha = Element.random(pairing, Zr)

    # 3. 用 alpha 的十六进制字符串来得到整数指数
    #    注意：str(alpha) 在 pypbc 里通常是 "0x1234abcd..." 这种形式
    alpha_hex = str(alpha)
    alpha_int = int(alpha_hex, 16)

    # 4. 公钥：pku = g^alpha_int
    pku = g ** alpha_int

    # 5. 私钥结构里仍然保存 Zr 元素 alpha（后面 ReKeyGen / TrapGen 都用它）
    sku = {
        "sk": alpha,   # Zr 元素
        "Kx": Kx,
        "Kz": Kz,
        "Kl": Kl,
    }

    # 6. 写入文件：公钥存 str(pku)，私钥里的 sk 存十六进制字符串（和原始 code1 一样）
    pkuCopy = str(pku)
    skuCopy = {
        "sk": alpha_hex,     # ★ 存十六进制字符串
        "Kx": Kx.decode() if isinstance(Kx, bytes) else str(Kx),
        "Kz": Kz.decode() if isinstance(Kz, bytes) else str(Kz),
        "Kl": Kl.decode() if isinstance(Kl, bytes) else str(Kl),
    }

    uKeyCopy = [pkuCopy, skuCopy]

    createFile(
        ServerPathFromTools + ParameterPath + "uKey.dat",
        str(uKeyCopy),
        "w"
    )
    logger.info("Creating Server/Parameter/uKey.dat file")
    logger.info("==================KeyGenReceiver End==================")

    return [pku, sku]


if __name__ == '__main__':
    KeyGenReceiverUThroughFile()