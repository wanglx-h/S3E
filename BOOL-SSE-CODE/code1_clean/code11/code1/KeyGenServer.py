from tools import *
from pypbc import *

ParameterPath = ParameterPathFromTools

def KeyGenServerThroughFile():
    logger.info("==================KeyGenServer Start==================")

    pp = readPP(ServerPathFromTools)
    [params, g, Kx, Kz, Kl] = pp
    pairing = Pairing(params)

    gamma = Element.random(pairing, Zr)
    eta = Element.random(pairing, Zr)

    # 用十六进制串把 Zr 元素映射为 Python 整数（pypbc 原来也是这么存 sk 的）
    gamma_hex = str(gamma)  # 例如 "0x1234abcd..."
    eta_hex = str(eta)

    gamma_int = int(gamma_hex, 16)
    eta_int = int(eta_hex, 16)

    # 用整数指数做幂，规避 pypbc 对 Element 指数的限制
    pkfs = g ** gamma_int
    skfs = {"sk": gamma, "Kx": Kx, "Kz": Kz, "Kl": Kl}

    pkbs = g ** eta_int
    skbs = {"sk": eta, "Kx": Kx, "Kz": Kz, "Kl": Kl}

    pkfsCopy = str(pkfs)
    skfsCopy = {"sk": gamma_hex, "Kx": str(Kx), "Kz": str(Kz), "Kl": str(Kl)}

    pkbsCopy = str(pkbs)
    skbsCopy = {"sk": eta_hex, "Kx": str(Kx), "Kz": str(Kz), "Kl": str(Kl)}

    serverKeyCopy = [pkfsCopy, skfsCopy, pkbsCopy, skbsCopy]
    createFile(ServerPathFromTools + ParameterPath + "serverKey.dat", str(serverKeyCopy), "w")

    logger.info("==================KeyGenServer End==================")
    return [pkfs, skfs, pkbs, skbs]
