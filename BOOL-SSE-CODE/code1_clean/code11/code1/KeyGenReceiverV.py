from tools import *
from pypbc import *

ParameterPath = ParameterPathFromTools

def KeyGenReceiverVThroughFile():
    logger.info("==================KeyGenReceiverV Start==================")

    [params, g, Kx, Kz, Kl] = readPP(ServerPathFromTools)
    pairing = Pairing(params)

    beta = Element.random(pairing, Zr)
    beta_hex = str(beta)
    beta_int = int(beta_hex, 16)

    pkv = g ** beta_int
    skv = {"sk": beta, "Kx": Kx, "Kz": Kz, "Kl": Kl}

    pkvCopy = str(pkv)
    skvCopy = {
        "sk": beta_hex,
        "Kx": Kx.decode() if isinstance(Kx, bytes) else str(Kx),
        "Kz": Kz.decode() if isinstance(Kz, bytes) else str(Kz),
        "Kl": Kl.decode() if isinstance(Kl, bytes) else str(Kl),
    }

    vKeyCopy = [pkvCopy, skvCopy]
    createFile(
        ServerPathFromTools + ParameterPath + "vKey.dat",
        str(vKeyCopy),
        "w"
    )

    logger.info("Creating Server/Parameter/vKey.dat file")
    logger.info("==================KeyGenReceiverV End==================")

    return [pkv, skv]



if __name__ == '__main__':
    KeyGenReceiverVThroughFile()