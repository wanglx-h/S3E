from tools import *
from pypbc import *

ParameterPath=ParameterPathFromTools

def GlobalSetupThroughFile(qbits=512, rbits=160):
    """[KGC generate public parameter]

    Args:
        qbits (int, optional): [description]. Defaults to 512.
        rbits (int, optional): [description]. Defaults to 160.

    Returns:
        [type]: [description]
    """
    logger.info("==================GlobalSetup Start==================")
    # params = Parameters(qbits=qbits, rbits=rbits)   #参数初始化
    param_string = (
        "type a\n"
        "q 8780710799663312522437781984754049815806883199414208211028653399266475630880222"
        "957078625179422662221423155858769582317459277713367317481324925129998224791\n"
        "h 1201601226489114607938882136674053420480295440125131182291961513104720728935970"
        "4531102844802183906537786776\n"
        "r 730750818665451621361119245571504901405976559617\n"
        "exp2 159\n"
        "exp1 107\n"
        "sign1 1\n"
        "sign0 1\n"
    )
    params = Parameters(param_string)

    pairing = Pairing(params)  # 根据参数实例化双线性对
    g = Element.random(pairing, G1)  # g是G1的一个生成元
    Kx=generate_random_str(16).encode('utf-8')
    Kz=generate_random_str(16).encode('utf-8')
    Kl=generate_random_str(16).encode('utf-8')

    paramsCopy = str(params)
    gCopy = str(g)
    KxCopy = str(Kx)
    KzCopy = str(Kz)
    KlCopy = str(Kl)

    ppCopy = [paramsCopy,gCopy,KxCopy,KzCopy,KlCopy]

    createFile(ServerPathFromTools+ParameterPath+"pp.dat",str(ppCopy),"w")
    logger.info("==================GlobalSetup End==================")
    return [params,g,Kx,Kz,Kl]

if __name__ == '__main__':
    GlobalSetupThroughFile()