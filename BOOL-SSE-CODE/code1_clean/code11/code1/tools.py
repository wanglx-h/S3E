import logging
import os
import random
from pypbc import *

from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex

import hashlib
Hash1 = hashlib.sha256
Hash2 = hashlib.sha256

ParameterPathFromTools="Parameter/"
ServerPathFromTools="Server/"
ClientPathFromTools="Client/"
MailEncPathFromTools="MailEnc/"
MailDecPathFromTools="MailDec/"

logger=logging.getLogger("Caedios")
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
file_handler = logging.FileHandler("log")
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logTime=logging.getLogger("logTime")
logTime.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
fileTime_handler = logging.FileHandler("logTime")
fileTime_handler.setLevel(level=logging.INFO)
fileTime_handler.setFormatter(formatter)
logTime.addHandler(fileTime_handler)

def writeFile(filePath,data):
    logger.info("Creating %s file",filePath)
    fileCopy=open(filePath,"w")
    fileCopy.write(str(data))

def createFile(dstpath,data,type):
    """[给定一个文件路径,自动创建文件夹并新建文件]

    Args:
        data ([str|byte]): [Depends on parameter type]
        dstpath ([str]): [The file path you want to create]
        type ([str]): [r,w,x,b]

    Returns:
        [int]: [Not in use]
    """
    logger.info("Creating %s file",dstpath)
    path=dstpath.split("/")
    i=0
    temp=""
    while(i<len(path)):
        temp += path[i]
        if(i+1<len(path)):#代表这就是个文件夹
            if(not os.path.exists(temp)):
                os.mkdir(temp)
            temp+="/"
        else:#即文件
            logging.info("Creating %s",temp)
            f=open(temp,type)
            f.write(data)
        i+=1

    return 0

def createDir(dstpath):
    """[递归创建文件夹]

    Args:
        dstpath ([type]): [description]
        type ([type]): [description]
    """
    path=dstpath.split("/")
    i=0
    temp=""
    while(i<len(path)):
        temp += path[i]
        if(i+1<len(path)):#代表这就是个文件夹
            if(not os.path.exists(temp)):
                os.mkdir(temp)
            temp+="/"
        
        i+=1

def loadFile(filePath):
    logger.info("Reading %s file",filePath)
    fileCopy=open(filePath,"r")
    Copy=eval(fileCopy.read())
    return Copy

# def readPP(root):
#     """[summary]
#
#     Args:
#         root ([type]): [Client/ or Server/]
#
#     Returns:
#         [type]: [description]
#     """
#     ppCopy=loadFile(root+ParameterPathFromTools+"pp.dat")
#     [paramsCopy,gCopy,KxCopy,KzCopy,KlCopy]=ppCopy
#
#     params = Parameters(paramsCopy)
#     pairing = Pairing(params)
#     g=Element(pairing,G1,value=gCopy)
#
#     Kx=KxCopy.encode()
#     Kz=KzCopy.encode()
#     Kl=KlCopy.encode()
#
#     pp=[params,g,Kx,Kz,Kl]
#
#     return pp
def readPP(root):
    ppCopy = loadFile(root + ParameterPathFromTools + "pp.dat")
    [paramsCopy, gCopy, KxCopy, KzCopy, KlCopy] = ppCopy

    params = Parameters(paramsCopy)
    pairing = Pairing(params)

    # 原来是 g = Element(pairing, G1, value=gCopy)
    g = Element(pairing, G1, gCopy)

    Kx = KxCopy.encode()
    Kz = KzCopy.encode()
    Kl = KlCopy.encode()

    return [params, g, Kx, Kz, Kl]


# def readServerKey(root):
#     [params,g,Kx,Kz,Kl]=readPP(root)
#     pairing = Pairing(params)
#
#     serverKeyCopy=loadFile(root+ParameterPathFromTools+"serverKey.dat")
#     [pkfsCopy,skfsCopy,pkbsCopy,skbsCopy]=serverKeyCopy
#     skfsCopy['Kx']=Kx
#     skfsCopy['Kz']=Kz
#     skfsCopy['Kl']=Kl
#     skbsCopy['Kx']=Kx
#     skbsCopy['Kz']=Kz
#     skbsCopy['Kl']=Kl
#
#     pkfs=Element(pairing,G1,value=pkfsCopy)
#     skfs=skfsCopy
#     skfs['sk']=Element(pairing,Zr,value=int(skfsCopy['sk'],16))
#
#     pkbs=Element(pairing,G1,value=pkbsCopy)
#     skbs=skbsCopy
#     skbs['sk']=Element(pairing,Zr,value=int(skbsCopy['sk'],16))
#
#     serverKey=[pkfs,skfs,pkbs,skbs]
#
#     return serverKey
def readServerKey(root):
    [params, g, Kx, Kz, Kl] = readPP(root)
    pairing = Pairing(params)

    serverKeyCopy = loadFile(root + ParameterPathFromTools + "serverKey.dat")
    [pkfsCopy, skfsCopy, pkbsCopy, skbsCopy] = serverKeyCopy

    skfsCopy["Kx"] = Kx
    skfsCopy["Kz"] = Kz
    skfsCopy["Kl"] = Kl
    skbsCopy["Kx"] = Kx
    skbsCopy["Kz"] = Kz
    skbsCopy["Kl"] = Kl

    pkfs = Element(pairing, G1, str(pkfsCopy))
    pkbs = Element(pairing, G1, str(pkbsCopy))

    gamma_int = int(skfsCopy["sk"], 16)
    eta_int   = int(skbsCopy["sk"], 16)

    gamma = Element(pairing, Zr, str(gamma_int))
    eta   = Element(pairing, Zr, str(eta_int))

    skfs = skfsCopy
    skfs["sk"] = gamma

    skbs = skbsCopy
    skbs["sk"] = eta

    return [pkfs, skfs, pkbs, skbs]


def readReceiverUKey(root):
    [params, g, Kx, Kz, Kl] = readPP(root)
    pairing = Pairing(params)

    uKeyCopy = loadFile(root + ParameterPathFromTools + "uKey.dat")
    [pkuCopy, skuCopy] = uKeyCopy
    skuCopy["Kx"] = Kx
    skuCopy["Kz"] = Kz
    skuCopy["Kl"] = Kl

    # 公钥：G1 元素，第三个参数必须是字符串
    pku = Element(pairing, G1, str(pkuCopy))

    # 私钥：先按 16 进制转整数，再转成字符串喂给 Element
    alpha_int = int(skuCopy["sk"], 16)
    alpha = Element(pairing, Zr, str(alpha_int))

    sku = skuCopy
    sku["sk"] = alpha

    return [pku, sku]



def readReceiverVKey(root):
    [params, g, Kx, Kz, Kl] = readPP(root)
    pairing = Pairing(params)

    vKeyCopy = loadFile(root + ParameterPathFromTools + "vKey.dat")
    [pkvCopy, skvCopy] = vKeyCopy
    skvCopy["Kx"] = Kx
    skvCopy["Kz"] = Kz
    skvCopy["Kl"] = Kl

    pkv = Element(pairing, G1, str(pkvCopy))

    beta_int = int(skvCopy["sk"], 16)
    beta = Element(pairing, Zr, str(beta_int))

    skv = skvCopy
    skv["sk"] = beta

    return [pkv, skv]




def add_to_16(text):
    """[Append text to be enough for times of 16]

    Args:
        text ([str]): [Original text]

    Returns:
        [str]: [Qualified text]
    """
    #text=text.encode()
    if len(text) % 16:
        add = 16 - (len(text) % 16)
    else:
        add = 0
    text = text + ('\0' * add)
    return text

def encrypt(text, key, iv):
    """Encrypt data by AES-CBC (16 字节分组)，在 bytes 层面补 '\0'。"""

    # key / iv：保持原来的用法
    key = key.encode("utf-8")
    iv  = iv.encode("utf-8")
    mode = AES.MODE_CBC

    # 统一转成 bytes
    if isinstance(text, str):
        text_bytes = text.encode("utf-8")
    else:
        # 万一外面已经传进来的是 bytes，就直接用
        text_bytes = text

    # 在 bytes 层面做 16 字节对齐的补 '\0'
    block_size = 16
    pad_len = block_size - (len(text_bytes) % block_size)
    if pad_len == block_size:
        pad_len = 0
    text_bytes = text_bytes + b'\0' * pad_len

    cryptos = AES.new(key, mode, iv)
    cipher_text = cryptos.encrypt(text_bytes)

    # 为了便于保存，仍然转成 16 进制字符串
    return b2a_hex(cipher_text)

# def encrypt(text, key, iv):
#     """[Encrypt data by AES]
#
#     Args:
#         text ([str]): [Plain data]
#         key ([str]): [AES key must be times of 16]
#         iv  ([str]): [iv must be times of 16]
#
#     Returns:
#         [byte]: [Encrypted data]
#     """
#     # key / iv 仍然用 utf-8 转成 bytes
#     key = key.encode("utf-8")
#     mode = AES.MODE_CBC
#     iv  = iv.encode("utf-8")
#
#     # 先在字符串层面补齐到 16 的倍数
#     text = add_to_16(text)
#
#     # 再把补齐后的明文转成 bytes
#     text_bytes = text.encode("utf-8")
#
#     cryptos = AES.new(key, mode, iv)
#     cipher_text = cryptos.encrypt(text_bytes)
#
#     # 转 16 进制字符串方便保存
#     return b2a_hex(cipher_text)

def decrypt(text,key,iv):
    """[Decrypt encrypted data by AES]

    Args:
        text ([str]): [Encrypted data]
        key ([str]): [AES key must be times of 16]
        iv ([str]): [iv must be times of 16]

    Returns:
        [str]: [Decrypted data]
    """
    key = key.encode('utf-8')
    mode = AES.MODE_CBC
    #iv = b'qqqqqqqqqqqqqqqq'
    iv=iv.encode('utf-8')
    cryptos = AES.new(key, mode, iv)
    plain_text = cryptos.decrypt(a2b_hex(text))
    return bytes.decode(plain_text).rstrip('\0')# 解密后，去掉补足的空格用strip() 去掉

def generate_random_str(randomlength=16):
    """[生成一个指定长度的随机字符串]

    Args:
        randomlength (int, optional): [description]. Defaults to 16.

    Returns:
        [str]: [String in random]
    """
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(randomlength):
        random_str += base_str[random.randint(0, length)]
    return random_str

def PRF_F(key,msg):
    """[PRF_F]

    Args:
        key ([type]): [description]
        msg ([type]): [description]

    Returns:
        [type]: [Random number]
    """
    random.seed(key+msg)
    final=random.random()*1000000000000000000
    return final

def PRF_Fp(params, key, msg):
    pairing = Pairing(params)
    # 原来：Hash2((key+msg)).hexdigest()
    hash_value = Element.from_hash(
        pairing, Zr,
        Hash2((key + msg)).digest()
    )
    return hash_value


def readWSet(root):
    path=root+ParameterPathFromTools+"WSet.dat"
    WSetLocal={}
    if(os.path.exists(path)):
        fileWSet=open(path,"r")
        WSetLocal=eval(fileWSet.read())
    return WSetLocal

def readInds(root):
    path=root+ParameterPathFromTools+"Inds.dat"
    IndsLocal={}
    if(os.path.exists(path)):
        fileInds=open(path,"r")
        IndsLocal=eval(fileInds.read())
    return IndsLocal














