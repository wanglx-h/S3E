"""
protocols_stub.py
提供 Beaver 三元组与 Rabbit 安全比较协议的封装。
支持两方安全计算。
"""

import numpy as np
import platform

# ====== CrypTen 初始化 ======
CRYPTEN_AVAILABLE = False
try:
    import crypten
    import torch
    if platform.system().lower().startswith("win"):
        crypten.init_thread(rank=0, world_size=1)
    else:
        crypten.init()
    CRYPTEN_AVAILABLE = True
    print("[Info] CrypTen 已初始化")
except Exception as e:
    print(f"[Warning] CrypTen 未加载或初始化失败: {e}")
    CRYPTEN_AVAILABLE = False

# ====== Rabbit 初始化 ======
try:
    from rabbit import comparisons
    RABBIT_AVAILABLE = True
    print("[Info] Rabbit 已加载")
except Exception as e:
    print(f"[Warning] Rabbit 未加载: {e}")
    RABBIT_AVAILABLE = False


# def beaver_multiply(x, y, *, allow_fallback=True):
#     """
#     CrypTen 安全点积（浮点）
#     单机时自动回退明文模拟
#     """
#     import torch
#
#     if not CRYPTEN_AVAILABLE:
#         if allow_fallback:
#             res = float(np.dot(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)))
#             print("[Info] secure_dot_two_party: 使用明文回退")
#             return res
#         raise RuntimeError("CrypTen 不可用")
#
#     try:
#         x_enc = crypten.cryptensor(torch.tensor(np.array(x, dtype=np.float32)))  # 确保torch.tensor
#         y_enc = crypten.cryptensor(torch.tensor(np.array(y, dtype=np.float32)))  # 确保torch.tensor
#         z_enc = x_enc * y_enc
#         dot = z_enc.sum().get_plain_text().item()
#         print("Encrypted dot product:", dot)
#         return float(dot)
#     except Exception as e:
#         if allow_fallback:
#             print(f"[Error] CrypTen dot 异常 {e} -> 使用明文回退")
#             return float(np.dot(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)))
#         raise
P_DEFAULT = 2305843009213694000

def beaver_multiply(x, y, P: int = P_DEFAULT):
    """
    本地模拟的 Beaver 乘法：
    输入是两边的“share 形式的向量” x, y（这里我们一台机子全拿到了），
    返回 (x * y) mod P 的“结果 share”（其实就是结果本身，因为我们在一台机子上）。
    真实两方协议时，这段逻辑会拆成双方各自算 e,f，然后交换 e,f，再拼。
    """
    x = np.array(x, dtype=object) % P
    y = np.array(y, dtype=object) % P

    # 生成三元组 (a, b, c=a*b mod P)，也要用大整数
    rng = np.random.default_rng()
    a = rng.integers(0, P, size=x.shape, dtype=np.int64).astype(object)
    b = rng.integers(0, P, size=x.shape, dtype=np.int64).astype(object)
    c = (a * b) % P

    # e, f 也都在大整数里做
    e = (x - a) % P
    f = (y - b) % P

    # Beaver 合成：z = c + e*b + f*a + e*f (mod P)
    z = (c + (e * b) % P + (f * a) % P + (e * f) % P) % P
    return z




def rabbit_argmax(sim_scores, *, allow_fallback=True):
    max_idx = 0
    max_val = sim_scores[0]
    for i in range(1, len(sim_scores)):
        if comparisons.lt_const(max_val, sim_scores[i]):
            max_val = sim_scores[i]
            max_idx = i
    return int(max_idx), max_val


def rabbit_argmax_two_shares(scores_a, scores_b, P, *, allow_fallback=True):
    """
    将两份整数 share 合并（mod P），再做 Rabbit 安全比较。
    """
    merged = [(int(a) + int(b)) % P for a, b in zip(scores_a, scores_b)]
    print("a+b:",merged)
    return rabbit_argmax(merged, allow_fallback=allow_fallback)


if __name__ == "__main__":
    print("=== 测试 CrypTen Dot ===")
    a = [0.1, 0.2, 0.3]
    b = [0.4, 0.5, 0.6]
    print("plain dot =", np.dot(a, b))
    print("secure dot =", beaver_multiply(a, b))

    print("=== 测试 Rabbit ===")
    sims = []
    sA = [257294993843293550, 2040441790251270247, 324641903511253915, 811592266094112377, 1569980896547063804, 927779220910175210, 1945688522497076647, 2294494220282819297, 1625885880307563195, 706113996135195459, 1109468463460227035, 1120147355444161199, 17710471822153299, 2056480047970519836, 12234541598176818, 99422549967992019, 2073852205406014897, 883412708149307972, 410681918406391941, 987412758365441358, 1989617892368201457, 2126527674605190726, 447889604671044206, 1770329814732435702, 1815872433051099464, 896021564866453302, 2159302569023749492, 984767596507981411, 89301434927875929, 767704291381642036, 1037239871880528844, 1623967686230810763, 605764060828350976, 207328939153660343, 2218057091067478495, 287863396768109507, 884603141187219250, 24232313400470710, 921921426626662619, 1564837255566312225, 693933593660925637, 388135445370040740, 1216227156532120341, 2223477193356403557, 814103032754220366, 2257076654126019447, 209231638382656133, 1488310823232976493, 1474708765621365718, 1991210394269078895]
    sB = [2048548015370400401, 265401218962423704, 1981201105702440036, 1494250743119581574, 735862112666630147, 1378063788303518741, 360154486716617304, 11348788930874654, 679957128906130756, 1599729013078498492, 1196374545753466916, 1185695653769532752, 2288132537391540652, 249362961243174115, 2293608467615517133, 2206420459245701932, 231990803807679054, 1422430301064385979, 1895161090807302010, 1318430250848252593, 316225116845492494, 179315334608503225, 1857953404542649745, 535513194481258249, 489970576162594487, 1409821444347240649, 146540440189944459, 1321075412705712540, 2216541574285818022, 1538138717832051915, 1268603137333165107, 681875322982883188, 1700078948385342975, 2098514070060033608, 87785918146215456, 2017979612445584444, 1421239868026474701, 2281610695813223241, 1383921582587031332, 741005753647381726, 1611909415552768314, 1917707563843653211, 1089615852681573610, 82365815857290394, 1491739976459473585, 48766355087674504, 2096611370831037818, 817532185980717458, 831134243592328233, 314632614944615056]

    idx, max_val = rabbit_argmax_two_shares(sA,sB,P=200)
    print("max idx =", idx, "val =", max_val)
