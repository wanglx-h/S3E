import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random
import math


# ============================================================
# AVL TREE NODE
# ============================================================

class AVLTreeNode:
    def __init__(self, key: int, vector_enc_1: np.ndarray, vector_enc_2: np.ndarray,
                 left=None, right=None):
        """
        key: category id
        vector_enc_1: encrypted node vector (M1^T * v1)
        vector_enc_2: encrypted node vector (M2^T * v2)
        """
        self.key = key
        self.left = left
        self.right = right
        self.height = 1

        self.vector_enc_1 = vector_enc_1
        self.vector_enc_2 = vector_enc_2


class AVLTree:
    def __init__(self):
        self.root = None

    # ---------------------------------------
    # height
    # ---------------------------------------
    @staticmethod
    def get_height(node: Optional[AVLTreeNode]):
        if node is None:
            return 0
        return node.height

    # ---------------------------------------
    # balance factor
    # ---------------------------------------
    def get_balance(self, node: Optional[AVLTreeNode]):
        if node is None:
            return 0
        return AVLTree.get_height(node.left) - AVLTree.get_height(node.right)

    # ---------------------------------------
    # rotations
    # ---------------------------------------
    def right_rotate(self, y: AVLTreeNode):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2

        y.height = 1 + max(AVLTree.get_height(y.left), AVLTree.get_height(y.right))
        x.height = 1 + max(AVLTree.get_height(x.left), AVLTree.get_height(x.right))
        return x

    def left_rotate(self, x: AVLTreeNode):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2

        x.height = 1 + max(AVLTree.get_height(x.left), AVLTree.get_height(x.right))
        y.height = 1 + max(AVLTree.get_height(y.left), AVLTree.get_height(y.right))
        return y

    # ---------------------------------------
    # insert node
    # ---------------------------------------
    def insert(self, key: int, vec1: np.ndarray, vec2: np.ndarray):
        self.root = self._insert(self.root, key, vec1, vec2)

    def _insert(self, node: Optional[AVLTreeNode], key: int,
                vec1: np.ndarray, vec2: np.ndarray):

        if node is None:
            return AVLTreeNode(key, vec1, vec2)

        if key < node.key:
            node.left = self._insert(node.left, key, vec1, vec2)
        elif key > node.key:
            node.right = self._insert(node.right, key, vec1, vec2)
        else:
            return node

        node.height = 1 + max(
            AVLTree.get_height(node.left),
            AVLTree.get_height(node.right)
        )

        balance = self.get_balance(node)

        # LL
        if balance > 1 and key < node.left.key:
            return self.right_rotate(node)

        # RR
        if balance < -1 and key > node.right.key:
            return self.left_rotate(node)

        # LR
        if balance > 1 and key > node.left.key:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # RL
        if balance < -1 and key < node.right.key:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    # ---------------------------------------
    # find best category by secure inner product
    # ---------------------------------------
    def secure_search(self, VTD_1: np.ndarray, VTD_2: np.ndarray):
        """
        根据 VTD（加密陷门）在加密 AVL 树中执行 secure kNN 路径选择。
        """
        node = self.root
        best_key = None
        best_score = -1e9

        while node is not None:
            s_left = -1e9
            s_right = -1e9

            if node.left is not None:
                s_left = float(node.left.vector_enc_1.dot(VTD_1) +
                               node.left.vector_enc_2.dot(VTD_2))

            if node.right is not None:
                s_right = float(node.right.vector_enc_1.dot(VTD_1) +
                                node.right.vector_enc_2.dot(VTD_2))

            # choose better branch
            if s_left >= s_right:
                best_score = s_left
                node = node.left
            else:
                best_score = s_right
                node = node.right

            # if reach leaf
            if node is None:
                break
            best_key = node.key

        return best_key
# ============================================================
# Secure Index Builder  —— 支持 m=2000 维向量、加密 inner-product 搜索
# ============================================================

class SecureIndexBuilder:
    def __init__(self, config):
        """
        config 需要包含：
            config.secure_dim = 2000
            config.epsilon = 10
        """
        self.config = config
        self.m = config.secure_dim
        self.epsilon = config.epsilon

        # 加密参数
        self.M1 = None
        self.M2 = None
        self.M1_T = None
        self.M2_T = None
        self.M1_inv_T = None
        self.M2_inv_T = None

    # ------------------------------------------------------------
    # 生成加密矩阵 + S 拆分向量
    # ------------------------------------------------------------
    def generate_security_parameters(self):
        dim = self.m + self.epsilon

        # 随机正交矩阵（近似）
        A = np.random.randn(dim, dim)
        M1, _ = np.linalg.qr(A)

        B = np.random.randn(dim, dim)
        M2, _ = np.linalg.qr(B)

        self.M1 = M1
        self.M2 = M2

        self.M1_T = M1.T
        self.M2_T = M2.T

        self.M1_inv_T = np.linalg.inv(self.M1).T
        self.M2_inv_T = np.linalg.inv(self.M2).T

    # ------------------------------------------------------------
    # 将向量 v ∈ R^m 扩展到 R^{m+ε} 并按 S 拆分为 v1、v2
    # ------------------------------------------------------------
    def extend_and_split(self, vec_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        输入：
            vec_m: shape = (m,)

        输出：
            v1, v2: shape = (m+ε,)
        """
        dim = self.m + self.epsilon
        v_ext = np.zeros(dim, dtype=float)
        v_ext[:self.m] = vec_m

        # 随机拆分向量
        r = np.random.rand(dim)
        v1 = v_ext * r
        v2 = v_ext - v1
        return v1, v2

    # ------------------------------------------------------------
    # 加密向量： (M1^T * v1,  M2^T * v2)
    # ------------------------------------------------------------
    def encrypt_vector(self, vec_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对类别节点向量进行加密存储。
        """
        v1, v2 = self.extend_and_split(vec_m)
        enc1 = self.M1_T.dot(v1)
        enc2 = self.M2_T.dot(v2)
        return enc1, enc2

    # ------------------------------------------------------------
    # 生成查询陷门（Query Trapdoor）
    # ------------------------------------------------------------
    def generate_query_trapdoor(self, vec_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v1, v2 = self.extend_and_split(vec_m)

        # ✅ 直接使用预先计算好的逆矩阵转置
        if self.M1_inv_T is None or self.M2_inv_T is None:
            raise RuntimeError("Security parameters not initialized. Call generate_security_parameters() first.")

        VTD_1 = self.M1_inv_T.dot(v1)
        VTD_2 = self.M2_inv_T.dot(v2)
        return VTD_1, VTD_2

    # ------------------------------------------------------------
    # 构建两层索引（AVL + 倒排索引）
    # document_clusters: cat_id → [doc objects]
    # cluster_keyword_distribution: cat_id → {word: prob}
    # vocabulary: word → index(0~1999)
    # ------------------------------------------------------------
    def build_two_layer_index(
        self,
        document_clusters: Dict[int, List[Any]],
        cluster_keyword_distribution: Dict[int, Dict[str, float]],
        vocabulary: Dict[str, int],
    ):
        """
        返回：
            avl_tree: AVLTree（加密向量存储）
            inverted_index: cat_id → [doc_id]
        """

        # step 1：生成加密参数 M1, M2
        self.generate_security_parameters()

        # step 2：构建加密 AVL 节点
        avl = AVLTree()

        for cat_id, kw_prob in cluster_keyword_distribution.items():
            # 构造 m=2000 维向量
            vec_m = np.zeros(self.m, dtype=float)
            for w, p in kw_prob.items():
                if w in vocabulary:
                    idx = vocabulary[w]
                    if idx < self.m:
                        vec_m[idx] = p

            # 加密节点向量
            enc1, enc2 = self.encrypt_vector(vec_m)

            # 插入 AVL
            avl.insert(cat_id, enc1, enc2)

        # step 3：倒排索引
        inverted_index = {}
        for cat_id, docs in document_clusters.items():
            inverted_index[cat_id] = [d.doc_id for d in docs]

        return avl, inverted_index

    # ------------------------------------------------------------
    # 执行安全搜索
    # top-k
    # ------------------------------------------------------------
    def secure_search(self, avl: AVLTree, inverted_index: Dict[int, List[int]],
                      VTD_1: np.ndarray, VTD_2: np.ndarray, top_k: int = 40):
        # Step 1：根据 VTD 在加密 AVL 上选择最佳类别
        best_cat = avl.secure_search(VTD_1, VTD_2)
        if best_cat is None:
            return []

        # Step 2：根据倒排索引取 top-k 个文档
        docs = inverted_index.get(best_cat, [])
        if not docs:
            return []

        return docs[:top_k]

