# utils.py - 通用函数模块
import time
import json
import numpy as np

def now_ms():
    return int(time.time() * 1000)

class Timer:
    def __init__(self):
        self.start_time = None
    def start(self):
        self.start_time = time.time()
    def stop(self):
        return time.time() - self.start_time

def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return np.dot(a, b)
