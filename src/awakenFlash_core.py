#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash core — train + infer functions (INT8)
"""

import numpy as np
from numba import njit, prange

# --------------------------
# Global Config (UPDATED FOR SPEED VALIDATION)
# --------------------------
N_SAMPLES = 100_000  # เพิ่มจำนวน Sample ให้สอดคล้องกับ CI 
N_FEATURES = 40      # <--- แก้ไขตรงนี้เป็น 40 
N_CLASSES = 3        # <--- แก้ไขตรงนี้เป็น 3 เพื่อให้เหมือน CI ก่อนหน้า 
B = 1024
H = 448              # เพิ่ม H เพื่อให้สอดคล้องกับ N_SAMPLES ใหม่
CONF_THRESHOLD = 80
LS = 0.006           # ใช้ LS ค่าเดิมที่เคยใช้

# ... (ฟังก์ชัน train_step และ infer เหมือนเดิมทุกประการ) ...

# --------------------------
# Optional Data Stream Generator
# --------------------------
# ใช้ data_stream แบบง่ายๆ สำหรับการทดสอบความถูกต้อง
def data_stream(n=1000):
    X = np.random.randn(n, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, N_CLASSES, size=n).astype(np.int32)
    yield X, y
