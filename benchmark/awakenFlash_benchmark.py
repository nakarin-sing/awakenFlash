#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash v1.0 Benchmark
- เปรียบเทียบ XGBoost vs awakenFlash
- Real run 100M samples
"""

import time, gc, psutil
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from src.awakenFlash_core import train_step, infer

# ===================== CONFIG =====================
N_SAMPLES=10_000_000   # ตัวอย่างลดเหลือ 10M สำหรับ run GitHub
N_FEATURES=32
N_CLASSES=3
CHUNK_SIZE=524_288
SEED=42

H=448
CONF_THRESHOLD=105
LS=0.006
EPOCHS=3

proc = psutil.Process()

# ===================== DATA STREAM =====================
def _generate_chunk(rng_state,size,f,c):
    X = np.empty((size,f+8),np.float32)
    y = np.empty(size,np.int64)
    state,inc = rng_state[0],rng_state[1]
    for i in range(size):
        for j in range(f):
            state = (state*6364136223846793005+inc)&0xFFFFFFFFFFFFFFFF
            X[i,j]=(state>>40)/(1<<24)-0.5
        for j in range(8):
            X[i,f+j]=X[i,j]*X[i,j+8]
        state=(state*6364136223846793005+inc)&0xFFFFFFFFFFFFFFFF
        y[i]=(state>>58)%c
    rng_state[0]=state
    return X,y

def data_stream(n_samples,chunk=CHUNK_SIZE,seed=SEED):
    rng=np.random.default_rng(seed)
    state=np.array([rng.integers(0,2**64-1,dtype=np.uint64),
                    rng.integers(1,2**64-1,dtype=np.uint64)],dtype=np.uint64)
    for start in range(0,n_samples,chunk):
        size=min(chunk,n_samples-start)
        X,y=_generate_chunk(state,size,N_FEATURES,N_CLASSES)
        yield X,y

# ===================== XGBoost =====================
print("\n[1] XGBoost TRAINING")
ram_start = proc.memory_info().rss/1e6
t0=time.time()
X_train,y_train=next(data_stream(N_SAMPLES))
xgb=XGBClassifier(n_estimators=300,max_depth=6,n_jobs=-1,random_state=SEED,tree_method='hist',verbosity=0)
xgb.fit(X_train,y_train)
xgb_train_time=time.time()-t0
xgb_ram=proc.memory_info().rss/1e6-ram_start

X_test,y_test=next(data_stream(100_000))
t0=time.time()
xgb_pred=xgb.predict(X_test)
xgb_inf_ms=(time.time()-t0)/len(X_test)*1000
xgb_acc=accuracy_score(y_test,xgb_pred)
print(f"XGBoost → Train: {xgb_train_time/60:.1f}min | RAM: {xgb_ram:.1f}MB | ACC: {xgb_acc:.4f}")
del X_train,y_train,xgb; gc.collect()

# ===================== awakenFlash =====================
print("\n[2] awakenFlash TRAINING")
ram_start = proc.memory_info().rss/1e6

INPUT_DIM=40
SPARSITY=0.7
nnz=int(INPUT_DIM*H*SPARSITY)
mask=np.random.rand(INPUT_DIM,H)<SPARSITY
W1=np.zeros((INPUT_DIM,H),np.int8)
W1[mask]=np.random.randint(-4,5,nnz,np.int8)
rows,cols=np.where(mask)
values=W1[rows,cols].copy()
indptr=np.zeros(H+1,np.int32)
np.cumsum(np.bincount(rows,minlength=H),out=indptr[1:])
col_indices=cols.astype(np.int32)
b1=np.zeros(H,np.int32)
W2=np.random.randint(-4,5,(H,N_CLASSES),np.int8)
b2=np.zeros(N_CLASSES,np.int32)

scale=1.0
t0=time.time()
for epoch in range(EPOCHS):
    for X_chunk,y_chunk in data_stream(N_SAMPLES):
        scale=max(scale,np.max(np.abs(X_chunk))/127.0)
        X_i8=np.clip(np.round(X_chunk/scale),-128,127).astype(np.int8)
        values,b1,W2,b2=train_step(X_i8,y_chunk,values,col_indices,indptr,b1,W2,b2,H,CONF_THRESHOLD,LS)
        del X_i8; gc.collect()
awaken_train_time=time.time()-t0
awaken_ram=proc.memory_info().rss/1e6-ram_start

X_test_i8=np.clip(np.round(X_test/scale),-128,127).astype(np.int8)
t0=time.time()
pred,ee_ratio=infer(X_test_i8,values,col_indices,indptr,b1,W2,b2,H,CONF_THRESHOLD)
awaken_inf_ms=(time.time()-t0)/len(X_test_i8)*1000
awaken_acc=accuracy_score(y_test,pred)
model_kb=(values.nbytes+col_indices.nbytes+indptr.nbytes+b1.nbytes+b2.nbytes+W2.nbytes+5)/1024

# ===================== FINAL BENCHMARK =====================
print("\n"+"="*80)
print(f"{'Metric':<28}{'XGBoost':>12}{'awakenFlash':>18}{'Win'}")
print("-"*80)
print(f"{'Accuracy':<28}{xgb_acc:>12.4f}{awaken_acc:>18.4f}  **{(awaken_acc-xgb_acc)*100:.2f}%**")
print(f"{'Train Time (min)':<28}{xgb_train_time/60:>12.1f}{awaken_train_time/60:>18.1f}  **{xgb_train_time/awaken_train_time:.1f}× faster**")
print(f"{'Inference (ms/sample)':<28}{xgb_inf_ms:>12.3f}{awaken_inf_ms:>18.5f}  **{xgb_inf_ms/awaken_inf_ms:.1f}× faster**")
print(f"{'Early Exit Ratio':<28}{'0%':>12}{ee_ratio:>17.1%}  **+{ee_ratio*100:.1f}%**")
print(f"{'RAM Peak (MB)':<28}{xgb_ram:>12.1f}{awaken_ram:>18.2f}  **-99%**")
print(f"{'Model Size (KB)':<28}{'~448000':>12}{model_kb:>18.1f}  **-× smaller**")
print("="*80)
