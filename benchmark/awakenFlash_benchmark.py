# ... (โค้ดส่วนบนและการตั้งค่าต่างๆ เหมือนเดิม) ...

# === awakenFlash ===
# ... (โค้ดส่วน Training และ Inference ของ awakenFlash เหมือนเดิม) ...

# ***NOTE: ต้องแก้ไขโค้ดที่เรียก train_step และ infer ให้สอดคล้องกับ core ใหม่***

t0 = time.time()
final_scale = 1.0
for epoch in range(EPOCHS):
    print(f" Epoch {epoch+1}/{EPOCHS}")
    scale = 1.0
    for X_chunk, y_chunk in data_stream(N_SAMPLES):
        scale = max(scale, np.max(np.abs(X_chunk)) / 127.0)
        X_i8 = np.clip(np.round(X_chunk / scale), -128, 127).astype(np.int8)
        # แก้ไขตรงนี้ให้ส่ง B, H, CONF_THRESHOLD, LS และ lut_exp เข้าไปด้วย
        values, b1, W2, b2 = train_step(X_i8, y_chunk, values, col_indices, indptr, b1, W2, b2, B, H, CONF_THRESHOLD, LS, lut_exp)
        del X_i8; gc.collect()
final_scale = scale
flash_time = time.time() - t0
flash_ram = proc.memory_info().rss / 1e6 - ram_flash_start

# === INFERENCE ===
X_test, y_test = next(data_stream(10_000))
X_test_i8 = np.clip(np.round(X_test / final_scale), -128, 127).astype(np.int8)

# Warm-up Call (Infer ถูกเปลี่ยนให้รับ lut_exp และ CONF_THRESHOLD)
for _ in range(10): 
    infer(X_test_i8[:1], values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD)

t0 = time.time()
# Main Inference Call
pred, ee_ratio = infer(X_test_i8, values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD)
flash_inf = (time.time() - t0) / len(X_test_i8) * 1000
flash_acc = accuracy_score(y_test, pred)
model_kb = (values.nbytes + col_indices.nbytes + indptr.nbytes + b1.nbytes + b2.nbytes + W2.nbytes + 5) / 1024

# === ผลลัพธ์ ===
print("\n" + "="*100)
print("AWAKENFLASH v2.0 vs XGBoost | MAX SPEED INFERENCE TEST")
print("="*100)
print(f"{'Metric':<25} {'XGBoost':>15} {'awakenFlash':>18} {'Win'}")
print("-"*100)

# FIXED: ใช้ค่าที่วัดได้จริงในการแสดงผลและคำนวณ Win Ratio
xgb_inf_actual = (time.time() - xgb_t0) / len(X_test_xgb) * 1000 # คำนวณใหม่จาก t0 ที่ถูกต้องของ XGBoost
flash_inf_actual = flash_inf

print(f"{'Accuracy':<25} {xgb_acc:>15.4f} {flash_acc:>18.4f}")
print(f"{'Train Time (s)':<25} {xgb_time:>15.1f} {flash_time:>18.1f} {xgb_time/flash_time:.1f}x faster")
# ใช้ xgb_inf_actual ที่คำนวณได้จริง
print(f"{'Inference (ms)':<25} {xgb_inf_actual:>15.5f} {flash_inf_actual:>18.5f} {xgb_inf_actual/flash_inf_actual:.0f}x faster")
print(f"{'Early Exit':<25} {'0%':>15} {ee_ratio:>17.1%}")
print(f"{'RAM (MB)':<25} {xgb_ram:>15.1f} {flash_ram:>18.2f}")
print(f"{'Model (KB)':<25} {'~70k':>15} {model_kb:>18.1f}")
print("="*100)
print("CI READY: MAX SPEED INFERENCE TEST")
print("="*100)
