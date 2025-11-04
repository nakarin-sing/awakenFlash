def main_adaptive():
    print("="*80)
    print("🔬 ADAPTIVE NON-LINEAR MACHINE LEARNING BENCHMARK")
    print("="*80)
    print("Real-time adaptive streaming + batch learning + ensemble + memory-aware insights\n")
    
    # Initialize memory tracker
    memory_tracker = MemoryTracker()
    
    # Load data
    chunks, all_classes = load_data(n_chunks=10, chunk_size=10000)
    
    # ===== Adaptive Streaming Scenario =====
    WINDOW_SIZE = 5
    first_sgd = first_pa = True
    xgb_all_X, xgb_all_y = [], []
    results = []
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        # SGD
        memory_tracker.snapshot(f'sgd_chunk{chunk_id}_start')
        start = time.time()
        if first_sgd:
            sgd = SGDClassifier(loss="log_loss", max_iter=10, warm_start=True, random_state=42, n_jobs=-1)
            sgd.partial_fit(X_train, y_train, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train, y_train)
        sgd_pred = sgd.predict(X_test)
        sgd_metrics = compute_metrics(y_test, sgd_pred)
        sgd_time = time.time() - start
        memory_tracker.snapshot(f'sgd_chunk{chunk_id}_end')
        
        # Passive-Aggressive
        memory_tracker.snapshot(f'pa_chunk{chunk_id}_start')
        start = time.time()
        if first_pa:
            pa = PassiveAggressiveClassifier(C=0.01, max_iter=10, warm_start=True, random_state=42, n_jobs=-1)
            pa.partial_fit(X_train, y_train, classes=all_classes)
            first_pa = False
        else:
            pa.partial_fit(X_train, y_train)
        pa_pred = pa.predict(X_test)
        pa_metrics = compute_metrics(y_test, pa_pred)
        pa_time = time.time() - start
        memory_tracker.snapshot(f'pa_chunk{chunk_id}_end')
        
        # XGBoost sliding window
        memory_tracker.snapshot(f'xgb_chunk{chunk_id}_start')
        start = time.time()
        xgb_all_X.append(X_train); xgb_all_y.append(y_train)
        if len(xgb_all_X) > WINDOW_SIZE:
            xgb_all_X = xgb_all_X[-WINDOW_SIZE:]; xgb_all_y = xgb_all_y[-WINDOW_SIZE:]
        X_xgb_train = np.vstack(xgb_all_X)
        y_xgb_train = np.concatenate(xgb_all_y)
        dtrain = xgb.DMatrix(X_xgb_train, label=y_xgb_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": len(all_classes),
             "max_depth": 5, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
             "verbosity": 0, "nthread": -1},
            dtrain, num_boost_round=20
        )
        xgb_pred = xgb_model.predict(dtest)
        xgb_metrics = compute_metrics(y_test, xgb_pred)
        xgb_time = time.time() - start
        memory_tracker.snapshot(f'xgb_chunk{chunk_id}_end')
        
        # Ensemble (majority vote)
        ensemble_pred = np.array([sgd_pred, pa_pred, xgb_pred]).T
        ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=ensemble_pred)
        ensemble_metrics = compute_metrics(y_test, ensemble_pred)
        
        # Geometric F1 (non-linear metric)
        geom_f1 = np.cbrt(sgd_metrics['f1'] * pa_metrics['f1'] * xgb_metrics['f1'])
        
        results.append({
            'chunk': chunk_id,
            'sgd_acc': sgd_metrics['accuracy'], 'sgd_f1': sgd_metrics['f1'], 'sgd_time': sgd_time,
            'pa_acc': pa_metrics['accuracy'], 'pa_f1': pa_metrics['f1'], 'pa_time': pa_time,
            'xgb_acc': xgb_metrics['accuracy'], 'xgb_f1': xgb_metrics['f1'], 'xgb_time': xgb_time,
            'ensemble_acc': ensemble_metrics['accuracy'], 'ensemble_f1': ensemble_metrics['f1'],
            'geom_f1': geom_f1
        })
    
    streaming_results = pd.DataFrame(results)
    
    # ===== Batch Scenario =====
    batch_results = scenario_2_batch(chunks, all_classes, memory_tracker)
    
    # ===== Visualizations =====
    plot_results_adaptive(streaming_results)
    
    # ===== Memory Analysis =====
    memory_df = memory_tracker.get_usage()
    
    # ===== Generate Report =====
    generate_report_adaptive(streaming_results, batch_results, memory_df)
    
    # ===== Save Raw Results =====
    os.makedirs('benchmark_results', exist_ok=True)
    streaming_results.to_csv('benchmark_results/streaming_results_adaptive.csv', index=False)
    batch_results.to_csv('benchmark_results/batch_results_adaptive.csv')
    memory_df.to_csv('benchmark_results/memory_usage.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ ADAPTIVE BENCHMARK COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 benchmark_results/adaptive_benchmark.png")
    print("  📄 benchmark_results/BENCHMARK_REPORT_ADAPTIVE.md")
    print("  📈 benchmark_results/streaming_results_adaptive.csv")
    print("  📈 benchmark_results/batch_results_adaptive.csv")
    print("  💾 benchmark_results/memory_usage.csv")
    print("\n🎯 Check BENCHMARK_REPORT_ADAPTIVE.md for full adaptive analysis!")
