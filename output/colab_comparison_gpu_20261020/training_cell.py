#@title 2. 全比較を開始・再開（完了まで実行中）
START_TRAINING = True #@param {type:"boolean"}
WORKERS = 8 #@param {type:"integer"}
REPORT_EVERY = 1 #@param {type:"integer"}
if START_TRAINING:
    import hashlib, json, os, subprocess, sys
    from pathlib import Path
    WORK = Path("/content/pacman_comparison_gpu_code")
    LOCAL = Path("/content/pacman_comparison_gpu_study")
    FOLDER = Path("/content/drive/MyDrive/DQN_Pacman_comparisons_gpu_20261020_9e75afec")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PACMAN_DEVICE"] = "cuda"
    os.environ["PACMAN_TORCH_THREADS"] = "1"
    import torch
    assert torch.cuda.is_available(), "GPU runtime required; CPU fallback is disabled"
    WORKERS = min(WORKERS, max(1, (os.cpu_count() or 2)-2))
    print("CUDA_TRAINING:", torch.cuda.get_device_name(0), "workers", WORKERS, flush=True)
    meta = json.loads((FOLDER / "deployment.json").read_text())
    assert meta["tests_passed"] and meta["bundle_sha256"] == "9e75afec6a09a84cf5199b2a1bc1d612747d9ad7369f7a50907222189759b0fd", "Run cell 1 first"
    frozen = json.loads((FOLDER / "source_hashes.json").read_text())
    for name, expected in frozen.items():
        assert hashlib.sha256((WORK / name).read_bytes()).hexdigest() == expected, name
        assert hashlib.sha256((FOLDER / "source" / name).read_bytes()).hexdigest() == expected, name
    available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1024**3
    if not 1 <= WORKERS <= 8 or available < WORKERS * 3 + 4:
        raise RuntimeError(f"workers={WORKERS}, available={available:.1f}GiB: reduce parallelism")
    assert REPORT_EVERY >= 1
    command = [sys.executable, "-u", "comparison_study.py", "all", "--out", str(LOCAL),
               "--backup", str(FOLDER / "results"), "--workers", str(WORKERS),
               "--report-every", str(REPORT_EVERY)]
    print("全因子比較 → 報酬/γ/PSO/探索法比較。切断後は同じセルで再開。", flush=True)
    print("保存先:", FOLDER / "results", flush=True)
    with (WORK / "full_study.log").open("a", buffering=1) as log:
        process = subprocess.Popen(command, cwd=WORK, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        print("COMPARISON_PID:", process.pid, flush=True)
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"exit={code}: inspect /content/pacman_comparison_gpu_code/full_study.log")
    print("ALL_COMPARISONS_FINISHED: 各COMPLETE.jsonのfailed_trialsも確認してください。", flush=True)
else:
    print("未開始。START_TRAININGにチェックして実行すると開始・再開します。")
