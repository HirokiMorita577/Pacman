#@title 2. 全比較を開始・再開（完了まで実行中）
START_TRAINING = True #@param {type:"boolean"}
WORKERS = 2 #@param {type:"integer"}
REPORT_EVERY = 1 #@param {type:"integer"}
if START_TRAINING:
    import hashlib, json, os, subprocess, sys
    from pathlib import Path
    WORK = Path("/content/pacman_comparison_code")
    LOCAL = Path("/content/pacman_comparison_study")
    FOLDER = Path("/content/drive/MyDrive/DQN_Pacman_comparisons_20261020_3c20ffba")
    meta = json.loads((FOLDER / "deployment.json").read_text())
    assert meta["tests_passed"] and meta["bundle_sha256"] == "3c20ffba064724062181dd97882e20b419168eb75c662d562ca4068836297efa", "Run cell 1 first"
    frozen = json.loads((FOLDER / "source_hashes.json").read_text())
    for name, expected in frozen.items():
        assert hashlib.sha256((WORK / name).read_bytes()).hexdigest() == expected, name
        assert hashlib.sha256((FOLDER / "source" / name).read_bytes()).hexdigest() == expected, name
    available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1024**3
    if not 1 <= WORKERS <= 8 or available < WORKERS * 2 + 2:
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
        raise RuntimeError(f"exit={code}: inspect /content/pacman_comparison_code/full_study.log")
    print("ALL_COMPARISONS_FINISHED: 各COMPLETE.jsonのfailed_trialsも確認してください。", flush=True)
else:
    print("未開始。START_TRAININGにチェックして実行すると開始・再開します。")
