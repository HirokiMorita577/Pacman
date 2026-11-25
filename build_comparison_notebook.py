"""Freeze the complete comparison runner, tests and a self-contained Colab notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import make_round3_upload as legacy

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "colab_comparison_study_20261020"

DEPLOY = r'''#@title 1. 比較コードを展開・全テスト・Driveへ保存
import base64, hashlib, importlib.util, io, json, os, platform, shutil, subprocess, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
EXPECTED_BUNDLE = "__HASH__"
WORK = Path("/content/pacman_comparison_code")
LOCAL = Path("/content/pacman_comparison_study")
FOLDER = Path("/content/drive/MyDrive/DQN_Pacman_comparisons_20261020___SHORT__")
archive = base64.b64decode("__PAYLOAD__")
assert hashlib.sha256(archive).hexdigest() == EXPECTED_BUNDLE
with zipfile.ZipFile(io.BytesIO(archive)) as z:
    frozen = json.loads(z.read("source_hashes.json"))
    assert set(z.namelist()) == set(frozen) | {"source_hashes.json"}
    for name, expected in frozen.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        data = z.read(name)
        assert hashlib.sha256(data).hexdigest() == expected, name
        target = WORK / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
print("SOURCE_VERIFIED:", len(frozen), "files; bundle", EXPECTED_BUNDLE, flush=True)
missing = [p for p in ("numpy", "gymnasium", "torch", "pytest", "matplotlib", "PIL")
           if importlib.util.find_spec(p) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    *["Pillow" if p == "PIL" else p for p in missing]], check=True)
test = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                      cwd=WORK, capture_output=True, text=True)
test_log = test.stdout + "\n" + test.stderr
(WORK / "deployment_tests.log").write_text(test_log, encoding="utf-8")
print(test_log, flush=True)
test.check_returncode()
subprocess.run([sys.executable, "-u", "comparison_study.py", "all", "--out", str(LOCAL), "--plan-only"],
               cwd=WORK, check=True)
from google.colab import drive
drive.mount("/content/drive")
manifest_path = FOLDER / "source_hashes.json"
if FOLDER.exists() and any(FOLDER.iterdir()) and not manifest_path.exists():
    raise RuntimeError("Destination exists without a manifest; use a new folder")
if manifest_path.exists() and json.loads(manifest_path.read_text()) != frozen:
    raise RuntimeError("Existing source differs; use a new deployment folder")
FOLDER.mkdir(parents=True, exist_ok=True)
for name, expected in frozen.items():
    target = FOLDER / "source" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WORK / name, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, name
manifest_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
meta = dict(bundle_sha256=EXPECTED_BUNDLE, tests_passed=True, source_files=len(frozen),
            deployed_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
            packages={p:version(p) for p in ("torch", "numpy", "gymnasium", "pytest", "matplotlib")},
            local=str(LOCAL), drive=str(FOLDER))
(FOLDER / "deployment.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
shutil.copy2(WORK / "deployment_tests.log", FOLDER / "deployment_tests.log")
shutil.copy2(LOCAL / "all_plan.json", FOLDER / "all_plan.json")
print("DEPLOY_READY:", FOLDER, flush=True)
print("66比較項目・250条件×5seed。全画像の横軸は学習回数。次のセルで全比較を開始・再開します。", flush=True)
'''

TRAIN = r'''#@title 2. 全比較を開始・再開（完了まで実行中）
START_TRAINING = True #@param {type:"boolean"}
WORKERS = 2 #@param {type:"integer"}
REPORT_EVERY = 1 #@param {type:"integer"}
if START_TRAINING:
    import hashlib, json, os, subprocess, sys
    from pathlib import Path
    WORK = Path("/content/pacman_comparison_code")
    LOCAL = Path("/content/pacman_comparison_study")
    FOLDER = Path("/content/drive/MyDrive/DQN_Pacman_comparisons_20261020___SHORT__")
    meta = json.loads((FOLDER / "deployment.json").read_text())
    assert meta["tests_passed"] and meta["bundle_sha256"] == "__HASH__", "Run cell 1 first"
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
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(set(legacy.SOURCES + [
        "ghost4_round4.py", "ghost4_round5.py", "ghost4_route_v2.py",
        "comparison_study.py", "COMPARISON_STUDY.md", "requirements.txt",
        "requirements-comparison.txt", "ROUTE_V2.md",
    ] + [p.relative_to(ROOT).as_posix() for folder, pattern in
         (("tests", "test_*.py"), ("experiments", "*.py")) for p in (ROOT / folder).glob(pattern)]))
    blobs = {name:(ROOT / name).read_bytes() for name in names}
    hashes = {name:hashlib.sha256(data).hexdigest() for name, data in blobs.items()}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in {**blobs, "source_hashes.json":json.dumps(hashes, indent=1).encode()}.items():
            entry = zipfile.ZipInfo(name, (2026, 10, 20, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, data)
    data = stream.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    deploy = DEPLOY.replace("__HASH__", digest).replace("__SHORT__", digest[:8]).replace(
        "__PAYLOAD__", base64.b64encode(data).decode())
    train = TRAIN.replace("__HASH__", digest).replace("__SHORT__", digest[:8])
    compile(deploy, "<deploy>", "exec")
    compile(train, "<training>", "exec")
    notebook = dict(nbformat=4, nbformat_minor=0,
                    metadata={"colab":{"name":"Pacman_comparison_study_20261020.ipynb"},
                              "kernelspec":{"name":"python3", "display_name":"Python 3"}},
                    cells=[])
    notebook["cells"].append(dict(cell_type="markdown", metadata={}, source=[
        "# パックマン：報酬・DQN/SARSA・PSOの比較\n",
        "1番で全テストとDriveへのコード保存、2番で全比較を開始・再開。\n",
        "横軸は学習回数（エピソード数）。250条件×5seedとPSO等の探索・独立再学習。\n",
        "計画はall_plan.json、画像はresults/factors/figures/とresults/search/以下。\n",
        "Colab切断後は再接続して1番→2番。同一版の保存済み実験を復元します。\n"]))
    for i, code in enumerate((deploy, train), 1):
        notebook["cells"].append(dict(cell_type="code", metadata={"id":f"comparison-{i}"},
                                     source=code.splitlines(keepends=True), execution_count=None, outputs=[]))
    (OUT / "Pacman_comparison_study_20261020.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "deployment_cell.py").write_text(deploy, encoding="utf-8")
    (OUT / "training_cell.py").write_text(train, encoding="utf-8")
    (OUT / "source_bundle.zip").write_bytes(data)
    metadata = dict(bundle_sha256=digest, source_hashes=hashes,
                    drive_folder=f"DQN_Pacman_comparisons_20261020_{digest[:8]}")
    (OUT / "bundle_manifest.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    print(json.dumps(dict(bundle_sha256=digest, files=len(hashes), zip_bytes=len(data),
                          drive_folder=metadata["drive_folder"], notebook=str(OUT)), indent=1))


if __name__ == "__main__":
    main()
