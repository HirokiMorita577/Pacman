"""Build a separate GPU release without overwriting the frozen CPU deployment."""
import json
from pathlib import Path

import build_comparison_notebook as builder

builder.OUT = builder.ROOT / "output" / "colab_comparison_gpu_20261020"
builder.legacy.SOURCES = [*builder.legacy.SOURCES, "COMPARISON_GPU.md"]
for name in ("DEPLOY", "TRAIN"):
    value = getattr(builder, name)
    value = value.replace("/content/pacman_comparison_code", "/content/pacman_comparison_gpu_code")
    value = value.replace("/content/pacman_comparison_study", "/content/pacman_comparison_gpu_study")
    value = value.replace("DQN_Pacman_comparisons_20261020_", "DQN_Pacman_comparisons_gpu_20261020_")
    setattr(builder, name, value)

builder.DEPLOY = builder.DEPLOY.replace(
    'test = subprocess.run(',
    '''os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PACMAN_DEVICE"] = "cpu"
os.environ["PACMAN_TORCH_THREADS"] = "1"
import torch
assert torch.cuda.is_available(), "Select a GPU runtime before deployment"
gpu = torch.cuda.get_device_properties(0)
hardware = dict(gpu=gpu.name, gpu_memory_gib=gpu.total_memory/1024**3,
                cpu_count=os.cpu_count(), cuda=torch.version.cuda,
                ram_gib=os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")/1024**3)
print("GPU_HARDWARE:", json.dumps(hardware), flush=True)
test = subprocess.run(''')
builder.DEPLOY = builder.DEPLOY.replace(
    'test.check_returncode()', 'test.check_returncode()\nos.environ["PACMAN_DEVICE"] = "cuda"')
builder.DEPLOY = builder.DEPLOY.replace(
    'print("DEPLOY_READY:", FOLDER, flush=True)',
    '''(FOLDER / "hardware.json").write_text(json.dumps(hardware, indent=1))
print("GPU_DEPLOY_READY:", FOLDER, hardware, flush=True)''')
builder.TRAIN = builder.TRAIN.replace('WORKERS = 2', 'WORKERS = 8')
builder.TRAIN = builder.TRAIN.replace(
    '    meta = json.loads',
    '''    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PACMAN_DEVICE"] = "cuda"
    os.environ["PACMAN_TORCH_THREADS"] = "1"
    import torch
    assert torch.cuda.is_available(), "GPU runtime required; CPU fallback is disabled"
    WORKERS = min(WORKERS, max(1, (os.cpu_count() or 2)-2))
    print("CUDA_TRAINING:", torch.cuda.get_device_name(0), "workers", WORKERS, flush=True)
    meta = json.loads''')
builder.TRAIN = builder.TRAIN.replace('WORKERS * 2 + 2', 'WORKERS * 3 + 4')


if __name__ == "__main__":
    builder.main()
    manifest_path = builder.OUT / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["drive_folder"] = manifest["drive_folder"].replace("comparisons_", "comparisons_gpu_", 1)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    notebook_path = builder.OUT / "Pacman_comparison_study_20261020.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["metadata"]["colab"]["name"] = "Pacman_comparison_GPU_20261020.ipynb"
    notebook["cells"][0]["source"].insert(1, "高性能GPU・ハイメモリ。CUDAで学習、最大8並列。実機はhardware.jsonへ記録。\n")
    notebook["cells"][0]["source"].append("GPUとCPUは別の実行版として保存します。\n")
    (builder.OUT / "Pacman_comparison_GPU_20261020.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
