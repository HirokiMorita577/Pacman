# 比較実験のGPU実行

## Windows PC で Colab の途中チェックポイントを引き継ぐ

Windows の CUDA 環境へ移す場合は `resume_local.py` を使う。Colab の保存先から
`results/factors/plan.json` と、未完了の試行ごとの `manifest.json`、
`backup_state.json`、そのレシートが指す `recovery_A.pt.gz` または
`recovery_B.pt.gz` をローカルにコピーする。試行フォルダー名は Drive 上と揃える。
ファイルは自身の信頼できる実験から取得する。

```powershell
$pcPython = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
& $pcPython -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch>=2.12.0"
.\.venv\Scripts\python.exe -m pip install -r requirements-comparison.txt
$env:PACMAN_DEVICE = "cuda"
$env:PACMAN_TORCH_THREADS = "1"
$trials = (Get-ChildItem output/local_resume_source -Directory).FullName
.\.venv\Scripts\python.exe resume_local.py --source-plan output/local_resume_source/plan.json --source-trial $trials --out output/local_resume_rtx3060ti
```

上記の torch 指定を使う場合は CUDA 版の配布インデックスと互換ドライバーが必要。
`--verify-only` では PyTorch を使わずに計画、試行 manifest、ソース、チェックポイントの SHA-256 を検査できる。
`--prepare-only` で全チェックポイントの照合と配置だけを先に実行できる。
学習は一試行ずつ実行する。同じ引数で再実行するとローカルに保存された最新地点から再開する。

このPCで移行済みの8件は、停止後に PowerShell で `.\run_local.ps1` を実行すると保存地点の
続きから動く。学習は50エピソードごとと試行終了時にチェックポイントを保存する。
強制終了や停電からの再開では、最後の保存以降の分をやり直す。
稼働中の学習を別ウィンドウで見るには `.\run_local.ps1 -Watch` を実行する。
監視画面は差分ログを毎秒読み、GPU使用率と使用メモリを5秒ごとに更新する。
学習ステップ数の表示は、学習側のチェックポイント保存時に更新される。
監視ウィンドウを閉じても学習プロセスは動き続ける。PCのスリープ中は計算が中断される。

`migration.json` に復元したエピソード・環境ステップ、移行元と移行先の manifest、
元の計画とチェックポイントの SHA-256 が残る。ネットワーク、optimizer、
再生バッファ、乱数状態と学習履歴を引き継ぐ。GPU と実行環境が変わるため、
この継続実験の成績は Colab の統合比較に合算せず、ローカル出力先に保存する。

学習途中の重みを調べる場合は、チェックポイントのコピーをCPUで開き、
開発用の共通評価seedで探索なし評価を行う。

```powershell
.\.venv\Scripts\python.exe evaluate_local_checkpoints.py --runs output/local_resume_rtx3060ti/runs --out output/local_resume_rtx3060ti/checkpoint_evaluation --workers 6
```

評価結果には保存時点の環境ステップ、エピソード数、チェックポイントの SHA-256 を記録する。
この開発評価は、学習終了後の未使用seedによる最終評価と別に保存する。

## 元の Colab 実験

高性能GPU・ハイメモリのColabランタイムを使用する。実際のGPU機種は hardware.json に記録する。
全学習方式（DQN、HRA、SARSA、PPO等）のネットワーク、更新用テンソルをCUDAに配置する。
ランダム・経路探索の基準方式とパックマン環境はCPUで実行する。
CUDAが利用できない場合は停止し、気づかずCPUで長時間実行することを防ぐ。

PACMAN_DEVICE=cuda、PACMAN_TORCH_THREADS=1。
最大8条件を並列学習し、CPUコア数とRAMが足りなければ並列数を減らす。
複数プロセスが同じGPUを共有する。環境シミュレーションや小さなネットワークではGPU利用率が低い場合もある。
マシン名だけから速度向上率を断定せず、TRIAL_PROGRESSと経過時間で確認する。

CPU版の途中データは旧Drive保存先の pre_gpu_snapshot/ に、チェックポイントを読み込み・ハッシュ照合して保存した。
GPU版はソースと実行環境が変わるため、別フォルダーで学習を開始する。CPUとGPUの途中結果を同じ曲線へ混ぜない。
横軸は引き続き学習エピソード数。比較する設定・seed・学習量はCPU版と同じ。

モデル・再生バッファ・optimizer・CPU/GPU乱数状態を保存する。
再開時はGPU機種、CUDA、ライブラリ、コードと設定を照合する。
同じGPU実行環境での停止・再開が連続学習と同じ重みになることを、全10学習方式の実GPUテストで検証する。
CPU版の既存テストも実行する。

生成は python build_comparison_gpu_notebook.py。
転送パッケージは output/colab_comparison_gpu_20261020/。
旧CPUパッケージ output/colab_comparison_study_20261020/ は保存する。
