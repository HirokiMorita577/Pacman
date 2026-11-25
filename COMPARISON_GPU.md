# 比較実験のGPU実行

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
