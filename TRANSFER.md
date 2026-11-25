# 別PCでの引き継ぎ

このリポジトリには、パックマン比較実験のコード、テスト、手順書、軽量な図表と発表資料を含めています。最新の作業状況は `HANDOFF_Codex.md` を参照してください。

## 取得

```sh
git clone https://github.com/HirokiMorita577/Pacman.git
cd Pacman
```

`requirements.txt` と `requirements-comparison.txt` を確認し、実行先に合わせて依存関係を導入してください。比較実験の入口は `comparison_study.py`、GPU版の手順は `COLAB_COMPARISON_GPU_DEPLOYMENT.md` と `COMPARISON_GPU.md` にあります。

## Git外の学習データ

進行中の1250試行の結果、モデル、再開用チェックポイントはGitには入れていません。Google Driveの `MyDrive/DQN_Pacman_comparisons_gpu_20261020_9e75afec/` が保存先です。Colabノートブックは `https://colab.research.google.com/drive/1Uz0XsNZRS4mSDsCkW1JlM3-Eg5roXD1j` です。別PCで再開する際は同じGoogleアカウントのDriveへ接続し、`COLAB_COMPARISON_GPU_DEPLOYMENT.md` の環境・コード・チェックポイント照合を通してください。

Gitに含む `output/colab_comparison_gpu_20261020/` は配置・再開用コードと当時の記録です。`output/dqn_vs_double_dqn_20261111/` などには発表用の軽量な比較図表を含めています。容量の大きい学習途中の出力、仮想環境、作業用キャッシュは含めていません。

2026年11月25日のDrive実ファイル確認では827/1250試行が完了し、失敗は0件でした。Colab実プロセスとGPUの稼働状態は未確認です。保存済みファイルの更新停止だけで停止と判断せず、重複起動を避けてください。
