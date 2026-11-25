# MediaDesignTeam1

自作パックマンの報酬設計実験と、別系統のALE版HRA再現を管理する。
現在の引継ぎは `HANDOFF_Codex.md` を参照。

報酬・パワーえさ・γ・DQN/SARSA・PSOの統合比較は [COMPARISON_STUDY.md](COMPARISON_STUDY.md) を参照。
入口は `comparison_study.py`。グラフの横軸は学習回数（エピソード数）。
GPU対応・高性能ランタイムへの切り替えは [COMPARISON_GPU.md](COMPARISON_GPU.md)、
実機と保存先は [COLAB_COMPARISON_GPU_DEPLOYMENT.md](COLAB_COMPARISON_GPU_DEPLOYMENT.md) を参照。

## 自作環境の改善実験（v2）

Colab CPUハイメモリで `codex_improvements.ipynb` を使用する。
`codex_bundle.zip` は改修済みコードとテストをまとめた転送用ファイル。
旧 `DQN_reward_study` の結果とは別の `DQN_reward_study_v2_20261005` に保存する。

ローカル実行には Python、torch、numpy、gymnasium、pytest が必要。
図表・録画には matplotlib、Pillow、pygame、imageio-ffmpeg を追加する。
実行時のPython・torch・numpy・gymnasiumのバージョンとコードハッシュはmanifestに記録される。

```sh
python -m pytest tests/test_reward_study.py -q
python improvement_study.py --out runs_v2 --workers 4 --steps 200000
```

- 旧モデル用の観測は1699次元のまま。`include_lives=True` の新条件は1700次元。
- 残機1/3、残機入力なし/ありを総環境ステップ数20万で比較する。
- 残機3・入力ありでγ0.95/0.99/0.995 × クリア0/1560を比較する。各条件seed101/202/303。
- 探索用評価はseed910000以降30回。クリア率、次いで得点で条件を選び、選択を保存してから未使用seed920000以降で最良条件と基準条件を評価する。
- 時間上限・学習予算は外部の打ち切りと定義し、真のゲーム終了だけ次状態の価値を0にする。旧実験と学習条件が変わるので別実験として扱う。
- 頭別損失・頭別Q値・総ステップ数・更新回数をlog.csvに保存する。
- 50エピソードごと、予算到達時、学習終了時にtraining.ptを一時ファイルから置換して保存。同一コマンドで最後の保存境界から再開する。checkpointには経験再生・乱数・target・optimizer・進行状態を含む。
- コード・設定・依存環境が変わると実験IDが変わる。旧キャッシュは再利用しない。
- 3学習seedのbootstrap区間は粗い推定。評価30回を独立した30学習試行とは数えない。
- 中断直前の最大49エピソードは再実行される。巨大checkpointのDrive書き込みが失敗した場合は既存checkpointから再開する。

完了後は `foundation.csv`、`validation.csv`、`selection.json`、`heldout.csv`、`COMPLETE.json` を確認する。
PSOの残機3での再探索は、この比較結果を見て別の予算を決める。今回の自動実験には含めない。

## Gitと旧資料

`C:\PRO\Pacman` の既存Git状態は引継ぎ時から不整合がある。reset/cleanは行わない。
正常な別クローン `C:\PRO\MediaDesignTeam1` は確認時main/acb13c3。
旧資料のブランチ名・評価未完了という記載を最新状態と混同しない。
ALE版の情報は `hra/WORK_STATUS.md` を参照。
