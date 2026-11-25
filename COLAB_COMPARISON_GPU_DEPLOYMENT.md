# GPUへの切り替え記録（2026-10-27更新）

2026-10-27：ユーザーの指示で全比較を再開した。
ColabのPythonが3.13.15から3.13.16へ変わったため、uvで3.13.15を用意。
元のライブラリ群を同じsite-packagesから使用し、保存済みplanのruntime全項目との一致を確認。
学習セルにこの復元・照合処理を追加した。学習コード・設定・seed・予算は維持。
完了済み198試行を復元し、親PID17272、8並列で実行中。
未完了のDQN試行が100→150エピソードへ進むこと、GPU使用率90%、失敗記録0件を確認。
REPORT_EVERYは25に変更し、再開時の大量のレポート再生成を抑える。
再開セル：output/colab_comparison_gpu_20261020/resume_training_cell_20261027.py。
これは全実験完了ではない。以下は初回切り替え時の記録。

ノートブック：https://colab.research.google.com/drive/1Uz0XsNZRS4mSDsCkW1JlM3-Eg5roXD1j

名称：Pacman_comparison_GPU_20261020.ipynb

H100を要求したが、空きがなくColabがG4を割り当てた。
実機確認結果：

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition
- GPUメモリ：94.9708 GiB
- CPU：48論理コア
- システムRAM：176.8848 GiB
- CUDA：13.0

学習をCUDAへ配置し、最大8並列、プロセス内のPyTorch CPUスレッド数は1。
学習条件・seed・環境ステップ予算・横軸の学習エピソード数は維持。
GPU種類と実行設定をmanifestへ記録し、CPU版と混ぜない。

CPU版は50エピソード時点の2本（2,242 / 2,355ステップ）のチェックポイントを読み込んで検証。
全ファイルを旧保存先 pre_gpu_snapshot/ へコピーし、SHA-256一致を確認してから4プロセスを停止した。
CPU版保存先：MyDrive/DQN_Pacman_comparisons_20261020_3c20ffba/。
GPU版は変更後のソース・実行環境で新規に開始し、このCPU版の数値を途中へ継ぎ足さない。

GPU版保存先：MyDrive/DQN_Pacman_comparisons_gpu_20261020_9e75afec/。
パッケージ：output/colab_comparison_gpu_20261020/、51ファイル。
SHA-256：9e75afec6a09a84cf5199b2a1bc1d612747d9ad7369f7a50907222189759b0fd。

ローカル：138テスト合格、実GPU専用10件はGPUのないPCではスキップ。
初回Colab検証は147合格・1失敗。実GPU専用10件は合格したが、
既存CPU用DQNのチェックポイント保存がforkされたプロセスでCUDAを初期化して失敗した。
保存・復元時のCUDA RNG操作を、実際のエージェントのデバイスがCUDAの場合に限定して修正。
CPUでCUDA状態へ触れない回帰テストを追加。関連ローカル検証は12合格・GPU専用10件スキップ。
修正版はColabの実GPUで全149テスト合格（59.55秒、スキップなし）。
2026-10-21、ユーザーによる再認証後に配置処理を再実行。
全149テストが再度合格（54.38秒）、DriveマウントとGPU_DEPLOY_READYを確認。
09:20頃（日本時間）、セル1で本学習を開始。親PID24874。
PLAN_START conditions=250 trials=1250 environment_steps=252750000 workers=8を確認。
8本のTRIAL_DEVICEがdevice=cuda、RTX PRO 6000 Blackwell、CUDA13.0を記録。
開始約1分後、8本すべてが50学習エピソードに到達。
各2,327〜2,576環境ステップ、2,200〜2,449回の更新が保存済み。
GPU使用率は確認時94%、8プロセス各820MiB、GPU全体6,606MiB。
FAILED.jsonは0件。この値は開始直後の確認値で、全実験の品質・完了を示すものではない。
証跡は出力フォルダーのcolab_gpu_verification.log、colab_gpu_training_start.log、
gpu_runtime_check.log、gpu_training_started.png。
全比較の完了ではなく、因子比較段階の開始。続いて探索・PSO・再学習段階へ進む。
