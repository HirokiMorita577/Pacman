# パックマン：報酬・学習方式・PSOの比較実験

入口は comparison_study.py。旧実験のコード・結果を残し、別フォルダーへ保存する。
全グラフの横軸は **学習回数（エピソード数、Learning episodes）**。
PSOの横軸は候補学習に実際に消費したエピソード数の累計。世代数・試行番号ではない。
学習量を揃える停止条件は環境ステップ数であり、CSVには両方を記録する。

## 実行

    pip install -r requirements-comparison.txt
    python comparison_study.py catalog
    python comparison_study.py all --out output/full_study --plan-only
    python comparison_study.py all --out output/full_study --workers 2 --report-every 25

Colabは /content/ を学習先、Driveを --backup で指定する保存先にする。
同じソース・ライブラリ・設定・出力先で再実行すると再開する。版を変えたら出力先も変える。
--report-every 1 なら毎条件PNGを更新する（作図時間も必要）。

標準の全実験は66比較項目、重複排除後250条件×5学習seed＝1,250学習。
因子比較252,750,000環境ステップ。PSO等の探索・独立再学習を含む計画上限は
**3,800,550,000環境ステップ**。評価プレイのステップは別。同一候補の再利用で実消費は減る。
all_plan.json が確定計画。Colab切断後は再接続して同じセルを実行する。
無期限の連続稼働は前提にしない。

機能確認（この短い結果を性能比較には使わない）：

    python comparison_study.py smoke --out output/comparison_smoke --workers 1
    python -m pytest tests -q -p no:cacheprovider

## 比較対象

- 13方式：Q-learning、SARSA、Expected SARSA、DQN、Double DQN、再生ありSARSA、HRA、Dueling DQN、Dueling Double DQN、PPO、ランダム、最寄りえさ、安全経路。
- 報酬：全16項目の重み、項目ごとの無効化、旧報酬・設計報酬・ゲーム得点のみ・えさのみ・クリアのみ・シェーピングなし等。
- パワーえさ報酬：0 / 10 / 25 / 50 / 100 / 200 / 400。
- γ：0.8 / 0.9 / 0.95 / 0.99 / 0.995 / 0.999。
- 学習率、探索開始率・終了率・減衰方式・期間、総学習量、バッチ、再生容量、蓄積量、target更新間隔・ソフト更新率、更新頻度・回数。
- ネットワーク幅・層数、活性化、最適化、損失、勾配クリップ、報酬正規化・クリップ。
- ゴースト数・行動・敵なし対照、残機・残機入力、エピソード上限、観測旧版・安全版・先読み・経路特徴旧版/v2、恐怖終了時刻、接近報酬修正、安全特徴無効化。
- PPO：GAEのλ、クリップ、エントロピー係数、rollout長、更新epoch数、価値損失係数。
- 相互作用：パワーえさ×γ、死亡罰×クリア報酬、学習方式×パワーえさ・γ・報酬セット。

具体的な値と66項目の名前は catalog で表示する。
標準は単独効果と主要相互作用の有限カタログ。連続値の全数・無限の組合せではない。
任意の値・組合せは --grid JSON で追加できる。

    python comparison_study.py run --suites reward_power gamma power_gamma algorithms online_td --out output/power_gamma --workers 2
    python comparison_study.py all --stages factors --out output/factors_only
    python comparison_study.py all --stages search --out output/search_only

カスタムグリッド例（custom_grid.json）：

    {"name":"power_gamma_extra","base":{"algorithm":"double_dqn"},"axes":{"rewards.power":[0,50,150,300],"agent.gamma":[0.90,0.97,0.999]}}

    python comparison_study.py run --grid custom_grid.json --out output/custom --workers 2

共通設定を変更するJSONは --base config.json。
--steps は共通基準の学習量を変えるが、training_budget 比較の学習量は比較として残る。

## PSOと探索法

5空間：γ、パワーえさ＋γ、主要5報酬、学習ハイパーパラメータ、主要報酬＋学習同時。
9方式：未調整基準、ランダム、予算内グリッド、通常PSO、一定慣性、認知重視、社会重視、慣性なし、粒子数2倍。
慣性重み・認知係数・社会係数・粒子数も比較する。PPOにはPPO用の空間を適用する。

    python comparison_study.py search --out output/pso_comparison --algorithms dqn sarsa hra --spaces all --methods all --objectives score_mean reference_return_mean clear_rate

全実験の標準ではDQN、Double DQN、SARSA、Expected SARSA、HRA、PPOを対象とする。
各探索16候補、通常4粒子（大群8粒子）、候補2万ステップ×3学習seed。
未調整基準だけは候補1件。他の探索法は同じ提案予算。
グリッドは予算制限した格子の部分集合。完全探索とは呼ばない。
同一候補は再利用し、提案予算と実学習量を別に記録する。

探索乱数seed41/42/43、探索時の学習seed101/202/303。
選択条件は606/707/808で最初から再学習し、未使用の最終環境seedで評価。
選択を保存してから最終評価する。候補の現在評価と過去最高値を区別する。
得点・共通基準報酬・クリア率を別の目的関数で探索する。
報酬重みを変える探索で、自分の報酬合計を目的にすることは拒否する。

--search-space-file space.json で任意の境界・選択肢を設定できる。
例：{"agent.gamma":{"low":0.8,"high":0.999},"rewards.power":{"low":0,"high":400}}
基準値を境界に含める。
--search-methods-file methods.json でPSO係数を追加できる。
例：{"pso_custom":{"w_start":0.9,"w_end":0.3,"c1":1.7,"c2":1.7}}

## 発表用PSO比較

Double DQN の報酬と環境を基準設定に固定し、学習率 α、割引率 γ、最終探索率 ε を同時に調整する。
PSO、ランダム探索、予算内グリッド探索には候補16件ずつを割り当て、未調整条件も評価する。
候補ごとに3学習seedで各2万環境ステップ学習し、開発評価の平均クリア率で選ぶ。
探索seedは41、42、43。選択後は別の5学習seed（606、707、808、909、1001）で各20万ステップを
最初から学習し、共通の未使用評価盤面でクリア率とゲームスコアを記録する。

- 探索対象は α が 0.00001〜0.003（対数尺度）、γ が 0.8〜0.999、最終 ε が 0.001〜0.2。
- 探索図では、候補の学習に使った累計エピソード数と、開発評価クリア率の過去最高を示す。
- 最終比較図では、方式ごとのスコアとクリア率を示す。学習seedに加えて探索seedを表示し、同じ探索で選んだ設定からの再学習を独立した探索として数えない。
- 選ばれた α、γ、ε は別パネルか表に示す。固定値を振った既存の因子比較と区別する。

計画を確認した時点では未実行。上限は候補学習882万、最終再学習1,200万環境ステップで、評価プレイは別。
以下は学習を開始せず計画だけを出力する。

    python comparison_study.py all --stages search --plan-only --suites learning_rate gamma epsilon_end --out output/local_resume_pso_plan --algorithms double_dqn --spaces hyperparameters --methods baseline random grid pso --objectives clear_rate --search-seeds 41 42 43 --train-seeds 101 202 303 --final-train-seeds 606 707 808 909 1001 --candidates 16 --particles 4 --search-steps 20000

## 結果と読み方

figures/：PNG。summary.csv：最終指標。evaluation_episodes.csv：評価全プレイ。
reward_parameters.csv：報酬設定。paired_differences.csv：同一学習seed間の差。
figure_manifest.json：図一覧と横軸定義。runs/：各条件・seedの設定・ハッシュ・曲線・モデル・評価。
search/：目的関数別の探索結果。

得点、自分の報酬、固定の共通基準報酬、クリア率、えさ回収率、ゴースト捕食、残機損失、
無効行動、損失・TD誤差・勾配等を作図する。
探索パラメータ推移、候補/最高評価、独立再学習、最終評価の図も出力する。
ゲーム得点と学習報酬は別。重みが違う条件間では得点・共通基準報酬を主に比較する。

SARSAは選んだ次行動を更新と次の実行で共有するオンライン方式。
replay_sarsa は保存行動を使う再生あり方式であり、厳密なオンポリシーSARSAとは区別する。
online_td は再生・targetなし同一ネットワークの3更新式比較。
algorithms は方式全体の比較なので、更新回数・パラメータ数・実時間も記録する。
通常は同じ環境ステップ予算・同じ評価環境seed・5学習seed。難易度変更は難易度比較として読む。
95%区間は学習seed間のブートストラップ。100評価プレイを100独立学習とは扱わない。
曲線は共通の学習回数範囲だけ補間し、外挿しない。
学習中は探索込み、開発・最終評価は探索なし（ランダム基準を除く）。
多数比較は探索的分析。多重比較を無視した有意差の断定はしない。

`output/all_comparisons_20261125/index.html` から66比較項目の図を確認できる。
「学習時の平均」はエピソード番号が一致する学習seedの平均で、平滑化しない。
「損失」は各seedの直近50エピソードで記録された値を平均し、共通範囲でseed間を平均する。
損失を記録しない方式や未取得の条件には0を入れない。
条件別CSVは学習時と最終評価の平均、学習seed数、学習エピソード範囲を併記する。

## 再開・容量・失敗

学習・探索乱数、再生バッファ、PPO途中rolloutも保存する。
Driveは照合付きA/Bの2世代。破損した新保存ではなく正常保存を復元する。
容量不足ではローカルと最後の正常バックアップを残す。
完了モデルをDriveへ検証保存した後、その実験自身の大きな再生用チェックポイントを整理する。
探索候補モデルは選択・最終再学習終了後に整理し、設定・指標・ログ・選択結果・最終モデルを残す。
候補モデルも残す場合は --retain-candidate-models（容量は増える）。別実験ファイルは削除しない。
失敗はFAILED.jsonとして記録し、0点にせず他条件を継続する。
--retry-failed で再試行。ソース変更時は別フォルダー。COMPLETE.json の failed_trials を確認する。
全条件終了と、一時停止・切断を区別する。
