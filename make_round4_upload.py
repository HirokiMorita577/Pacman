"""
第4回比較 (ghost4_round4) を Colab で回すための Drive アップロード用フォルダを作る (make_round3_upload と同じ方式)。

  python make_round4_upload.py --out output/ghost4_round4_upload

出力: <out>/DQN_ghost4_round4/ (source/・source_hashes.json・colab_launch_round4.py・launcher.json) と <out>/colab_cell.txt。
フォルダごと Drive の MyDrive/ に置く (Colab でハッシュを照合してから展開してもよい)。
"""
import make_round3_upload as base

base.DRIVE_DIR = "DQN_ghost4_round4"
base.LAUNCHER_VERSION = 1
base.SOURCES = base.SOURCES + ["ghost4_round4.py", "make_round3_upload.py", "make_round4_upload.py"]
base.LAUNCHER = (base.LAUNCHER
                 .replace("第3回比較 (launcher", "第4回比較 (launcher")
                 .replace("/content/ghost4_round3_code", "/content/ghost4_round4_code")
                 .replace("/content/ghost4_round3_run.log", "/content/ghost4_round4_run.log")
                 .replace("'第3回比較: 9条件 x 3 seed → 開発用評価 → 選択 → 学習seed 5本で未使用seed 950000〜 の最終評価'",
                          "'第4回比較: safe / safe_lookahead x 学習seed 5本 → 開発用評価 (各100回) → 未使用seed 960000〜 の最終評価'")
                 .replace("'--module', 'ghost4_round3'", "'--module', 'ghost4_round4'")
                 .replace("'/content/ghost4_round3_study'", "'/content/ghost4_round4_study'")
                 .replace("'--workers', '8']", "'--workers', '10', '--seeds', '101', '202', '303', '404', '505']")
                 .replace("'第3回比較が完了しました。", "'第4回比較が完了しました。"))
base.CELL_NAME = "colab_launch_round4.py"
base.ROUND_LABEL = "第4回比較"
for old in ("第3回", "ghost4_round3", "'--workers', '8']"):
    assert old not in base.LAUNCHER, "launcher substitution missed: " + old

if __name__ == "__main__":
    base.main()
