"""Finite, inspectable comparison catalog plus arbitrary JSON factorial grids."""
import copy
from itertools import product

from pacman_env import OLD_REWARDS, DESIGNED_REWARDS, REWARD_SPEC, SHAPING_ITEMS
from .config import ALGORITHMS, DEFAULT, REFERENCE_REWARDS, canonical, digest, merge, set_path, validate


def patch(path, value):
    result = {}
    set_path(result, path, value)
    return result


def catalog():
    suites = {}

    def add(name, description, variants, reference=True):
        suites[name] = dict(description=description, variants=variants, include_reference=reference)

    def axis(name, description, path, values):
        add(name, description, [(str(v), patch(path, v)) for v in values])

    add("algorithms", "アルゴリズム（学習方式全体の比較。更新回数・パラメータ数も記録）",
        [(a, {"algorithm": a}) for a in ALGORITHMS])
    add("online_td", "再生・targetを使わない同一構造のQ-learning / SARSA / Expected SARSA",
        [(a, {"algorithm": a}) for a in ("q_learning", "sarsa", "expected_sarsa")], reference=False)
    add("replay_td", "再生・targetを揃えたDQN / Double DQN / 保存行動SARSA（後者はオンポリシーではない）",
        [(a, {"algorithm": a}) for a in ("dqn", "double_dqn", "replay_sarsa")])
    zero = {k: 0.0 for k in REWARD_SPEC}
    reward_sets = {
        "reference": REFERENCE_REWARDS,
        "old": OLD_REWARDS,
        "designed": DESIGNED_REWARDS,
        "game_score_only": {**zero, "dot": 10.0, "power": 50.0, "eat_ghost": 200.0},
        "food_only": {**zero, "dot": 10.0, "power": 50.0},
        "sparse_clear": {**zero, "clear": 1560.0},
        "score_and_death": {**zero, "dot": 10.0, "power": 50.0, "eat_ghost": 200.0, "death": 100.0},
        "no_shaping": {**REFERENCE_REWARDS, **{k: 0.0 for k in SHAPING_ITEMS}},
    }
    add("reward_sets", "報酬セット全体の比較", [(n, {"rewards": r}) for n, r in reward_sets.items()])
    reward_values = {
        "death": [0, 25, 100, 500, 1560, 15600],
        "clear": [0, 100, 500, 1560, 5000],
        "power": [0, 10, 25, 50, 100, 200, 400],
        "eat_ghost": [0, 50, 100, 200, 400, 800],
        "dot": [0, 2, 5, 10, 20],
        "time": [0, 0.02, 0.1, 0.5, 1],
    }
    for name in REWARD_SPEC:
        base = REFERENCE_REWARDS[name]
        values = reward_values.get(name, sorted(set([0.0, base / 2, base, base * 2, base * 5] + ([1, 5] if not base else []))))
        axis("reward_" + name, REWARD_SPEC[name][2] + "の重み", "rewards." + name, values)
    add("reward_ablation", "各報酬項目を1つずつ0にする寄与比較",
        [(k, {"rewards": {k: 0.0}}) for k in REWARD_SPEC])
    axis("gamma", "割引率γ", "agent.gamma", [0.8, 0.9, 0.95, 0.99, 0.995, 0.999])
    axis("learning_rate", "学習率", "agent.lr", [0.00001, 0.0001, 0.0005, 0.001, 0.003])
    axis("epsilon_end", "最終探索率", "train.epsilon_end", [0, 0.01, 0.0248, 0.05, 0.1, 0.2])
    axis("epsilon_start", "初期探索率", "train.epsilon_start", [0.25, 0.5, 1.0])
    axis("exploration_steps", "探索期間（総学習量から独立）", "train.explore_steps", [10000, 40000, 80000, 160000])
    axis("exploration_schedule", "探索率の減らし方", "train.schedule", ["linear", "exponential", "cosine", "constant"])
    axis("training_budget", "総環境ステップ数（探索8万stepを維持）", "train.steps", [50000, 100000, 200000, 400000, 800000])
    axis("batch_size", "ミニバッチサイズ", "algorithm_options.batch_size", [32, 64, 128, 256])
    axis("replay_capacity", "経験再生容量", "algorithm_options.buffer_capacity", [10000, 50000, 100000, 200000])
    axis("warmup", "学習開始前の蓄積量", "algorithm_options.warmup", [128, 1000, 5000])
    axis("target_update", "targetのハード更新間隔（勾配更新回数）", "algorithm_options.target_update", [100, 500, 1000, 5000])
    axis("target_soft_update", "targetのソフト更新率", "algorithm_options.target_tau", [0.001, 0.005, 0.01, 0.05])
    axis("train_every", "学習を行う環境ステップ間隔", "algorithm_options.train_every", [1, 4, 8])
    axis("updates_per_step", "1回あたり勾配更新回数", "algorithm_options.updates_per_step", [1, 2, 4])
    add("replay_ablation", "Double DQNの再生/targetの寄与",
        [("no_replay", {"algorithm_options": {"replay": False, "buffer_capacity": 1, "batch_size": 1, "warmup": 1}}),
         ("no_target", {"algorithm_options": {"use_target": False}}),
         ("online", {"algorithm_options": {"replay": False, "use_target": False, "buffer_capacity": 1,
                                           "batch_size": 1, "warmup": 1}})])
    axis("network", "ネットワーク幅・層数", "agent.hidden_sizes", [[64, 32], [128, 64], [256, 128], [512, 256, 128], [1024, 512, 256]])
    axis("activation", "活性化関数", "agent.activation", ["relu", "tanh", "leaky_relu"])
    axis("optimizer", "最適化手法", "agent.optimizer", ["adam", "rmsprop", "sgd"])
    axis("loss", "TD誤差の損失", "agent.loss", ["huber", "mse"])
    axis("gradient_clip", "勾配クリップ", "agent.grad_clip", [None, 0.5, 1.0, 5.0])
    axis("reward_normalization", "学習報酬の正規化", "train.reward_normalization", ["none", "pellet", "fixed100"])
    axis("reward_clipping", "学習報酬のクリップ（HRAの合計報酬も揃える）", "train.reward_clip", [None, 1.0, 5.0])
    axis("action_mask", "壁方向の行動マスク", "train.mask_actions", [False, True])
    axis("truncation", "外部打切り時のbootstrap（Falseは従来方式の寄与比較）", "train.bootstrap_truncation", [True, False])
    axis("ghost_count", "ゴースト数", "env.num_ghosts", [1, 2, 3, 4])
    axis("ghost_policy", "ゴーストの行動", "env.ghost_mode", ["classic", "bfs"])
    axis("ghost_free", "敵なしの対照条件", "env.no_ghosts", [False, True])
    axis("lives", "残機数", "env.lives", [1, 3, 5])
    axis("lives_observation", "残機入力", "env.include_lives", [False, True])
    axis("episode_limit", "1エピソードの上限", "env.max_steps", [250, 500, 1000, 2000])
    add("observations", "状態表現",
        [("legacy", {"env": {"observation_version": "legacy", "classic_lookahead": False}}),
         ("tactical", {"env": {"observation_version": "tactical", "classic_lookahead": False}}),
         ("safe", {"env": {"classic_lookahead": False}}),
         ("lookahead", {"env": {"classic_lookahead": True}}),
         ("route_legacy", {"env": {"safe_route": True, "route_version": "legacy"}}),
         ("route_v2", {"env": {"safe_route": True, "route_version": "v2"}})])
    axis("fear_timing", "怖がり終了時刻を考慮", "env.safety_timing", [False, True])
    axis("leave_dot_fix", "エサ取得直後の接近罰の修正", "env.fix_leave_dot", [False, True])
    add("safety_ablation", "安全特徴群の寄与",
        [("no_" + key, {"env": {"safety_groups": [k for k in ("area", "dots", "margin") if k != key]}})
         for key in ("area", "dots", "margin")])
    for name, key, values in (
        ("ppo_lambda", "gae_lambda", [0.8, 0.95, 1.0]),
        ("ppo_clip", "clip_eps", [0.1, 0.2, 0.3]),
        ("ppo_entropy", "entropy_coef", [0.0, 0.01, 0.05]),
        ("ppo_rollout", "rollout_steps", [128, 512, 2048]),
        ("ppo_epochs", "epochs", [1, 4, 8]),
        ("ppo_value", "value_coef", [0.25, 0.5, 1.0]),
    ):
        add(name, "PPO: " + key, [(str(v), {"algorithm": "ppo", "algorithm_options": {key: v}}) for v in values],
            reference=False)
    add("power_gamma", "パワーエサ報酬×γ（交互作用）",
        [(f"power={p},gamma={g}", {"rewards": {"power": p}, "agent": {"gamma": g}})
         for p, g in product([0, 10, 50, 200], [0.9, 0.95, 0.99, 0.995])])
    add("death_clear", "死亡罰×クリア報酬",
        [(f"death={d},clear={c}", {"rewards": {"death": d, "clear": c}})
         for d, c in product([25, 100, 500, 1560], [0, 500, 1560, 5000])])
    add("algorithm_power", "アルゴリズム×パワーエサ報酬",
        [(f"{a},power={p}", {"algorithm": a, "rewards": {"power": p}})
         for a, p in product(["dqn", "double_dqn", "sarsa", "expected_sarsa", "hra", "ppo"], [0, 10, 50, 200])])
    add("algorithm_gamma", "アルゴリズム×γ",
        [(f"{a},gamma={g}", {"algorithm": a, "agent": {"gamma": g}})
         for a, g in product(["dqn", "double_dqn", "sarsa", "expected_sarsa", "hra", "ppo"], [0.9, 0.95, 0.99, 0.995])])
    add("algorithm_rewards", "アルゴリズム×報酬セット",
        [(f"{a},{r}", {"algorithm": a, "rewards": reward_sets[r]})
         for a, r in product(["dqn", "double_dqn", "sarsa", "expected_sarsa", "hra", "ppo"],
                             ["reference", "old", "game_score_only", "sparse_clear"])])
    return suites


def build_plan(suites, seeds, base=None, grid=None):
    if not seeds or len(set(seeds)) != len(seeds) or any(
            isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Use unique nonnegative training seeds below 2**32")
    base = merge(DEFAULT, base or {})
    available = catalog()
    if "all" in suites:
        priority = ["reward_power", "gamma", "algorithms", "online_td", "reward_sets",
                    "power_gamma", "algorithm_power", "algorithm_gamma"]
        suites = priority + [name for name in available if name not in priority]
    unknown = set(suites) - set(available)
    if unknown:
        raise ValueError("Unknown suites: " + ", ".join(sorted(unknown)))
    conditions, groups = {}, {}

    def register(group, label, changes):
        starting = copy.deepcopy(base)
        if changes.get("algorithm", starting["algorithm"]) != starting["algorithm"]:
            starting["algorithm_options"] = {}
        cfg = validate(merge(starting, changes))
        key = digest(cfg)[:20]
        conditions.setdefault(key, dict(id=key, config=cfg))
        members = groups.setdefault(group, [])
        existing = next((m for m in members if m["condition"] == key), None)
        if existing is None:
            members.append(dict(condition=key, label=label, changes=changes))
        elif existing["label"] == "reference" and label != "reference":
            existing["label"] = "reference (" + label + ")"

    for name in dict.fromkeys(suites):
        suite = available[name]
        if suite["include_reference"]:
            register(name, "reference", {})
        for label, changes in suite["variants"]:
            register(name, label, changes)
    if grid:
        if set(grid) - {"name", "base", "axes"}:
            raise ValueError("Unknown grid fields")
        axes = grid.get("axes", {})
        if not axes or any(not isinstance(v, list) or not v for v in axes.values()):
            raise ValueError("Grid axes must be nonempty lists")
        for values in product(*axes.values()):
            changes = copy.deepcopy(grid.get("base", {}))
            for path, value in zip(axes, values):
                set_path(changes, path, value)
            register(grid.get("name", "custom_grid"),
                     ", ".join(f"{k}={v}" for k, v in zip(axes, values)), changes)
    configs = list(conditions.values())
    if not configs:
        raise ValueError("Choose at least one suite or grid")
    total_steps = sum(c["config"]["train"]["steps"] for c in configs) * len(seeds)
    return dict(schema=1, seeds=list(seeds), conditions=configs, groups=groups,
                unique_conditions=len(configs), training_trials=len(configs) * len(seeds),
                total_environment_steps=total_steps,
                objective="score_mean",
                notes=["All PNG x axes are learning episodes, not environment steps.",
                       "Online SARSA has no replay/target; replay_sarsa is a separate off-policy variant.",
                       "Changing episode count/survival changes learning-episode duration; step budgets are equal unless varied.",
                       "Own-reward returns are not directly comparable across different reward definitions."])
