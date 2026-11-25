"""Record synchronized raw ALE / arcade frames without changing gameplay.

Explicit config is required: old checkpoints do not contain their configuration.
Life-loss markers report ALE's life counter, not the exact collision instant.
"""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .agent import HRAAgent
from .arcade import ArcadeRenderer
from .env import MsPacmanEnv, RAM_PLAYER, RAM_GHOSTS, RAM_TO_PIX
from .run import load_config


class PairedRecorder:
    def __init__(self, env, writer, log, capture_start=0):
        self.ale = env.ale
        self.renderer = ArcadeRenderer(env, actor_mode='source')
        self.writer, self.log = writer, log
        self.frames = 0
        self.score = 0
        self.last_lives = self.ale.lives()
        self.events = []
        self.event_frame = -1000
        self.capture_start = capture_start

    def act(self, action):
        reward = self.ale.act(action)
        self.frames += 1
        self.score += reward
        lives = self.ale.lives()
        lost = lives < self.last_lives
        if lost:
            self.event_frame = self.frames
            self.events.append({'frame': self.frames, 'seconds': self.frames / 60,
                                'lives_before': self.last_lives, 'lives_after': lives})
        self.last_lives = lives
        ram = self.ale.getRAM()
        coords = [int(ram[a]) for pair in [RAM_PLAYER, *RAM_GHOSTS] for a in pair]
        self.log.writerow([self.frames, self.frames / 60, action, reward, self.score,
                           lives, int(lost), *coords])
        # No temporal frame blending: both panels come from this exact ALE frame.
        if self.frames % 2 == 0 and self.frames >= self.capture_start:
            raw = self.ale.getScreenRGB()
            arcade = self.renderer.frame(raw, self.score, self.score, lives)
            canvas = Image.new('RGB', (1280, 896))
            # Same maze scale AND origin as the reconstructed side (header=64).
            canvas.paste(Image.fromarray(raw[:172]).resize((640, 688), Image.Resampling.NEAREST), (0, 120))
            canvas.paste(Image.fromarray(arcade), (640, 56))
            d = ImageDraw.Draw(canvas)
            d.text((10, 8), 'RAW ALE (original pixels)', fill='white')
            d.text((650, 8), 'REDRAW WALLS / EXACT SOURCE ACTORS', fill='white')
            label = f'frame={self.frames}  t={self.frames / 60:.3f}s  lives={lives}  score={self.score}'
            d.text((10, 30), label, fill='white')
            if self.frames - self.event_frame < 120:
                d.text((650, 30), 'ALE LIFE COUNTER DECREASED (may lag contact)', fill='red')
            self.writer.append_data(np.asarray(canvas))
        return reward

    def __getattr__(self, name):
        return getattr(self.ale, name)


def diagnose(checkpoint, config, out, max_frames=12000, seed=0, focus_first_loss=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # Preserve previous diagnostic evidence.
    cfg = load_config(config)
    with open(checkpoint, 'rb') as f:
        ck = pickle.load(f)
    def prepare():
        env = MsPacmanEnv(cfg, seed=seed)
        env.load_sites(ck['maze_colors'], ck['pellet_sites'])
        ag = HRAAgent(cfg, seed=seed)
        ag.load_state_dict(ck['agent'])
        ag.rng = np.random.default_rng(seed + 1000)
        return env, ag, env.reset()

    capture_start = 0
    if focus_first_loss:
        probe, agent, obj = prepare()
        try:
            t = 0
            while probe.frames < max_frames:
                a = agent.act(obj, agent.state(obj), t)
                obj, _, _, done, info = probe.step(a)
                t += 1
                if info['life_lost']:
                    capture_start = max(0, probe.frames - 240)
                    max_frames = min(max_frames, probe.frames + 120)
                    print(f'First loss near frame {probe.frames}; recording from {capture_start}', flush=True)
                    break
                if done:
                    break
        finally:
            probe.close()
    env, ag, objs = prepare()
    try:
        with open(out / 'frames.csv', 'w', newline='', encoding='utf-8') as f, imageio.get_writer(
                str(out / 'comparison.mp4'), fps=30, macro_block_size=1, quality=7) as writer:
            log = csv.writer(f)
            log.writerow(['frame', 'seconds', 'ale_action', 'reward', 'score', 'lives', 'life_lost',
                          'pac_x', 'pac_y', 'red_x', 'red_y', 'pink_x', 'pink_y',
                          'cyan_x', 'cyan_y', 'orange_x', 'orange_y'])
            recorder = PairedRecorder(env, writer, log, capture_start)
            env.ale = recorder
            t = 0
            while recorder.frames < max_frames:
                action = ag.act(objs, ag.state(objs), t)
                objs, _, _, done, _ = env.step(action)
                t += 1
                if done:
                    break
        report = {'checkpoint': str(Path(checkpoint).resolve()), 'trained_episodes': ck['episode'],
                  'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  'config': cfg, 'seed': seed, 'frames': recorder.frames, 'score': recorder.score,
                  'capture_start_frame': capture_start,
                  'actor_mode': 'source', 'panels_aligned': True,
                  'life_losses': recorder.events, 'ram_to_pixel_offset': RAM_TO_PIX,
                  'note': 'Diagnostic rollout, not benchmark evaluation. No GVF learning or checkpoint writes. '
                          'Life decrement can occur after collision animation. Display shapes are not hitboxes.'}
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: report[k] for k in ('trained_episodes', 'frames', 'score', 'life_losses')}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-frames', type=int, default=12000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--focus-first-loss', action='store_true')
    a = p.parse_args()
    diagnose(a.checkpoint, a.config, a.out, a.max_frames, a.seed, a.focus_first_loss)
