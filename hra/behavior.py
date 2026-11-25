"""Passive metrics: reversals are A-B-A cell moves, not joystick changes."""
from collections import deque
import json


class BehaviorMetrics:
    def __init__(self, trace=None):
        self.path = deque(maxlen=3)
        self.moves = self.reversals = self.deaths = self.stall_frames = 0
        self.no_reward_frames = self.max_no_reward_frames = 0
        self.trace = trace
        self.recent = deque(maxlen=100)
        self.last_dump = -10000
        self.active = False

    def decision(self, env, agent, objs, state, step):
        if self.trace is None:
            return
        scores, ghosts = agent.head_values(objs, state) if state is not None else (None, None)
        d = agent.last_decision
        self.recent.append(dict(frame=env.frames, step=step, maze=objs.maze,
                                position=objs.pacman, remaining=len(objs.pellets) + len(objs.power_pellets),
                                action=d['action'], ties=d['ties'], q=d['q'].tolist(),
                                score_head=None if scores is None else scores.tolist(),
                                ghost_head=None if ghosts is None else ghosts.tolist()))

    def dump(self, reason):
        if self.trace is not None and self.recent:
            self.trace.write(json.dumps({'reason': reason, 'decisions': list(self.recent)}) + '\n')
            self.trace.flush()

    def observe(self, before, after, reward, info, frames):
        if info['life_lost']:
            self.deaths += 1
            self.dump('life_counter_decreased_not_exact_contact_time')
        valid = (not info['life_lost'] and not info['level_cleared']
                 and not info['in_transition'] and before.maze == after.maze
                 and before.pacman is not None and after.pacman is not None)
        if not valid:
            self.path.clear()
            self.no_reward_frames = 0
            self.active = False
            return
        if after.pacman != before.pacman:
            self.active = True
        if not self.active:
            return
        if not self.path:
            self.path.append(before.pacman)
        if after.pacman != before.pacman:
            self.moves += 1
            self.path.append(after.pacman)
            if len(self.path) == 3 and self.path[0] == self.path[2]:
                self.reversals += 1
        else:
            self.stall_frames += frames
        self.no_reward_frames = self.no_reward_frames + frames if reward == 0 else 0
        self.max_no_reward_frames = max(self.max_no_reward_frames, self.no_reward_frames)
        if self.no_reward_frames >= 600 and self.no_reward_frames - self.last_dump >= 600:
            self.dump('no_reward_600_frames')
            self.last_dump = self.no_reward_frames
        if reward:
            self.last_dump = -10000

    def summary(self):
        return dict(moves=self.moves, reversals=self.reversals, deaths=self.deaths,
                    reversals_per_100_moves=100 * self.reversals / max(1, self.moves),
                    stationary_frames=self.stall_frames,
                    max_no_reward_frames=self.max_no_reward_frames)
