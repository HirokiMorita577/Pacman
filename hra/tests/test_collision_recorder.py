import csv
import io
from types import SimpleNamespace

import numpy as np

from hra.diagnose_collision import PairedRecorder


def test_recording_does_not_change_actions_rewards_or_lives():
    class ALE:
        def __init__(self):
            self.actions = []

        def act(self, a):
            self.actions.append(a)
            return a

        def lives(self):
            return 3 if len(self.actions) < 3 else 2

        def getRAM(self):
            return np.arange(128, dtype=np.uint8)

        def getScreenRGB(self):
            return np.full((210, 160, 3), len(self.actions), dtype=np.uint8)

    ale = ALE()
    frames = []
    stream = io.StringIO()
    rec = PairedRecorder(SimpleNamespace(ale=ale), SimpleNamespace(append_data=frames.append), csv.writer(stream))
    observed = []

    def render(raw, *args):
        observed.append(int(raw[0, 0, 0]))
        return np.full((796, 640, 3), raw[0, 0, 0], dtype=np.uint8)

    rec.renderer.frame = render
    assert [rec.act(a) for a in [1, 2, 3, 4]] == [1, 2, 3, 4]
    assert ale.actions == [1, 2, 3, 4]
    assert observed == [2, 4]
    assert len(rec.events) == 1 and rec.events[0]['frame'] == 3
    assert len(stream.getvalue().splitlines()) == 4
    assert frames[-1][200, 10, 0] == frames[-1][200, 650, 0] == 4
