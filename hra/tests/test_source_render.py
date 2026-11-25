from types import SimpleNamespace
import numpy as np
from hra.arcade import ArcadeRenderer, HEADER, S, MAZE_ROWS
from hra.env import C_PACMAN, C_GHOSTS


def test_source_mode_preserves_sprite_extent_and_death_shape():
    env = SimpleNamespace(maze=0, wall=12345, background=0, crop_top=6,
                          pellet_sites={}, remaining=[], direction=1, level=1,
                          ale=SimpleNamespace(getRAM=lambda: np.zeros(128, dtype=np.uint8)))
    render = ArcadeRenderer(env, actor_mode='source')
    # Deliberately asymmetric shapes: replacing with a circle would fail this test.
    raw = np.zeros((210, 160, 3), dtype=np.uint8)
    for code, ys, xs in [(C_PACMAN, slice(70, 74), slice(50, 52)),
                         (C_GHOSTS[0], slice(65, 70), slice(60, 64))]:
        raw[ys, xs] = [(code >> 16) & 255, (code >> 8) & 255, code & 255]
    frame = render.frame(raw, 0, 0, 3)
    expected = np.repeat(np.repeat(raw[:MAZE_ROWS], S, 0), S, 1)
    np.testing.assert_array_equal(frame[HEADER:HEADER + MAZE_ROWS*S], expected)
    raw[:] = 0
    frame = render.frame(raw, 0, 0, 2)
    assert not frame[HEADER:HEADER + MAZE_ROWS*S].any()
