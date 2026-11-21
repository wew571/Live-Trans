from types import SimpleNamespace

import numpy as np
import pytest

from core.window_capture import _WindowFrames


def test_window_capture_discards_old_sizes_and_copies_latest_bgr():
    frames = _WindowFrames()
    region = (10, 20, 4, 3)
    assert frames.grab(region) is None

    frames.on_frame_arrived(SimpleNamespace(frame_buffer=np.zeros((2, 4, 4), dtype=np.uint8)), None)
    assert frames.grab(region) is None

    pixels = np.full((3, 4, 4), [1, 2, 3, 255], dtype=np.uint8)
    frames.on_frame_arrived(SimpleNamespace(frame_buffer=pixels), None)
    result = frames.grab(region)
    pixels[:] = 0
    assert result.flags.c_contiguous
    assert result.shape == (3, 4, 3)
    assert tuple(result[0, 0]) == (1, 2, 3)
    assert frames.grab(region) is None

    frames.on_closed()
    with pytest.raises(RuntimeError, match="stopped capturing"):
        frames.grab(region)
