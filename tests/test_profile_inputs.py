"""Layer profiles must run with the same complete input set as benchmarks."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import profile_model  # noqa: E402


class Slot:
    def __init__(self, size, dtype=np.float32):
        self.host = np.empty(size, dtype=dtype)


def tr2m_spec():
    return {
        "input": {"name": "image"},
        "extra_inputs": [{"name": "text_features"}],
    }


def test_saved_inputs_follow_the_spec_order_and_keep_the_values(tmp_path):
    image = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    text = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
    np.save(tmp_path / "tr2m.npy", image)
    np.save(tmp_path / "tr2m__text_features.npy", text)

    feeds = profile_model.saved_inputs("tr2m", tr2m_spec(), str(tmp_path))

    assert [name for name, _ in feeds] == ["image", "text_features"]
    assert np.array_equal(feeds[0][1], image)
    assert np.array_equal(feeds[1][1], text)


def test_every_engine_input_binding_is_filled():
    feeds = [
        ("image", np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)),
        ("text_features", np.arange(6, dtype=np.float32).reshape(1, 1, 6)),
    ]
    slots = [Slot(12), Slot(6)]

    profile_model.bind_saved_inputs(slots, feeds)

    assert np.array_equal(slots[0].host, feeds[0][1].ravel())
    assert np.array_equal(slots[1].host, feeds[1][1].ravel())


def test_an_unfilled_or_undeclared_binding_is_a_hard_failure():
    feeds = [("image", np.zeros((1, 3, 2, 2), np.float32))]
    with pytest.raises(ValueError, match="2 input bindings"):
        profile_model.bind_saved_inputs([Slot(12), Slot(6)], feeds)


def test_a_missing_recorded_extra_input_is_a_hard_failure(tmp_path):
    np.save(tmp_path / "tr2m.npy", np.zeros((1, 3, 2, 2), np.float32))
    with pytest.raises(FileNotFoundError, match="text_features"):
        profile_model.saved_inputs("tr2m", tr2m_spec(), str(tmp_path))
