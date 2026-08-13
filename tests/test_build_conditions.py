"""The check that says an engine was built under a clock that was moving.

Why this is worth a test of its own: the defect it guards against is invisible
by construction. The engine is valid, the fingerprint matches, the benchmark
runs, the accuracy check passes -- and the engine is still one of several
different engines TensorRT would have produced from the same inputs. Four
builds of unik3d gave 19.04, 13.35, 8.41 and 14.49 ms that way. The only
evidence is the pair of clock readings taken either side of the build, so the
rule that reads them has to be right.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.build_conditions import complaint, path_for  # noqa: E402


def _info(before, after, top=2130):
    return {"clock_before_mhz": before, "clock_after_mhz": after,
            "clock_max_mhz": top}


def test_a_pinned_build_has_nothing_to_say():
    # A pin holds the clock at idle too, so both samples read the same.
    assert complaint(_info(1800, 1800), now_mhz=1800) is None


def test_a_moving_clock_is_reported():
    # 210 idle before, boosted after several minutes of building.
    msg = complaint(_info(210, 1935))
    assert msg and "free GPU clock" in msg
    assert "210" in msg and "1935" in msg


def test_a_build_at_the_maximum_is_still_free_if_it_moved():
    # Pinning at the card's maximum is legitimate; drifting up to it is not,
    # and only the two samples tell them apart.
    assert complaint(_info(2130, 2130)) is None
    assert complaint(_info(1200, 2130)) is not None


def test_running_at_a_different_clock_than_the_build():
    msg = complaint(_info(1800, 1800), now_mhz=1500)
    assert msg and "not comparable" in msg


def test_an_engine_with_no_record_is_reported():
    assert "before build conditions were recorded" in complaint(None)
    assert "before build conditions were recorded" in complaint({})


def test_missing_readings_are_not_turned_into_a_complaint():
    # nvidia-smi absent, or a GPU that does not report clocks: unknown is not
    # the same as bad, and crying wolf here would get the check ignored.
    assert complaint(_info(0, 0, 0)) is None
    assert complaint({"clock_before_mhz": 1800}) is None


def test_the_record_sits_beside_its_engine():
    assert path_for("engine/model_fp16.engine") == "engine/model_fp16.buildinfo.json"
