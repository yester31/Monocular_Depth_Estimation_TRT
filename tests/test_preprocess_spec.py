"""Every model declares how its input is preprocessed, and the claim is checked.

P1 wants the preprocessing type decidable from spec.json rather than by reading
fourteen scripts. Writing it down is the easy half; the half that matters is
that the declaration keeps matching the code.

So an adapter_check declaration has to be backed by an evaluate_gt adapter.
The structured record also says whether the comparison was exact or merely
within the accepted tolerance; free-form wording does not control the test.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TYPES = {"stretch", "keep_ratio_pad"}
NORMS = {"imagenet", "over_255", "half", "none"}
INTERPS = {"linear", "cubic", "area", "torch_bilinear",
           # torchvision transforms.Resize on a PIL image: bilinear WITH
           # antialias, which is neither cv2 INTER_LINEAR nor torch's
           # F.interpolate bilinear. hyden is the first model to use it.
           "pil_bilinear_antialias"}


def specs():
    from core.spec import load_all
    return load_all()


def test_every_model_declares_its_preprocessing():
    missing = [m for m, s in specs().items() if "preprocess" not in s]
    assert not missing, (
        f"no preprocess block in: {', '.join(sorted(missing))}. P1 requires the "
        f"type to be decidable from spec.json.")


@pytest.mark.parametrize("field,allowed", [("type", TYPES), ("normalise", NORMS),
                                           ("interpolation", INTERPS)])
def test_declared_values_come_from_a_closed_set(field, allowed):
    bad = {m: s["preprocess"].get(field) for m, s in specs().items()
           if s.get("preprocess", {}).get(field) not in allowed}
    assert not bad, (
        f"{field} outside {sorted(allowed)}: {bad}. A new value has to be added "
        f"here so it is a decision rather than a typo.")


def test_a_verified_claim_has_an_adapter_behind_it():
    """The point of the test: verification cannot be asserted, only earned.

    evaluate_gt cannot be imported here -- it needs OpenCV -- so the adapter
    table is read out of the source. That is enough: the claim is about which
    models have one.
    """
    src = open(os.path.join(ROOT, "tools", "evaluate_gt.py"), encoding="utf-8").read()
    claimed = {m for m, s in specs().items()
               if "adapter_check" in s.get("preprocess", {})}
    unbacked = [m for m in claimed if f'"{m}": {{' not in src]
    assert not unbacked, (
        f"these declare an adapter check and evaluate_gt has none: "
        f"{', '.join(sorted(unbacked))}")


def test_adapter_check_states_the_observed_difference_and_tolerance():
    for model, s in specs().items():
        pre = s.get("preprocess", {})
        check = pre.get("adapter_check")
        if check is None:
            continue
        assert set(check) == {"oracle", "max_abs_diff", "atol"}, model
        assert check["oracle"] == f"reports/inputs/{model}.npy", model
        assert check["max_abs_diff"] <= check["atol"], model
        note = pre.get("verified", "")
        assert "byte-exact" not in note, model
        if check["max_abs_diff"] == 0:
            assert "exact against" in note, model
        else:
            assert "within 1e-4" in note, model


def test_unverified_models_say_so_rather_than_staying_silent():
    for model, s in specs().items():
        note = s.get("preprocess", {}).get("verified", "")
        assert note, f"{model} declares preprocessing without saying whether it is checked"


def test_the_normalisation_matches_what_the_script_does():
    """A spot check with teeth: models declared `over_255` must not subtract a mean.

    zipdepth is the reason this exists. It is the only model here whose
    preprocessing is /255 and nothing else, and applying ImageNet statistics to
    it would not crash -- it would score badly and look like a weak model.
    """
    for model, s in specs().items():
        if s["preprocess"]["normalise"] != "over_255":
            continue
        path = os.path.join(ROOT, "models", model, "onnx2trt.py")
        if not os.path.exists(path):
            continue
        body = open(path, encoding="utf-8").read()
        assert "0.485" not in body, (
            f"{model} declares over_255 but its onnx2trt.py mentions the "
            f"ImageNet mean")
