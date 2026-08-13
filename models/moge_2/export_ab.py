"""Export isolated MoGe graphs for exporter/simplifier A/B measurement.

The published ``moge-2_vits_normal_388x518_dynamo_sim.onnx`` is never touched.
This writes four explicitly experimental files instead:

* ``..._ab_dynamo.onnx``
* ``..._ab_dynamo_sim.onnx``
* ``..._ab_legacy.onnx``
* ``..._ab_legacy_sim.onnx``
"""

from onnx_export import export_variant


def main():
    common = dict(input_h=388, input_w=518, encoder="vits", normal=True,
                  onnx_sim=True)
    outputs = []
    outputs += export_variant(
        **common, dynamo=True,
        model_name="moge-2_vits_normal_388x518_ab_dynamo")
    outputs += export_variant(
        **common, dynamo=False,
        model_name="moge-2_vits_normal_388x518_ab_legacy")
    print("\n[MDET] A/B graphs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
