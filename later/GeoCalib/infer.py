# by yhpark 2025-8-14
from GeoCalib.geocalib import GeoCalib, viz2d
from GeoCalib.geocalib.utils import print_calibration

import matplotlib.pyplot as plt
import torch 
import os

from onnx_export import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    """Convert radians to degrees."""
    return rad / torch.pi * 180

def main():
    model = GeoCalib().to(DEVICE)

    # load image as tensor in range [0, 1] with shape [C, H, W]
    image = model.load_image(f"{CUR_DIR}/../data/example.jpg").to(DEVICE)
    results = model.calibrate(image)

    print("camera:", results["camera"])
    print("gravity:", results["gravity"])

    print_calibration(results)

    model2 = GeoCalibModelWrapper().to(DEVICE)
    vfov, hfov, focal, roll, pitch = model2(image)
    vfov = rad2deg(vfov)
    roll = rad2deg(roll)
    pitch = rad2deg(pitch)
    print("\nEstimated parameters (Pred):")
    print(f"Roll:  {roll.item():.1f}° (± {rad2deg(results['roll_uncertainty']).item():.1f})°")
    print(f"Pitch: {pitch.item():.1f}° (± {rad2deg(results['pitch_uncertainty']).item():.1f})°")
    print(f"vFoV:  {vfov.item():.1f}° (± {rad2deg(results['vfov_uncertainty']).item():.1f})°")
    print(f"Focal: {focal.item():.1f} px (± {results['focal_uncertainty'].item():.1f} px)")

    # show images
    fig = viz2d.plot_images([image.permute(1, 2, 0).cpu().numpy()] * 3)
    ax = fig.get_axes()
    viz2d.plot_perspective_fields([results["camera"][0]], [results["gravity"][0]], axes=[ax[0]])
    viz2d.plot_confidences([results[f"{k}_confidence"][0] for k in ["up", "latitude"]], axes=ax[1:])
    plt.show()

if __name__ == "__main__":
    main()