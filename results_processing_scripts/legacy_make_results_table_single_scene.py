from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tensorboard.backend.event_processing import event_accumulator
import argparse
import re
from tabulate import tabulate


class TensorboardDataLoader:
    def __init__(self, file):
        self.ea = event_accumulator.EventAccumulator(
            str(file),
            size_guidance={"scalars": 0},
        )
        self.ea.Reload()

    def read_param(self, param_name, step):
        if param_name not in self.ea.Tags().get("scalars", []):
            raise ValueError(f"Parameter {param_name} not found in TensorBoard logs.")

        scalars = self.ea.Scalars(param_name)
        if not scalars:
            raise ValueError(f"No scalar data found for parameter {param_name}.")

        for scalar in scalars:
            if scalar.step == step:
                return scalar.value

        raise ValueError(f"Step {step} not found for parameter {param_name}.")


def get_params(output_dir: Path, step: int, params: dict[str, str]) -> dict[str, Any]:
    tensorboard_file = next((output_dir / "tensorboard").glob("events.out.tfevents.*"))
    data_loader = TensorboardDataLoader(tensorboard_file)
    return {
        param_name: data_loader.read_param(param_path, step)
        for param_name, param_path in params.items()
    }


def make_pretty_preset_name(preset_name: str) -> str:
    preset_name = re.sub(r"depth_downsample_(\d+)", r"[\1]", preset_name)
    name = preset_name.replace("_", " ")
    substitutions = {
        "metric3d": "Metric3Dv2",
        "unidepth": "UniDepth",
        "depth anything v2": "DA V2",
        "moge": "MoGe",
    }
    explicitly_capitalized = []
    for sub in substitutions.values():
        explicitly_capitalized.extend(sub.split(" "))

    for key, value in substitutions.items():
        name = name.replace(key, value)
    name = " ".join(
        [
            word.capitalize() if word not in explicitly_capitalized else word
            for word in name.split(" ")
        ]
    )
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--scene", type=str, required=True, help="Scene name.")
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Step number to extract parameters from.",
    )
    args = parser.parse_args()

    dataset_dir = Path(f"./nerfbaselines_results/{args.dataset}/")
    scene_dir = dataset_dir / args.scene
    if not scene_dir.is_dir():
        raise ValueError(f"Scene directory {scene_dir} not found.")

    param_to_param_path = {
        "PSNR": "eval-all-test/psnr",
        "SSIM": "eval-all-test/ssim",
        "LPIPS": "eval-all-test/lpips",
        "Num Gaussians": "train/num-gaussians",
    }

    results_by_preset = {}
    for preset_dir in scene_dir.iterdir():
        if not preset_dir.is_dir():
            continue
        if "noise" in preset_dir.name or "depth_downsample_40" in preset_dir.name:
            continue
        param_vals = get_params(preset_dir, args.step, param_to_param_path)
        results_by_preset[preset_dir.name] = param_vals

    # Build the table (one row per preset)
    headers = ["Preset"] + list(param_to_param_path.keys())
    table = []

    # Find min/max across presets for highlight
    psnr_vals = [res["PSNR"] for res in results_by_preset.values()]
    ssim_vals = [res["SSIM"] for res in results_by_preset.values()]
    lpips_vals = [res["LPIPS"] for res in results_by_preset.values()]

    psnr_max = max(psnr_vals)
    ssim_max = max(ssim_vals)
    lpips_min = min(lpips_vals)

    def format_val(val, col_name):
        if col_name in ("PSNR", "SSIM", "LPIPS"):
            return f"{float(val):.3f}"
        if col_name == "Num Gaussians":
            return f"{int(val):,}"
        return val

    for preset_name, param_vals in sorted(results_by_preset.items()):
        row = [make_pretty_preset_name(preset_name)]
        for key in param_to_param_path:
            val = param_vals[key]
            if key == "PSNR" and val == psnr_max:
                row.append(f"\\textbf{{{format_val(val, key)}}}")
            elif key == "SSIM" and val == ssim_max:
                row.append(f"\\textbf{{{format_val(val, key)}}}")
            elif key == "LPIPS" and val == lpips_min:
                row.append(f"\\textbf{{{format_val(val, key)}}}")
            else:
                row.append(format_val(val, key))
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="latex_raw"))


if __name__ == "__main__":
    main()
