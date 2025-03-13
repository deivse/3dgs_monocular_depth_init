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


# @dataclass
# class PresetInfo:
#     preset_name: str
#     pretty_name: str

#     @staticmethod
#     def from_preset_name(preset_name: str) -> "PresetInfo":
#         return PresetInfo(
#             preset_name=preset_name,
#             pretty_name=transform_preset_name_to_header(preset_name),
#         )

#     def __str__(self) -> str:
#         return self.pretty_name

SCENES = {
    "mipnerf360": [
        "garden",
        "bonsai",
        "stump",
        "flowers",
        "bicycle",
        "kitchen",
        "treehill",
        "room",
        "counter",
    ],
    "tanksandtemples": ["train", "truck"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Step number to extract parameters from.",
    )
    args = parser.parse_args()

    dataset_dir = Path(f"./nerfbaselines_results/{args.dataset}/")
    param_to_param_path = {
        "PSNR": "eval-all-test/psnr",
        "SSIM": "eval-all-test/ssim",
        "LPIPS": "eval-all-test/lpips",
        # "LPIPS(VGG)": "eval-all-test/lpips_vgg",
        "Num Gaussians": "train/num-gaussians",
    }

    presets = None

    sum_by_preset_by_param = {}
    for scene_name in SCENES[args.dataset]:
        scene_dir = dataset_dir / scene_name
        if not scene_dir.is_dir():
            continue

        for preset_dir in scene_dir.iterdir():
            if not preset_dir.is_dir():
                continue
            if "noise" in preset_dir.name:
                continue
            if "depth_downsample_40" in preset_dir.name:
                continue
            param_vals = get_params(preset_dir, args.step, param_to_param_path)

            for name, val in param_vals.items():
                for_preset = sum_by_preset_by_param.setdefault(preset_dir.name, {})
                for_param = for_preset.setdefault(name, [])
                for_param.append(val)

    for preset_name, param_vals in sum_by_preset_by_param.items():
        for param_name, vals in param_vals.items():
            sum_by_preset_by_param[preset_name][param_name] = sum(vals) / len(vals)

    # make table with pretty names in the first column and param names in the header
    table = []
    # sort by preset name, put sfm first
    sum_by_preset_by_param = dict(sorted(sum_by_preset_by_param.items()))
    sfm = sum_by_preset_by_param.pop("sfm")
    sum_by_preset_by_param = {"sfm": sfm, **sum_by_preset_by_param}

    for preset_name, param_vals in sum_by_preset_by_param.items():
        pretty_preset_name = make_pretty_preset_name(preset_name)
        table.append([pretty_preset_name] + list(param_vals.values()))

    headers = ["Preset"] + list(param_vals.keys())

    col_indices = {
        h: i for i, h in enumerate(headers) if h in param_to_param_path.keys()
    }
    psnr_max = max(row[col_indices["PSNR"]] for row in table)
    ssim_max = max(row[col_indices["SSIM"]] for row in table)
    lpips_min = min(row[col_indices["LPIPS"]] for row in table)

    def format_val(val, col_name):
        if col_name in ("PSNR", "SSIM", "LPIPS"):
            return f"{float(val):.3f}"
        if col_name == "Num Gaussians":
            return f"{int(val):,}"
        return val

    for row in table:
        if row[col_indices["PSNR"]] == psnr_max:
            row[col_indices["PSNR"]] = (
                f"\\textbf{{{format_val(row[col_indices['PSNR']], 'PSNR')}}}"
            )
        else:
            row[col_indices["PSNR"]] = format_val(row[col_indices["PSNR"]], "PSNR")
        if row[col_indices["SSIM"]] == ssim_max:
            row[col_indices["SSIM"]] = (
                f"\\textbf{{{format_val(row[col_indices['SSIM']], 'SSIM')}}}"
            )
        else:
            row[col_indices["SSIM"]] = format_val(row[col_indices["SSIM"]], "SSIM")
        if row[col_indices["LPIPS"]] == lpips_min:
            row[col_indices["LPIPS"]] = (
                f"\\textbf{{{format_val(row[col_indices['LPIPS']], 'LPIPS')}}}"
            )
        else:
            row[col_indices["LPIPS"]] = format_val(row[col_indices["LPIPS"]], "LPIPS")
        row[col_indices["Num Gaussians"]] = format_val(
            row[col_indices["Num Gaussians"]], "Num Gaussians"
        )

    # make latex table
    print(tabulate(table, headers=headers, tablefmt="latex_raw"))


if __name__ == "__main__":
    main()
