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
    "tanksandtemples": [
        "train",
        "truck"
    ]
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
    parser.add_argument(
        "--param",
        type=str,
        required=True,
        help="the param to include in the table",
    )
    args = parser.parse_args()

    dataset_dir = Path(f"./nerfbaselines_results/{args.dataset}/")
    param_to_param_path = {
        "PSNR": "eval-all-test/psnr",
        "SSIM": "eval-all-test/ssim",
        "LPIPS": "eval-all-test/lpips",
        "LPIPS(VGG)": "eval-all-test/lpips_vgg",
        "Num Gaussians": "train/num-gaussians",
    }

    presets = None

    column_by_scene = {}
    for scene_name in SCENES[args.dataset]:
        scene_dir = dataset_dir / scene_name
        if not scene_dir.is_dir():
            continue

        vals_by_preset = {}

        for preset_dir in scene_dir.iterdir():
            if not preset_dir.is_dir():
                continue
            if "noise" not in preset_dir.name:
                continue
            if "depth_downsample_40" in preset_dir.name:
                continue
            param = {args.param: param_to_param_path[args.param]}
            param_val = get_params(preset_dir, args.step, param)[args.param]
            vals_by_preset[preset_dir.name] = param_val

        # sort by preset name
        vals_by_preset = dict(sorted(vals_by_preset.items()))
        # put sfm first
        try:
            sfm_val = vals_by_preset.pop("sfm")
            vals_by_preset = {"sfm": sfm_val, **vals_by_preset}
        except KeyError:
            pass

        if presets is None:
            presets = list(vals_by_preset.keys())

        column_by_scene[scene_dir.name] = vals_by_preset.values()

    first_column = [make_pretty_preset_name(preset) for preset in presets]
    header = ["Preset"] + list(column_by_scene.keys())
    table = [list(row) for row in zip(first_column, *column_by_scene.values())]

    # for Num Gaussians, format as integer in thousands
    if args.param == "Num Gaussians":
        for row in table:
            row[1:] = [f"{int(float(val) / 1000):,}K" for val in row[1:]]

    # highlight the best value in each column
    for col_ix in range(1, len(header)):
        if "LPIPS" in args.param or args.param == "Num Gaussians":
            best_val = min([row[col_ix] for row in table])
        else:
            best_val = max([row[col_ix] for row in table])
        for row in table:
            best = False
            if row[col_ix] == best_val:
                best = True

            # 3 decimal places for floats
            if isinstance(row[col_ix], float):
                row[col_ix] = f"{row[col_ix]:.3f}"
            if best and not args.param == "Num Gaussians":
                row[col_ix] = f"\\textbf{{{row[col_ix]}}}"

    print(tabulate(table, headers=header, tablefmt="latex_raw"))


if __name__ == "__main__":
    main()
