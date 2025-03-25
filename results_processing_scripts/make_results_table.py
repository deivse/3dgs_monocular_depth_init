import argparse
import logging
import re
from pathlib import Path
from typing import List

from parameters import (
    NerfbaselinesJSONParameter,
    ParamOrdering,
    Parameter,
    ParameterInstance,
    TensorboardParameter,
)
from tabulate import tabulate


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


PARAMS: dict[str, Parameter] = {
    "psnr": NerfbaselinesJSONParameter(
        name="PSNR", json_path="metrics.psnr", ordering=ParamOrdering.HIGHER_IS_BETTER
    ),
    "ssim": NerfbaselinesJSONParameter(
        name="SSIM", json_path="metrics.ssim", ordering=ParamOrdering.HIGHER_IS_BETTER
    ),
    "lpips": NerfbaselinesJSONParameter(
        name="LPIPS", json_path="metrics.lpips", ordering=ParamOrdering.LOWER_IS_BETTER
    ),
    "lpips_vgg": NerfbaselinesJSONParameter(
        name="LPIPS(VGG)",
        json_path="metrics.lpips_vgg",
        ordering=ParamOrdering.LOWER_IS_BETTER,
    ),
    "num_gaussians": TensorboardParameter(
        name="Num Gaussians",
        tensorboard_id="train/num-gaussians",
        formatter=lambda val: f"{int(float(val) / 1000):,}K",
        ordering=ParamOrdering.LOWER_IS_BETTER,
        should_highlight_best=False,
    ),
}

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


class PresetFilter:
    def __init__(self, in_regex: str | None, out_regex: str | None):
        self.in_regex = re.compile(in_regex) if in_regex else None
        self.out_regex = re.compile(out_regex) if out_regex else None

    def allows(self, preset_name: str) -> bool:
        if self.in_regex and not self.in_regex.search(preset_name):
            return False
        if self.out_regex and self.out_regex.search(preset_name):
            return False
        return True


class DataLoader:
    def __init__(
        self,
        dataset_dir: Path,
        scenes: list[str],
        param_names: list[str],
        step: int,
        preset_filter: PresetFilter,
    ):
        logging.debug(f"Loading data at step {step} for:")
        logging.debug(f"Scenes: {scenes}")
        logging.debug(f"Params: {param_names}")

        retval = {}  # {scene: param: value}
        all_presets = set()

        try:
            params = [PARAMS[param.lower()] for param in param_names]
        except KeyError as e:
            raise ValueError(f"Invalid parameter {e}")
        for scene_name in scenes:
            scene_dir = dataset_dir / scene_name
            if not scene_dir.is_dir():
                continue

            presets_for_scene = {}

            for preset_dir in scene_dir.iterdir():
                if not preset_filter.allows(preset_dir.name):
                    continue
                if not preset_dir.is_dir():
                    continue

                params_for_preset = {}

                logging.debug(f"Processing {preset_dir}")

                for param in params:
                    try:
                        params_for_preset[param.name] = param.load(preset_dir, step)
                    except Exception as e:
                        logging.error(
                            f"Error loading {param.name} for {preset_dir}: {e}"
                        )
                        continue

                presets_for_scene[preset_dir.name] = params_for_preset
                all_presets.add(preset_dir.name)
            retval[scene_dir.name] = presets_for_scene

        self.data = retval
        self.params = params
        self.presets = []
        if "sfm" in all_presets:
            self.presets.append("sfm")
            all_presets.remove("sfm")
        self.presets.extend(sorted(all_presets))

    @property
    def scenes(self):
        return self.data.keys()

    def try_get(self, scene, preset, param) -> ParameterInstance | None:
        try:
            if isinstance(param, str):
                param = PARAMS[param]
            return self.data[scene][preset][param.name]
        except KeyError:
            return None


def format_best(val, output_fmt):
    MARKDOWN_FORMATS = ["github", "grid", "pipe", "jira", "presto", "pretty", "rst"]
    if output_fmt in MARKDOWN_FORMATS:
        return f"***{val}***"
    if output_fmt == "latex":
        return f"\\textbf{{{val}}}"
    return f"*{val}"


class MakeTableFuncs:
    @staticmethod
    def single_param(data_loader: DataLoader, args):
        if args.param is None:
            raise ValueError("No parameter specified")

        first_column = ["Preset/Scene"] + [
            make_pretty_preset_name(preset) for preset in data_loader.presets
        ]
        table = [first_column]
        for scene in data_loader.scenes:
            params = [
                data_loader.try_get(scene, preset, args.param)
                for preset in data_loader.presets
            ]
            best_row_index = None

            for i, param in enumerate(params):
                if param is None:
                    continue
                if best_row_index is None or param > params[best_row_index]:
                    best_row_index = i

            formatted_params = []
            for i, param in enumerate(params):
                if param is None:
                    formatted_params.append("-")
                else:
                    formatted = param.get_formatted_value()
                    if i == best_row_index and param.should_highlight_best:
                        formatted = format_best(formatted, args.output_format)
                    formatted_params.append(formatted)

            table.append([scene] + formatted_params)

        return list(map(list, zip(*table)))

    @staticmethod
    def all_scene_avg(data_loader: DataLoader, args):
        """
        Average all parameters over all scenes for each preset.
        """
        table: List[List[ParameterInstance]] = []
        for preset in data_loader.presets:
            row = []
            for param in data_loader.params:
                average = param.make_instance(0.0)
                seen_count = 0.0
                for scene in data_loader.scenes:
                    instance = data_loader.try_get(scene, preset, param)
                    if instance is None:
                        continue
                    average.value += instance.value
                    seen_count += 1
                if seen_count == len(data_loader.scenes):
                    average.value /= seen_count
                    row.append(average)
                else:
                    row.append(None)
            table.append(row)

        best_row_per_param = []
        for param_ix in range(len(data_loader.params)):
            best_row = None
            for row_ix, row in enumerate(table):
                if row[param_ix] is not None and (
                    best_row is None
                    or row[param_ix].value > table[best_row][param_ix].value
                ):
                    best_row = row_ix
            best_row_per_param.append(best_row)

        first_row = ["Preset"] + [param.name for param in data_loader.params]
        formatted_table = [first_row]
        for (row_ix, row), preset in zip(enumerate(table), data_loader.presets):
            formatted_row = [make_pretty_preset_name(preset)]
            for param_ix, param in enumerate(row):
                if param is None:
                    formatted_row.append("-")
                    continue
                formatted = param.get_formatted_value()
                if (
                    param.should_highlight_best
                    and row_ix == best_row_per_param[param_ix]
                ):
                    formatted = format_best(formatted, args.output_format)
                formatted_row.append(formatted)
            formatted_table.append(formatted_row)

        return formatted_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=30000,
        help="Step number to extract parameters from.",
    )
    parser.add_argument(
        "--preset-regex",
        type=str,
        default=None,
        help="Filter presets using this regex.",
    )
    parser.add_argument(
        "--preset-exclude",
        type=str,
        default=None,
        help="Exclude presets using this regex.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="latex",
        help="Output format for the table (all formats supported by tabulate package).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file to write the table to.",
    )
    parser.add_argument("--debug", action="store_true")

    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    single_param_parser = subparsers.add_parser(
        "single_param",
        help="Create a table for a single parameter.",
    )
    single_param_parser.add_argument(
        "param",
        type=str,
        help="the param to include in the table",
        choices=PARAMS.keys(),
    )
    all_scene_average = subparsers.add_parser("all_scene_avg",)
    all_scene_average.add_argument("params", nargs="+", choices=PARAMS.keys())
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    dataset_dir = Path(f"./nerfbaselines_results/{args.dataset}/")

    if args.subcommand == "single_param":
        params = [args.param]
    else:
        params = args.params

    data_loader = DataLoader(
        dataset_dir,
        SCENES[args.dataset],
        params,
        args.step,
        PresetFilter(args.preset_regex, args.preset_exclude),
    )

    func = getattr(MakeTableFuncs, args.subcommand)
    table = func(data_loader, args)

    if args.output_format == "latex":
        args.output_format = "latex_raw"
    table_str = tabulate(table, headers="firstrow", tablefmt=args.output_format)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(table_str)
    else:
        print(table_str)


if __name__ == "__main__":
    main()
