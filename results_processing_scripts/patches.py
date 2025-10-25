import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, List
from tqdm import tqdm

from parameters import (
    NerfbaselinesJSONParameter,
    ParamOrdering,
    Parameter,
    ParameterInstance,
    TensorboardParameter,
)
from tabulate import tabulate

from results_processing_scripts.common import PARAMS, SCENES, PresetFilter, Table, output_table


class DataLoaderPatches:
    def __init__(
        self,
        dataset_dir: Path,
        scenes: list[str],
        param_names: list[str],
        step: int,
        preset_filter: PresetFilter,
    ):
        logging.debug(f"Loading per-patch data at step {step} for:")
        logging.debug(f"Scenes: {scenes}")
        logging.debug(f"Params: {param_names}")

        retval = {}  # {scene: preset: param: [[patch_value, ...], ...]}
        #                                       ^ list of patch values for each image
        all_presets = set()

        try:
            params = [PARAMS[param.lower()] for param in param_names]
        except KeyError as e:
            raise ValueError(f"Invalid parameter {e}")
        scene_pbar = tqdm(scenes, leave=False)
        for scene_name in scene_pbar:
            scene_pbar.set_description(f"Processing {scene_name}")
            scene_dir = dataset_dir / scene_name
            if not scene_dir.is_dir():
                continue

            presets_for_scene = {}

            preset_dirs = [
                preset_dir
                for preset_dir in scene_dir.iterdir()
                if preset_dir.is_dir() and preset_filter.allows(preset_dir.name)
            ]

            preset_pbar = tqdm(preset_dirs, leave=False)
            for preset_dir in preset_pbar:
                preset_pbar.set_description(f"Processing {preset_dir.name}")

                logging.debug(f"Processing {preset_dir}")

                param_pbar = tqdm(params, leave=False)
                params_for_preset = {}
                for param in param_pbar:
                    try:
                        params_for_preset[param.name] = param.load_patches(
                            preset_dir, step)
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
    def scenes(self) -> List[str]:
        return list(self.data.keys())

    def get(
        self, scene: str, preset: str, param: str | Parameter
    ) -> list[list[ParameterInstance]]:
        if isinstance(param, str):
            param = PARAMS[param]
        return self.data[scene][preset][param.name]

    def try_get(
        self, scene: str, preset: str, param: str | Parameter
    ) -> list[list[ParameterInstance]]:
        try:
            return self.get(scene, preset, param)
        except KeyError:
            return None


class MakeTableFuncs:
    @staticmethod
    def patches_binned_improvement(data_loader: DataLoaderPatches, args) -> list[Table]:
        retval = []

        def get_single_preset(filter, id):
            presets = [
                p for p in data_loader.presets if filter.allows(p)]

            if len(presets) != 1:
                raise RuntimeError(
                    f"Preset filter {id} macthed {len(presets)} presets, must be 1.")

            return presets[0]

        preset_a = get_single_preset(PresetFilter(
            args.preset_regex_a, args.preset_exclude_a), "A")
        preset_b = get_single_preset(PresetFilter(
            args.preset_regex_b, args.preset_exclude_b), "B")

        def make_bin_name(bin_ix):
            return f"[{args.bin_size * bin_ix}, {args.bin_size * (bin_ix+1)})"

        def make_table(scene: str, param: Parameter):
            try:
                patches_per_image_a = data_loader.get(scene, preset_a, param)
                patches_per_image_b = data_loader.get(scene, preset_b, param)
            except KeyError:
                return None

            num_sfm_points_per_image_a = data_loader.get(
                scene, preset_a, PARAMS["num_sfm_points"])
            assert num_sfm_points_per_image_a == data_loader.get(
                scene, preset_b, PARAMS["num_sfm_points"])

            bins_sum_count: dict[int, list[float, int]] = {}

            for patches_a, patches_b, patches_num_sfm_points in zip(patches_per_image_a, patches_per_image_b, num_sfm_points_per_image_a):
                for patch_a, patch_b, num_sfm_points in zip(patches_a, patches_b, patches_num_sfm_points):
                    delta = patch_b.value - patch_a.value
                    bin_id: int = num_sfm_points.value // args.bin_size
                    if bin_id not in bins_sum_count:
                        bins_sum_count[bin_id] = [delta, 1]
                    else:
                        bins_sum_count[bin_id][0] += delta
                        bins_sum_count[bin_id][1] += 1

            bins = {
                bin_id: sum / count for bin_id, (sum, count) in bins_sum_count.items()}

            min_bin_id, max_bin_id = min(bins.keys()), max(bins.keys())
            header_row = [f"{scene} / {param.name}"] + \
                [make_bin_name(i) for i in range(min_bin_id, max_bin_id+1)]

            values_row = [""]
            for bin_id in range(min_bin_id, max_bin_id + 1):
                if bin_id in bins:
                    values_row.append(param.make_instance(
                        bins[bin_id]).get_formatted_value())
                else:
                    values_row.append("-")

            return [header_row, values_row]

        for scene in data_loader.scenes:
            for param in data_loader.params:
                if param.name == PARAMS["num_sfm_points"].name:
                    continue
                table = make_table(scene, param)
                if table is None:
                    continue
                retval.append(table)
        return retval


def regex_union(regex_a: str | None, regex_b: str | None):
    if regex_a is None or regex_b is None:
        not_none = regex_a or regex_b
        return not_none  # not_none may be None if both are None
    return f"{regex_a}|{regex_b}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--scenes", nargs="+", default=None, help="Scenes to include in the table."
    )
    parser.add_argument(
        "--scenes-exclude",
        nargs="+",
        default=None,
        help="Scenes to exclude from the table.",
    )
    parser.add_argument("--results-dir", type=str,
                        default="./nerfbaselines_results")
    parser.add_argument(
        "--step",
        type=int,
        default=30000,
        help="Step number to extract parameters from.",
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

    patches_binned_impr = subparsers.add_parser("patches_binned_improvement")
    patches_binned_impr.add_argument(
        "--bin-size", type=int, default=100, help="Bin size for binning num sfm points")
    patches_binned_impr.add_argument("--preset-regex-a", type=str, default="init-type=sfm",
                                     help="Regex for the first preset, improvement for preset B will be computed relative to this. Must select single preset for each scene.")
    patches_binned_impr.add_argument("--preset-exclude-a", type=str, default=None,
                                     help="Regex for the first preset, improvement for preset B will be computed relative to this. Must select single preset for each scene.")
    patches_binned_impr.add_argument("--preset-regex-b", type=str, default=None,
                                     help="Regex for the second preset, improvement for this preset will be computed relative to preset A. Must select single preset for each scene.")
    patches_binned_impr.add_argument("--preset-exclude-b", type=str, default=None,
                                     help="Regex for the second preset, improvement for this preset will be computed relative to preset A. Must select single preset for each scene.")
    patches_binned_impr.add_argument(
        "params", nargs="+", choices=PARAMS.keys())

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    dataset_dir: Path = Path(args.results_dir) / args.dataset
    if not dataset_dir.is_dir():
        raise ValueError(
            f"Dataset directory {dataset_dir.absolute()} not found.")

    if "num_sfm_points" not in args.params:
        params = args.params + ["num_sfm_points"]

    if args.scenes is None:
        scenes = sorted(
            list(set(SCENES[args.dataset]).difference(
                args.scenes_exclude or []))
        )
    else:
        scenes = sorted(
            list(set(args.scenes).difference(args.scenes_exclude or [])))

    if args.subcommand == "patches_binned_improvement":
        preset_filter = PresetFilter(
            regex_union(args.preset_regex_a, args.preset_regex_b), None)
    else:
        preset_filter = PresetFilter(None, None)

    data_loader = DataLoaderPatches(
        dataset_dir,
        scenes,
        params,
        args.step,
        preset_filter
    )

    func = getattr(MakeTableFuncs, args.subcommand)
    tables = func(data_loader, args)

    for table in tables:
        output_table(args, table)


if __name__ == "__main__":
    main()
