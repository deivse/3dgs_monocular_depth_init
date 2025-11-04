import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, List
from tqdm import tqdm
import numpy as np

from results_processing_scripts.parameters import (
    NerfbaselinesJSONParameter,
    ParamOrdering,
    Parameter,
    ParameterInstance,
    TensorboardParameter,
)
from tabulate import tabulate
import uuid, pickle

from results_processing_scripts.common import (
    PARAMS,
    SCENES,
    PresetFilter,
    Table,
    output_table,
)


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
                            preset_dir, step
                        )
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


def get_presets(args, data_loader: DataLoaderPatches):
    def get_single_preset(filter, id):
        presets = [p for p in data_loader.presets if filter.allows(p)]

        if len(presets) != 1:
            raise RuntimeError(
                f"Preset filter {id} macthed {len(presets)} presets, must be 1."
            )

        return presets[0]

    preset_a = get_single_preset(
        PresetFilter(args.preset_regex_a, args.preset_exclude_a), "A"
    )
    preset_b = get_single_preset(
        PresetFilter(args.preset_regex_b, args.preset_exclude_b), "B"
    )
    return preset_a, preset_b


def make_bin_name(bin_size, bin_ix):
    return f"[{bin_size * bin_ix}, {bin_size * (bin_ix + 1)})"


def accumulate_param_bins(
    patches_a: list[ParameterInstance],
    patches_b: list[ParameterInstance],
    num_sfm_points: list[ParameterInstance],
    bin_size: int,
    scene: str,
    param_name: str,
):
    bins_sum_count: dict[int, list[float, int]] = {}

    patches_a: np.ndarray = np.array([p.value for p in patches_a])
    patches_b: np.ndarray = np.array([p.value for p in patches_b])
    num_sfm_points: np.ndarray = np.array(num_sfm_points)

    valid_indices = np.isfinite(patches_a) & np.isfinite(patches_b)
    if not np.all(valid_indices):
        logging.warning(
            "Encountered invalid patch values for scene %s param %s", scene, param_name
        )

    patches_a = patches_a[valid_indices]
    patches_b = patches_b[valid_indices]

    deltas: np.ndarray = patches_b - patches_a
    for i in range(deltas.size):
        bin_id: int = num_sfm_points[i] // bin_size
        if bin_id not in bins_sum_count:
            bins_sum_count[bin_id] = [deltas[i], 1]
        else:
            prev_sum = bins_sum_count[bin_id][0]
            bins_sum_count[bin_id][0] += deltas[i]
            bins_sum_count[bin_id][1] += 1

        if bins_sum_count[bin_id][0] != bins_sum_count[bin_id][0]:
            raise RuntimeError(
                f"Sum mismatch in bin {bin_id}: {bins_sum_count[bin_id][0]} vs {prev_sum} + {deltas[i]}"
            )

    # check vals, nan is creeping in somehow
    for bin_id, (sum, count) in bins_sum_count.items():
        if count == 0:
            raise RuntimeError(f"Bin {bin_id} has zero count.")
        if sum != sum:  # NaN check
            raise RuntimeError(f"Bin {bin_id} has NaN sum.")

    return {bin_id: sum / count for bin_id, (sum, count) in bins_sum_count.items()}, {
        bin_id: count for bin_id, (_, count) in bins_sum_count.items()
    }


def patch_percentile_indices(patch_vals: np.ndarray, percentiles: list[float]):
    assert sorted(percentiles) == percentiles
    p_vals = np.percentile(patch_vals, percentiles)
    prev_p = np.NINF
    for p in p_vals:
        indices = np.argwhere((patch_vals > prev_p) & (patch_vals <= p))
        yield indices
        prev_p = p


class MakeTableFuncs:
    @staticmethod
    def patches_binned_improvement(data_loader: DataLoaderPatches, args) -> list[Table]:
        retval = []

        preset_a, preset_b = get_presets(args, data_loader)
        print(
            f"Computing improvement of preset B '{preset_b}' over preset A '{preset_a}'"
        )

        def make_table(scene: str, param: Parameter):
            try:
                patches_per_image_a = data_loader.get(scene, preset_a, param)
                patches_per_image_b = data_loader.get(scene, preset_b, param)
            except KeyError:
                return None

            num_sfm_points_per_image_a = data_loader.get(
                scene, preset_a, PARAMS["num_sfm_points"]
            )
            assert num_sfm_points_per_image_a == data_loader.get(
                scene, preset_b, PARAMS["num_sfm_points"]
            )
            patches_a = (p for patches in patches_per_image_a for p in patches)
            patches_b = (p for patches in patches_per_image_b for p in patches)
            num_sfm_points = (
                p for patches in num_sfm_points_per_image_a for p in patches
            )

            bins, _ = accumulate_param_bins(
                patches_a, patches_b, num_sfm_points, args.bin_size, scene, param.name
            )

            min_bin_id, max_bin_id = min(bins.keys()), max(bins.keys())
            header_row = [f"{scene} / {param.name}"] + [
                make_bin_name(args.bin_size, i)
                for i in range(min_bin_id, max_bin_id + 1)
            ]

            values_row = [""]
            for bin_id in range(min_bin_id, max_bin_id + 1):
                if bin_id in bins:
                    values_row.append(
                        param.make_instance(bins[bin_id]).get_formatted_value()
                    )
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

    @staticmethod
    def patches_binned_improvement_dataset_avg(
        data_loader: DataLoaderPatches, args
    ) -> list[Table]:
        preset_a, preset_b = get_presets(args, data_loader)
        print(
            f"Computing improvement of preset B '{preset_b}' over preset A '{preset_a}'"
        )

        def make_param_bins(param: Parameter):
            patches_a, patches_b, num_sfm_points = [], [], []
            for scene in data_loader.scenes:
                try:
                    patches_per_image_a = data_loader.get(scene, preset_a, param)
                    patches_per_image_b = data_loader.get(scene, preset_b, param)
                except KeyError:
                    raise RuntimeError(
                        f"Parameter {param.name} not found for scene {scene} and presets {preset_a}, {preset_b}, can not compute dataset average."
                    )
                num_sfm_points_per_image_a = data_loader.get(
                    scene, preset_a, PARAMS["num_sfm_points"]
                )
                assert num_sfm_points_per_image_a == data_loader.get(
                    scene, preset_b, PARAMS["num_sfm_points"]
                )

                patches_a.extend(p for patches in patches_per_image_a for p in patches)
                patches_b.extend(p for patches in patches_per_image_b for p in patches)
                num_sfm_points.extend(
                    p for patches in num_sfm_points_per_image_a for p in patches
                )

            return accumulate_param_bins(
                patches_a, patches_b, num_sfm_points, args.bin_size, scene, param.name
            )

        min_bin_id, max_bin_id = None, None

        combined_bins = {}

        for param in data_loader.params:
            if param.name == PARAMS["num_sfm_points"].name:
                continue
            param_bins, bin_counts = make_param_bins(param)
            if min_bin_id is None or max_bin_id is None:
                min_bin_id, max_bin_id = min(param_bins.keys()), max(param_bins.keys())
            else:
                min_bin_id = min(min_bin_id, min(param_bins.keys()))
                max_bin_id = max(max_bin_id, max(param_bins.keys()))

            for bin_id, value in param_bins.items():
                if bin_id not in combined_bins:
                    combined_bins[bin_id] = {}
                combined_bins[bin_id][param.name] = value
                combined_bins[bin_id]["sample_count"] = (
                    combined_bins[bin_id].get("sample_count", 0) + bin_counts[bin_id]
                )

        header_row = [f"{args.dataset}"] + [
            make_bin_name(args.bin_size, i) for i in range(min_bin_id, max_bin_id + 1)
        ]
        table = [header_row]
        for param in data_loader.params:
            if param.name == PARAMS["num_sfm_points"].name:
                continue
            row = [param.name]
            for bin_id in range(min_bin_id, max_bin_id + 1):
                if bin_id in combined_bins and param.name in combined_bins[bin_id]:
                    row.append(
                        param.make_instance(
                            combined_bins[bin_id][param.name]
                        ).get_formatted_value()
                    )
                else:
                    row.append("-")
            table.append(row)
        bin_row = ["Samples in Bin"]
        for bin_id in range(min_bin_id, max_bin_id + 1):
            bin_row.append(
                f"{combined_bins[bin_id]['sample_count']}"
                if bin_id in combined_bins
                else "0"
            )
        table.append(bin_row)

        return [table]

    @staticmethod
    def patches_improvement_percentile(
        data_loader: DataLoaderPatches, args
    ) -> list[Table]:
        # TODO: adjust to be per-scene
        preset_a, preset_b = get_presets(args, data_loader)
        print(
            f"Computing improvement of preset B '{preset_b}' over preset A '{preset_a}'"
        )

        def table_for_param(param: Parameter):
            patches_a = []
            patches_b = []
            for scene in data_loader.scenes:
                try:
                    patches_per_image_a = data_loader.get(scene, preset_a, param)
                    patches_per_image_b = data_loader.get(scene, preset_b, param)
                except KeyError:
                    raise RuntimeError(
                        f"Parameter {param.name} not found for scene {scene} and presets {preset_a}, {preset_b}, can not compute dataset average."
                    )

                patches_a.extend(p for patches in patches_per_image_a for p in patches)
                patches_b.extend(p for patches in patches_per_image_b for p in patches)

            patch_vals_a = np.array([p.value for p in patches_a])
            patch_vals_b = np.array([p.value for p in patches_b])

            valid_indices = np.isfinite(patch_vals_a) & np.isfinite(patch_vals_b)
            if not np.all(valid_indices):
                logging.warning(
                    "Encountered invalid patch values for scene %s param %s",
                    scene,
                    param.name,
                )

            patch_vals_a = patch_vals_a[valid_indices]
            patch_vals_b = patch_vals_b[valid_indices]

            percentiles = list(range(10, 101, 10))
            percentile_indices = patch_percentile_indices(
                patch_vals_a
                if param.ordering == ParamOrdering.HIGHER_IS_BETTER
                else (patch_vals_a * -1),
                percentiles,
            )

            header_row = [f"{param.name} Percentile in A"] + [
                f"{p}%" for p in percentiles
            ]
            values_row = ["Improvement"]
            for indices in percentile_indices:
                improvements = patch_vals_b[indices] - patch_vals_a[indices]
                values_row.append(
                    f"{param.make_instance(np.mean(improvements)).get_formatted_value()}"
                )
            return [header_row, values_row]

        return [
            table_for_param(param)
            for param in data_loader.params
            if param.name != PARAMS["num_sfm_points"].name
        ]

    @staticmethod
    def patches_improvement_percentile_dataset_avg(
        data_loader: DataLoaderPatches, args
    ) -> list[Table]:
        preset_a, preset_b = get_presets(args, data_loader)
        print(
            f"Computing improvement of preset B '{preset_b}' over preset A '{preset_a}'"
        )

        percentiles = args.percentiles

        def table_for_param(param: Parameter):
            percentile_improvements = [np.empty([0]) for _ in range(len(percentiles))]
            for scene in data_loader.scenes:
                try:
                    patches_per_image_a = data_loader.get(scene, preset_a, param)
                    patches_per_image_b = data_loader.get(scene, preset_b, param)
                except KeyError:
                    raise RuntimeError(
                        f"Parameter {param.name} not found for scene {scene} and presets {preset_a}, {preset_b}, can not compute dataset average."
                    )

                patch_vals_a = np.array(
                    [p.value for patches in patches_per_image_a for p in patches]
                )
                patch_vals_b = np.array(
                    [p.value for patches in patches_per_image_b for p in patches]
                )
                valid_indices = np.isfinite(patch_vals_a) & np.isfinite(patch_vals_b)
                if not np.all(valid_indices):
                    logging.warning(
                        "Encountered invalid patch values for scene %s param %s",
                        scene,
                        param.name,
                    )

                patch_vals_a = patch_vals_a[valid_indices]
                patch_vals_b = patch_vals_b[valid_indices]

                m = (
                    -1
                    if (param.ordering.value == ParamOrdering.LOWER_IS_BETTER.value)
                    else 1
                )
                percentile_indices = patch_percentile_indices(
                    patch_vals_a * m, percentiles
                )
                for i, indices in enumerate(percentile_indices):
                    improvements = patch_vals_b[indices] - patch_vals_a[indices]
                    percentile_improvements[i] = np.hstack(
                        (percentile_improvements[i], improvements.flatten())
                    )

            header_row = [f"{param.name} Percentile in A"] + [
                f"{p}%" for p in percentiles
            ]
            values_row = ["Improvement"]
            for improvements in percentile_improvements:
                values_row.append(
                    f"{param.make_instance(np.mean(improvements)).get_formatted_value()}"
                )
            return [header_row, values_row]

        return [
            table_for_param(param)
            for param in data_loader.params
            if param.name != PARAMS["num_sfm_points"].name
        ]


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
    parser.add_argument("--results-dir", type=str, default="./nerfbaselines_results")
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
        "--bin-size", type=int, default=100, help="Bin size for binning num sfm points"
    )
    patches_binned_impr.add_argument(
        "--preset-regex-a",
        type=str,
        default="init-type=sfm",
        help="Regex for the first preset, improvement for preset B will be computed relative to this. Must select single preset for each scene.",
    )
    patches_binned_impr.add_argument(
        "--preset-exclude-a",
        type=str,
        default=None,
        help="Regex for the first preset, improvement for preset B will be computed relative to this. Must select single preset for each scene.",
    )
    patches_binned_impr.add_argument(
        "--preset-regex-b",
        type=str,
        default=None,
        help="Regex for the second preset, improvement for this preset will be computed relative to preset A. Must select single preset for each scene.",
    )
    patches_binned_impr.add_argument(
        "--preset-exclude-b",
        type=str,
        default=None,
        help="Regex for the second preset, improvement for this preset will be computed relative to preset A. Must select single preset for each scene.",
    )
    patches_binned_impr.add_argument("params", nargs="+", choices=PARAMS.keys())

    patches_binned_impr_da = subparsers.add_parser(
        "patches_binned_improvement_dataset_avg"
    )
    patches_binned_impr_da.add_argument(
        "--bin-size", type=int, default=100, help="Bin size for binning num sfm points"
    )
    patches_binned_impr_da.add_argument(
        "--preset-regex-a",
        type=str,
        default="init-type=sfm",
        help="Regex for the first preset, improvement for preset B will be computed relative to this. Must select single preset for each scene.",
    )
    patches_binned_impr_da.add_argument(
        "--preset-exclude-a",
        type=str,
        default=None,
        help="Regex for the first preset, improvement for preset B will be computed relative to this. Must select single preset for each scene.",
    )
    patches_binned_impr_da.add_argument(
        "--preset-regex-b",
        type=str,
        default=None,
        help="Regex for the second preset, improvement for this preset will be computed relative to preset A. Must select single preset for each scene.",
    )
    patches_binned_impr_da.add_argument(
        "--preset-exclude-b",
        type=str,
        default=None,
        help="Regex for the second preset, improvement for this preset will be computed relative to preset A. Must select single preset for each scene.",
    )
    patches_binned_impr_da.add_argument("params", nargs="+", choices=PARAMS.keys())

    patches_impr_perc_da = subparsers.add_parser(
        "patches_improvement_percentile_dataset_avg"
    )
    patches_impr_perc_da.add_argument(
        "--preset-regex-a",
        type=str,
        default="init-type=sfm",
        help="Regex for the first preset, improvement for preset B will be computed relative to this. Must select single preset for each scene.",
    )
    patches_impr_perc_da.add_argument(
        "--preset-exclude-a",
        type=str,
        default=None,
        help="Regex for the first preset, improvement for preset B will be computed relative to this. Must select single preset for each scene.",
    )
    patches_impr_perc_da.add_argument(
        "--preset-regex-b",
        type=str,
        default=None,
        help="Regex for the second preset, improvement for this preset will be computed relative to preset A. Must select single preset for each scene.",
    )
    patches_impr_perc_da.add_argument(
        "--preset-exclude-b",
        type=str,
        default=None,
        help="Regex for the second preset, improvement for this preset will be computed relative to preset A. Must select single preset for each scene.",
    )
    patches_impr_perc_da.add_argument("params", nargs="+", choices=PARAMS.keys())
    patches_impr_perc_da.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=list(range(25, 101, 25)),
        help="Percentiles to compute.",
    )

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    dataset_dir: Path = Path(args.results_dir) / args.dataset
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory {dataset_dir.absolute()} not found.")

    if "num_sfm_points" not in args.params:
        params = args.params + ["num_sfm_points"]

    if args.scenes is None:
        scenes = sorted(
            list(set(SCENES[args.dataset]).difference(args.scenes_exclude or []))
        )
    else:
        if args.subcommand == "patches_binned_improvement_dataset_avg":
            logging.warning(
                "Only a subset of scenes is selected for dataset average computation."
            )
        scenes = sorted(list(set(args.scenes).difference(args.scenes_exclude or [])))

    preset_filter = PresetFilter(
        regex_union(args.preset_regex_a, args.preset_regex_b), None
    )

    data_loader = DataLoaderPatches(
        dataset_dir, scenes, params, args.step, preset_filter
    )

    func = getattr(MakeTableFuncs, args.subcommand)
    tables = func(data_loader, args)

    for table in tables:
        output_table(args, table)


if __name__ == "__main__":
    main()
