"""
Runs training and evaluation for multiple scenes and initialization strategies, reports results.
"""

from datetime import datetime
from pathlib import Path
import subprocess

import argparse

from itertools import product
import sys
from gs_init_compare.nerfbaselines_integration.make_presets import (
    ALL_NOISE_STD_SCENE_FRACTIONS,
    get_preset_names,
)

from nerfbaselines import get_dataset_spec
from tensorboard.backend.event_processing import event_accumulator


class ANSIEscapes:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END_SEQUENCE = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def by_name(name: str):
        return getattr(ANSIEscapes, name.upper())

    @staticmethod
    def color(text: str, color: str):
        return f"{ANSIEscapes.by_name(color)}{text}{ANSIEscapes.END_SEQUENCE}"


def rename_old_dir_with_timestamp(dir: Path, results_dir: Path) -> Path:
    """
    Appends a timestamp to the directory name to avoid conflicts
    when the directory already exists.

    `dir` is not modified, a new Path object is returned.
    """
    last_edit_time = max(f.stat().st_mtime for f in dir.rglob("*"))
    last_edit_time_str = datetime.fromtimestamp(last_edit_time).strftime(
        "_%d-%m-%Y_%H:%M:%S"
    )
    new_old_dir_name = dir.name + last_edit_time_str

    backup_results_dir_path = results_dir.parent / f"{results_dir.name}_backup"

    new_relative_path = dir.relative_to(results_dir)
    new_relative_path = new_relative_path.parent / new_old_dir_name

    new_path = backup_results_dir_path / new_relative_path
    new_path.parent.mkdir(parents=True, exist_ok=True)

    # This doesn't point dir to the new location
    # (Which is what we want)
    return dir.rename(backup_results_dir_path / new_relative_path)


def directory_exists_and_has_files(dir: Path) -> bool:
    if not dir.exists():
        return False
    for d in dir.rglob("*"):
        if d.is_file():
            return True
    return False


def get_dataset_scenes(dataset_id: str, exclude_list) -> list[str]:
    scenes = get_dataset_spec(dataset_id)["metadata"]["scenes"]

    def excluded(scene_id):
        for block in exclude_list:
            if block in scene_id:
                return True
        return False

    return [
        f"{dataset_id}/{scene['id']}" for scene in scenes if not excluded(scene["id"])
    ]


# ALL_SCENES = [
#     *get_dataset_scenes("mipnerf360", []),
#     *get_dataset_scenes("tanksandtemples", []),
# ]

# print(ALL_SCENES)

ALL_SCENES = [
    "mipnerf360/garden",  # PSNR worse, others better
    "mipnerf360/bonsai",  # everything worse
    "mipnerf360/stump",  # everything better
    "mipnerf360/flowers",  # PSNR worse, others better
    "mipnerf360/bicycle",  # everything better
    "mipnerf360/kitchen",  # everything worse
    "mipnerf360/treehill",  # PSNR worse, others better
    "mipnerf360/room",  # everything better (but significantly more gaussians)
    "mipnerf360/counter",  # PSNR worse, others better
    "tanksandtemples/truck",  # PSNR worse, others better (only lpips) more gaussians
    "tanksandtemples/train",  # PSNR worse, others better (only lpips) less gaussians
]


def create_argument_parser():
    parser = argparse.ArgumentParser()

    def add_argument(*args, **kwargs):
        if "default" in kwargs:
            kwargs["help"] = f"(={kwargs['default']})\n{kwargs.get('help', '')}"
        parser.add_argument(*args, **kwargs)

    add_argument(
        "--presets",
        nargs="+",
        default=get_preset_names([None]),
        help="Presets to pass to the method.",
    )
    add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default="nerfbaselines_results",
        help="Output directory. Subdirectories will be created for each dataset and preset.",
    )
    add_argument(
        "--max-steps",
        type=int,
        default=30_000,
        help="Maximum number of steps to run training for.",
    )
    add_argument(
        "--scenes",
        nargs="+",
        default=ALL_SCENES,
        help="Scenes to train and evaluate on. Scene names passed to nerfbaselines with 'external://' prefix.",
    )
    add_argument(
        "--invalidate-mono-depth-cache",
        action="store_true",
        default=False,
        help="Invalidate the cache for monocular depth predictors",
    )
    add_argument(
        "--eval-frequency",
        type=int,
        default=2000,
        help="Evaluate all images every N steps.",
    )
    add_argument(
        "--run-label",
        type=str,
        default=None,
        help="A custom label to be added to the preset directories for this run.",
    )
    return parser


def make_method_config_overrides(args: argparse.Namespace) -> dict[str, str]:
    return {
        "max_steps": str(args.max_steps),
        "mdi.ignore_cache": str(args.invalidate_mono_depth_cache),
        "mdi.cache_dir": str(Path(args.output_dir, "__mono_depth_cache__").absolute()),
    }


def get_args_str(args: argparse.Namespace):
    args_copy = argparse.Namespace()
    args_copy.__dict__ = args.__dict__.copy()

    unhashed_params = [
        "eval_frequency",
        "output_dir",
        "scenes",
        "presets",
        "invalidate_mono_depth_cache",
    ]
    for param in unhashed_params:
        delattr(args_copy, param)

    return str(args_copy)


ARGS_STR_FILENAME = ".nerfbaselines_evaluator_args_hash"


def output_dir_needs_overwrite(
    output_dir: Path,
    args: argparse.Namespace,
    args_str: str,
    eval_all_iters: list[int],
) -> bool:
    if not directory_exists_and_has_files(output_dir):
        return False

    try:
        with open(output_dir / ARGS_STR_FILENAME, "r") as f:
            old_args_str = f.read().strip()
    except FileNotFoundError:
        return True

    for iter in eval_all_iters:
        if iter == 0:
            continue  # nerfbaselines never evals at 0

        if not (output_dir / f"results-{str(iter)}.json").exists():
            return True

    return old_args_str != args_str


def read_param_from_last_tensorboard_step(file, param_name):
    ea = event_accumulator.EventAccumulator(
        str(file),
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 1,
            event_accumulator.IMAGES: 1,
            event_accumulator.AUDIO: 1,
            event_accumulator.SCALARS: 1,
            event_accumulator.HISTOGRAMS: 1,
            event_accumulator.TENSORS: 1,
        },
    )
    ea.Reload()
    if param_name not in ea.Tags().get("scalars", []):
        raise ValueError(f"Parameter {param_name} not found in TensorBoard logs.")

    scalars = ea.Scalars(param_name)
    if not scalars:
        raise ValueError(f"No scalar data found for parameter {param_name}.")

    return scalars[-1].value


def get_final_num_gaussians(output_dir: Path) -> int:
    tensorboard_file = next((output_dir / "tensorboard").glob("events.out.tfevents.*"))
    value = read_param_from_last_tensorboard_step(
        tensorboard_file, "train/num-gaussians"
    )
    print(f"Final number of gaussians for sfm: {value}")
    return int(value)


def run_combination(scene, preset, args, args_str, eval_all_iters):
    print(
        ANSIEscapes.color("_" * 80, "bold"),
        ANSIEscapes.color("=" * 80 + "\n", "blue"),
        sep="\n",
    )
    curr_output_dir = Path(args.output_dir / scene / preset)
    if args.run_label:
        curr_output_dir = curr_output_dir.with_name(
            f"{curr_output_dir.name}_{args.run_label}"
        )

    if curr_output_dir.exists():
        if not curr_output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {curr_output_dir}")

        if not output_dir_needs_overwrite(
            curr_output_dir, args, args_str, eval_all_iters
        ):
            print(
                ANSIEscapes.color(
                    f"Skipping {preset} on {scene}. (Output exists and is up-to-date)",
                    "green",
                )
            )
            return

        new_path = rename_old_dir_with_timestamp(curr_output_dir, args.output_dir)
        print(
            ANSIEscapes.color(
                f"Detected results mismatch. Old output directory moved to: {new_path}",
                "yellow",
            )
        )
        assert not curr_output_dir.exists()

    print(
        ANSIEscapes.color(
            f"Training {preset} on {scene}. (Outputting to: {curr_output_dir})",
            "blue",
        )
    )
    curr_output_dir.mkdir(parents=True, exist_ok=True)
    with open(curr_output_dir / ARGS_STR_FILENAME, "w") as f:
        f.write(args_str)

    overrides_cli = []
    for kv_pair in make_method_config_overrides(args).items():
        overrides_cli.extend(["--set", "=".join(kv_pair)])

    if "mcmc" in preset:
        overrides_cli.extend(
            [
                "--set",
                f"strategy.cap_max={get_final_num_gaussians(curr_output_dir.parent / 'sfm')}",
            ]
        )

    subprocess.run(
        [
            "nerfbaselines",
            "train",
            "--backend=python",
            "--method=gs-init-compare",
            f"--output={curr_output_dir}",
            f"--presets={preset}",
            f"--data=external://{scene}",
            f"--eval-all-iters={','.join(map(str, eval_all_iters))}",
        ]
        + overrides_cli
    )

    try:
        # shutil.rmtree(curr_output_dir / "checkpoint-30000")

        # Remove unnecessary outputs cuz I would run out of disk space...
        Path(curr_output_dir / "output.zip").unlink()

        # Delete predictions except last step and middle step:
        for iter in eval_all_iters:
            if iter not in [0, 8000, 14000, args.max_steps]:
                Path(curr_output_dir / f"predictions-{str(iter)}.tar.gz").unlink()
    except FileNotFoundError as e:
        print(ANSIEscapes.color(f"Error: Training output not found:\n {e}", "red"))


BESTISH_PRESET_PER_SCENE = {
    "mipnerf360/garden": "metric3d_depth_downsample_10",
    "mipnerf360/bonsai": "metric3d_depth_downsample_30",
    "mipnerf360/stump": "metric3d_depth_downsample_10",
    "mipnerf360/flowers": "unidepth_depth_downsample_10",
    "mipnerf360/bicycle": "unidepth_depth_downsample_10",
    "mipnerf360/kitchen": "depth_anything_v2_outdoor_depth_downsample_10",
    "mipnerf360/treehill": "metric3d_depth_downsample_10",
    "mipnerf360/room": "metric3d_depth_downsample_10",
    "mipnerf360/counter": "metric3d_depth_downsample_10",
    "tanksandtemples/truck": "metric3d_depth_downsample_10",
    "tanksandtemples/train": "depth_anything_v2_outdoor_depth_downsample_20",
}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = create_argument_parser().parse_args()

    eval_all_iters = list(range(0, args.max_steps + 1, args.eval_frequency))
    if eval_all_iters[-1] != args.max_steps:
        eval_all_iters.append(args.max_steps)

    args_str = get_args_str(args)

    combinations = list(product(args.scenes, args.presets))
    print(
        ANSIEscapes.color("_" * 80, "bold"),
        ANSIEscapes.color(f"Will train {len(combinations)} combinations.", "bold"),
        ANSIEscapes.color("Settings:", "bold"),
        f"\tOutput directory: {ANSIEscapes.color(args.output_dir, 'cyan')}",
        f"\tMax steps: {ANSIEscapes.color(args.max_steps, 'cyan')}",
        f"\tEvaluation frequency: {ANSIEscapes.color(args.eval_frequency, 'cyan')}",
        f"\tPresets: {ANSIEscapes.color(args.presets, 'cyan')}",
        f"\tScenes: {ANSIEscapes.color(args.scenes, 'cyan')}",
        f"\tEval all iters: {ANSIEscapes.color(eval_all_iters, 'cyan')}",
        sep="\n",
    )

    for scene, preset in combinations:
        run_combination(scene, preset, args, args_str, eval_all_iters)

    # for scene in args.scenes:
    #     preset = BESTISH_PRESET_PER_SCENE[scene]
    #     for noise_std_frac in ALL_NOISE_STD_SCENE_FRACTIONS:
    #         if noise_std_frac is None:
    #             continue
    #         run_combination(
    #             scene,
    #             f"{preset}_noise_{noise_std_frac}",
    #             args,
    #             args_str,
    #             eval_all_iters,
    #         )


if __name__ == "__main__":
    main()
