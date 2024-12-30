"""
Runs training and evaluation for multiple scenes and initialization strategies, reports results.
"""

from datetime import datetime
from pathlib import Path
import shutil
import subprocess

import argparse

from itertools import product
from gs_init_compare.nerfbaselines_integration.make_presets import (
    PRESETS_DEPTH_DOWN_SAMPLE_FACTORS,
    all_preset_names,
)

from nerfbaselines import get_dataset_spec


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


def get_dataset_scenes(dataset_id: str) -> list[str]:
    scenes = get_dataset_spec(dataset_id)["metadata"]["scenes"]
    return [f"{dataset_id}/{scene['id']}" for scene in scenes]


ALL_SCENES = [
    *get_dataset_scenes("mipnerf360"),
    *get_dataset_scenes("tanksandtemples"),
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
        default=all_preset_names(),
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
        default=1000,
        help="Evaluate all images every N steps.",
    )
    return parser


def make_method_config_overrides(args: argparse.Namespace) -> dict[str, str]:
    return {
        "max_steps": str(args.max_steps),
        "ignore_mono_depth_cache": str(args.invalidate_mono_depth_cache),
        "mono_depth_cache_dir": str(
            Path(args.output_dir, "__mono_depth_cache__").absolute()
        ),
    }


def get_args_hash(args: argparse.Namespace):
    args_copy = argparse.Namespace()
    args_copy.__dict__ = args.__dict__.copy()

    unhashed_params = ["output_dir", "scenes", "presets", "invalidate_mono_depth_cache"]
    for param in unhashed_params:
        delattr(args_copy, param)

    return str(args_copy)


ARGS_HASH_FILENAME = ".nerfbaselines_evaluator_args_hash"


def output_dir_needs_overwrite(
    output_dir: Path,
    args: argparse.Namespace,
    args_hash: str,
    eval_all_iters: list[int],
) -> bool:
    if not directory_exists_and_has_files(output_dir):
        return False

    try:
        with open(output_dir / ARGS_HASH_FILENAME, "r") as f:
            old_args_hash = f.read().strip()
    except FileNotFoundError:
        return True

    for iter in eval_all_iters:
        if iter == 0:
            continue  # nerfbaselines never evals at 0

        if not (output_dir / f"predictions-{str(iter)}.tar.gz").exists():
            return True
        if not (output_dir / f"results-{str(iter)}.json").exists():
            return True

    if not (output_dir / f"checkpoint-{str(args.max_steps)}").exists():
        return True

    return old_args_hash != args_hash


def main():
    args = create_argument_parser().parse_args()

    eval_all_iters = list(range(0, args.max_steps + 1, args.eval_frequency))
    if eval_all_iters[-1] != args.max_steps:
        eval_all_iters.append(args.max_steps)

    args_hash = get_args_hash(args)

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
        print(
            ANSIEscapes.color("_" * 80, "bold"),
            ANSIEscapes.color("=" * 80 + "\n", "blue"),
            sep="\n",
        )
        curr_output_dir = Path(args.output_dir / scene / preset)

        if curr_output_dir.exists():
            if not curr_output_dir.is_dir():
                raise ValueError(f"Output path is not a directory: {curr_output_dir}")

            if not output_dir_needs_overwrite(
                curr_output_dir, args, args_hash, eval_all_iters
            ):
                print(
                    ANSIEscapes.color(
                        f"Skipping {preset} on {scene}. (Output exists and is up-to-date)",
                        "green",
                    )
                )
                continue

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
        with open(curr_output_dir / ARGS_HASH_FILENAME, "w") as f:
            f.write(args_hash)

        overrides_cli = []
        for kv_pair in make_method_config_overrides(args).items():
            overrides_cli.extend(["--set", "=".join(kv_pair)])

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


if __name__ == "__main__":
    main()
