"""
Runs training and evaluation for multiple scenes and initialization strategies, reports results.
"""

from datetime import datetime
from pathlib import Path
import subprocess

import argparse

from itertools import product
from gs_init_compare.nerfbaselines_integration.make_presets import all_preset_names


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


def rename_old_dir_with_timestamp(dir: Path) -> Path:
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
    # This doesn't point combined_tb_dir to the new location
    # (Which is what we want)
    return dir.rename(dir.parent / new_old_dir_name)


def directory_exists_and_has_files(dir: Path) -> bool:
    if not dir.exists():
        return False
    for d in dir.rglob("*"):
        if d.is_file():
            return True
    return False


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
        "--datasets",
        nargs="+",
        default=[
            "mipnerf360/garden",
            # TODO: Add more datasets
        ],
        help="Datasets to train and evaluate on. Dataset names passed to nerfbaselines with 'external://' prefix.",
    )
    add_argument(
        "--invalidate-mono-depth-cache",
        action="store_true",
        default=False,
        help="Invalidate the cache for monocular depth predictors",
    )
    add_argument(
        "--downsample-factors",
        nargs="+",
        default=[10, 20],
        type=int,
        help="Dense points downsample factor for monocular depth initialization.",
    )
    return parser


def make_method_config_overrides(args: argparse.Namespace) -> dict[str, str]:
    return {
        "max_steps": str(args.max_steps),
        "ignore_mono_depth_cache": str(args.invalidate_mono_depth_cache),
    }


def main():
    args = create_argument_parser().parse_args()

    for preset, dataset in product(args.presets, args.datasets):
        output_dir = args.output_dir / dataset / preset
        print(
            ANSIEscapes.color(
                f"Training {preset} on {dataset}. (Outputting to: {output_dir})", "blue"
            )
        )

        overrides_cli = []
        for kv_pair in make_method_config_overrides(args).items():
            overrides_cli.extend(["--set", "=".join(kv_pair)])

        subprocess.run(
            [
                "nerfbaselines",
                "train",
                "--backend=python",
                "--method=gs-init-compare",
                f"--output={output_dir}",
                f"--presets={preset}",
                f"--data=external://{dataset}",
            ]
            + overrides_cli
        )


if __name__ == "__main__":
    main()
