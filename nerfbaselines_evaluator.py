"""
Runs training and evaluation for multiple scenes and initialization strategies, reports results.
"""

from dataclasses import fields
from datetime import datetime
from enum import Enum
import itertools
from math import ceil
import os
from pathlib import Path
import shutil
import subprocess

import argparse

from itertools import product
import sys
from types import NoneType
import typing
from typing_extensions import Self
from gs_init_compare.config import Config

from nerfbaselines import get_dataset_spec
from tensorboard.backend.event_processing import event_accumulator


class ANSIEscapes(Enum):
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
    def format(text: str, escape: str | Self):
        seq = escape if isinstance(escape, ANSIEscapes) else ANSIEscapes.by_name(escape)
        return f"{seq.value}{text}{ANSIEscapes.END_SEQUENCE.value}"


def ansiesc_print(value: str, escape: str | ANSIEscapes):
    print(ANSIEscapes.format(value, escape))


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


ALL_SCENES = [
    *get_dataset_scenes("mipnerf360", []),
    *get_dataset_scenes("tanksandtemples", []),
]

# print(ALL_SCENES)

# ALL_SCENES = [
#     "mipnerf360/garden",  # PSNR worse, others better
#     "mipnerf360/bonsai",  # everything worse
#     "mipnerf360/stump",  # everything better
#     "mipnerf360/flowers",  # PSNR worse, others better
#     "mipnerf360/bicycle",  # everything better
#     "mipnerf360/kitchen",  # everything worse
#     "mipnerf360/treehill",  # PSNR worse, others better
#     "mipnerf360/room",  # everything better (but significantly more gaussians)
#     "mipnerf360/counter",  # PSNR worse, others better
#     "tanksandtemples/truck",  # PSNR worse, others better (only lpips) more gaussians
#     "tanksandtemples/train",  # PSNR worse, others better (only lpips) less gaussians
# ]


def create_argument_parser():
    parser = argparse.ArgumentParser()

    def add_argument(*args, **kwargs):
        if "default" in kwargs:
            kwargs["help"] = f"(={kwargs['default']})\n{kwargs.get('help', '')}"
        parser.add_argument(*args, **kwargs)

    add_argument(
        "--configs",
        nargs="*",
        help="Presets to pass to the method.",
        default=[],
        type=str,
    )
    add_argument(
        "--configs-file",
        help="File from which to read config strings, 1 per line.",
        type=str,
        default=None,
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
    add_argument("--print-default-presets", action="store_true", default=False)
    add_argument("--force-overwrite", action="store_true", default=False)
    add_argument("--pts-only", action="store_true", default=False)
    return parser


def get_config_strings(args: argparse.Namespace):
    num_exclusive_options_specified = int(len(args.configs) > 0) + int(
        args.configs_file is not None
    )
    if num_exclusive_options_specified == 0:
        raise ValueError("Either --configs or --configs-file must be specified.")
    if num_exclusive_options_specified > 1:
        raise ValueError("Only one of  {--configs, --configs-file} may be specified.")

    if args.configs_file is None:
        return args.configs

    with Path(args.configs_file).open("r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) != 0]


def get_all_possible_vals_of_param(name: str):
    name = name.replace("-", "_")
    address_parts = name.split(".")
    curr_type = Config
    for field_name in address_parts:
        curr_type = {field.name: field.type for field in fields(curr_type)}[field_name]

    def raise_if_empty(vals):
        if len(vals) == 0:
            raise RuntimeError(
                f"List of all possible values for param {name} is empty."
            )
        return vals

    def raise_unsupported():
        raise ValueError(
            f"Can't get all possible values of param {name}. Unsupported type: {curr_type}"
        )

    if typing.get_origin(curr_type) is typing.Union:
        args = typing.get_args(curr_type)
        if all(isinstance(x, str) for x in args):
            return raise_if_empty(list(args))
        if len(args) == 2 and args[1] is NoneType:
            curr_type = args[0]
        else:
            raise_unsupported()

    try:
        if issubclass(curr_type, Enum):
            return raise_if_empty([str(member.value) for member in list(curr_type)])
    except TypeError:
        pass

    try:
        args = typing.get_args(curr_type)
        if all(isinstance(x, str) for x in args):
            return raise_if_empty(list(args))
    except Exception:
        pass

    raise_unsupported()


ParsedConfigStr = list[tuple[str, list[str]]]
ParamList = list[tuple[str, str]]


def parse_config_string(config_str: str) -> list[ParamList]:
    """
    # Configs string example
    {default, mcmc} --mdi.predictor={metric3d,depth_pro} --mdi.depth-alignment-strategy={ransac,dynamic} --mdi.subsample-factor={adaptive}
    # Special value example
    {default, mcmc} --mdi.predictor=[ALL] --mdi.depth-alignment-strategy={ransac,dynamic} --mdi.subsample-factor={adaptive}
    """

    parts: list[str] = []
    current_part = ""
    brace_count = 0
    in_quotes = False
    quote_char = None

    for char in config_str:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
            current_part += char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current_part += char
        elif char == "{" and not in_quotes:
            brace_count += 1
            current_part += char
        elif char == "}" and not in_quotes:
            brace_count -= 1
            current_part += char
        elif char == " " and brace_count == 0 and not in_quotes:
            if current_part:
                parts.append(current_part)
                current_part = ""
        else:
            current_part += char

    if current_part:
        parts.append(current_part)

    ALL = "[ALL]"
    special_val_handlers = {ALL: get_all_possible_vals_of_param}
    parsed: ParsedConfigStr = []
    for part in parts:
        eq_pos = part.find("=")
        if eq_pos == -1:
            raise ValueError(
                f"'=' not found in \"{part}\" All config string param definitions must be formatted as"
                + "key={value1, value2, ...}"
            )
        name = part[:eq_pos].removeprefix("-").removeprefix("-")

        found_special_val = False
        for val, handler in special_val_handlers.items():
            full_values_part = part[eq_pos + 1 :]
            if full_values_part == val:
                values = handler(name)
                if len(values) != 0:
                    parsed.append((name, values))
                found_special_val = True
                break
        if found_special_val:
            continue

        if part[eq_pos + 1] == "{":  # List of options
            if not part[-1] == "}":
                raise ValueError("Invalid config string: unclosed {} at " + part)
            values = part[eq_pos + 2 : -1].replace(" ", "").split(",")
            parsed.append((name, values))
            continue

        if "{" in part or "}" in part:
            raise ValueError(
                "{} contained in part, but open brace is not on first pos: " + part
            )
        value = part[eq_pos + 1 :]
        parsed.append((name, [value]))

    with_param_name = []
    for name, values in parsed:
        with_param_name.append([(name, val) for val in values])
    # Pass through set to deduplicate
    return list(set(itertools.product(*with_param_name)))


CONFIG_STR_FORBIDDEN_PARAM_NAMES = {
    "max_steps",
    "mdi.ignore_cache",
    "mdi.cache_dir",
    "mdi.pts_only",
}


def make_method_config_overrides(args: argparse.Namespace) -> dict[str, str]:
    retval = {
        "max_steps": str(args.max_steps),
        "mdi.ignore_cache": str(args.invalidate_mono_depth_cache),
        "mdi.cache_dir": str(Path(args.output_dir, "__mono_depth_cache__").absolute()),
        "mdi.pts_only": str(args.pts_only),
    }
    assert CONFIG_STR_FORBIDDEN_PARAM_NAMES == set(retval.keys())
    return retval


IGNORED_PARAM_NAMES = {
    "mdi.pts-output-dir",
    "mdi.pts-output-per-image",
    "mdi.no-pts-output-per-image",
}


def make_config_name(params: ParamList) -> str:
    out = []
    RENAMES = {
        "mdi.predictor": None,
        "mdi.depth-alignment-strategy": "align",
        "mdi.subsample-factor": "subsample",
    }

    def with_tildes(n):
        return n.replace("_", "-")

    for name, value in params:
        if name is not None and with_tildes(name) in RENAMES:
            name = RENAMES[with_tildes(name)]
        if name is None:
            # NOTE: SPECIAL HANDLING FOR DEFAULT STRATEGY!
            if value != "default":
                out.append(value)
            continue

        name_tilde = with_tildes(name)
        if name in IGNORED_PARAM_NAMES or name_tilde in IGNORED_PARAM_NAMES:
            continue

        if (
            name in CONFIG_STR_FORBIDDEN_PARAM_NAMES
            or name_tilde in CONFIG_STR_FORBIDDEN_PARAM_NAMES
        ):
            raise ValueError(
                f"Parameter {name_tilde} can not be part of config string"
                " (it's either set by evaluator automatically, or evaluator argument should be used instead of passing directly)"
            )

        name = name.removeprefix("mdi.")  # Just adds useless noise

        out.append(f"{name}={value}")
    return "|".join(out)


def get_args_str(args: argparse.Namespace):
    args_copy = argparse.Namespace()
    args_copy.__dict__ = args.__dict__.copy()

    UNHASHED_PARAMS = [
        "eval_frequency",
        "output_dir",
        "scenes",
        "configs",
        "configs_file",
        "invalidate_mono_depth_cache",
    ]
    for param in UNHASHED_PARAMS:
        if hasattr(args_copy, param):
            delattr(args_copy, param)

    return str(args_copy)


ARGS_STR_FILENAME = ".nerfbaselines_evaluator_args_hash"


def output_dir_needs_overwrite(
    output_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    eval_all_iters: list[int],
) -> bool:
    if args.force_overwrite:
        return True

    if not directory_exists_and_has_files(output_dir):
        return True

    try:
        with open(output_dir / ARGS_STR_FILENAME, "r") as f:
            old_run_id = f.read().strip()
    except FileNotFoundError:
        return True

    for iter in eval_all_iters:
        if iter == 0:
            continue  # nerfbaselines never evals at 0

        if not (output_dir / f"results-{str(iter)}.json").exists():
            return True

    return old_run_id != run_id


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


MCMC_GAUSSIAN_CAPS = {
    "mipnerf360/garden": 6000000,
    "mipnerf360/bonsai": 4800000,
    "mipnerf360/stump": 4700000,
    "mipnerf360/flowers": 3700000,
    "mipnerf360/bicycle": 6100000,
    "mipnerf360/kitchen": 4300000,
    "mipnerf360/treehill": 3800000,
    "mipnerf360/room": 5500000,
    "mipnerf360/counter": 4000000,
}


def run_combination(
    scene: str,
    config: ParamList,
    args: argparse.Namespace,
    args_str: str,
    eval_all_iters: list[int],
):
    print(
        ANSIEscapes.format("_" * 80, "bold"),
        ANSIEscapes.format("=" * 80 + "\n", "blue"),
        sep="\n",
    )
    config_name = make_config_name(config)
    curr_output_dir = Path(args.output_dir / scene / config_name)
    if args.run_label:
        curr_output_dir = curr_output_dir.with_name(
            f"{curr_output_dir.name}_{args.run_label}"
        )

    run_id = args_str + config_name + scene

    if directory_exists_and_has_files(curr_output_dir) and not args.pts_only:
        if not curr_output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {curr_output_dir}")

        if not output_dir_needs_overwrite(
            curr_output_dir, args, run_id, eval_all_iters
        ):
            print(
                ANSIEscapes.format(
                    f"Skipping {config} on {scene}. (Output exists and is up-to-date)",
                    "green",
                )
            )
            return

        new_path = rename_old_dir_with_timestamp(curr_output_dir, args.output_dir)
        print(
            ANSIEscapes.format(
                f"Detected results mismatch. Old output directory moved to: {new_path}",
                "yellow",
            )
        )
        assert not curr_output_dir.exists()

    print(
        ANSIEscapes.format(
            f"Training {config_name} on {scene}. (Outputting to: {curr_output_dir})",
            "blue",
        )
    )
    curr_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.pts_only:
        with open(curr_output_dir / ARGS_STR_FILENAME, "w") as f:
            f.write(run_id)

    overrides_cli = []
    for kv_pair in make_method_config_overrides(args).items():
        overrides_cli.extend(["--set", "=".join(kv_pair)])

    if "mcmc" in {param_name for param_name, _ in config}:
        overrides_cli.extend(
            [
                "--set",
                f"strategy.cap_max={MCMC_GAUSSIAN_CAPS[scene]}",
            ]
        )

    for param_name, value in config:
        param_name = param_name.replace("-", "_")
        if param_name == "strategy":
            if value.lower() == "default":
                value = "DefaultStrategy"
            elif value.lower() == "mcmc":
                value = "MCMCStrategy"
        overrides_cli.extend(["--set", f"{param_name}={value}"])

    subprocess.run(
        [
            "nerfbaselines",
            "train",
            "--backend=python",
            "--method=gs-init-compare",
            f"--output={curr_output_dir}",
            f"--data=external://{scene}",
            f"--eval-all-iters={','.join(map(str, eval_all_iters))}",
        ]
        + overrides_cli
    )

    try:
        checkpoint_dir = curr_output_dir / f"checkpoint-{args.max_steps}"
        splat_file = checkpoint_dir / f"splats_{args.max_steps}.ply"
        if splat_file.exists():
            splat_file.rename(curr_output_dir / splat_file.name)
        shutil.rmtree(checkpoint_dir)

        # Remove unnecessary outputs cuz I would run out of disk space...
        Path(curr_output_dir / "output.zip").unlink()

        # Delete predictions except last step and middle step:
        for iter in eval_all_iters:
            if iter not in [0, 8000, 14000, args.max_steps]:
                Path(curr_output_dir / f"predictions-{str(iter)}.tar.gz").unlink()
    except FileNotFoundError as e:
        print(ANSIEscapes.format(f"Error: Training output not found:\n {e}", "red"))


def adjust_combinations_if_slurm(
    combinations: list[tuple[str, ParamList]],
) -> list[tuple[str, ParamList]]:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    task_id_min = os.environ.get("SLURM_ARRAY_TASK_MIN", None)
    array_size = os.environ.get("SLURM_ARRAY_TASK_COUNT", None)

    if task_id is None:
        return combinations

    if array_size is None or task_id_min is None:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is set, but SLURM_ARRAY_TASK_COUNT or SLURM_ARRAY_TASK_MIN is not!"
        )
    task_id, task_id_min, array_size = int(task_id), int(task_id_min), int(array_size)

    job_ix = task_id - task_id_min
    ansiesc_print(
        f"Running in SLURM: {job_ix=} {array_size=} {task_id=} {task_id_min=}",
        ANSIEscapes.YELLOW,
    )

    base_tasks_per_job = len(combinations) // array_size
    remaining_tasks = len(combinations) - base_tasks_per_job * array_size

    num_tasks_per_job = [base_tasks_per_job for _ in range(array_size)]
    for i in range(remaining_tasks):
        num_tasks_per_job[i] += 1

    tasks_before_this_job = sum(num_tasks_per_job[:job_ix])
    this_job_tasks = (
        base_tasks_per_job + 1 if job_ix < remaining_tasks else base_tasks_per_job
    )

    ansiesc_print(
        f"This job will run {this_job_tasks} combinations - [{tasks_before_this_job}, {tasks_before_this_job + this_job_tasks - 1}].",
        ANSIEscapes.YELLOW,
    )
    return combinations[tasks_before_this_job : tasks_before_this_job + this_job_tasks]


def get_eval_it_list(args: argparse.Namespace):
    eval_all_iters = list(range(0, args.max_steps + 1, args.eval_frequency))
    if eval_all_iters[-1] != args.max_steps:
        eval_all_iters.append(args.max_steps)
    return eval_all_iters


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = create_argument_parser().parse_args()
    configs: list[ParamList] = []
    for config_str in get_config_strings(args):
        configs.extend(parse_config_string(config_str))

    args_str = get_args_str(args)
    eval_all_iters = get_eval_it_list(args)

    combinations = list(product(args.scenes, configs))
    combinations = adjust_combinations_if_slurm(combinations)
    configs = {cfg for _, cfg in combinations}
    scenes = {scene for scene, _ in combinations}
    print(
        ANSIEscapes.format("_" * 80, "bold"),
        ANSIEscapes.format(f"Will train {len(combinations)} combinations.", "bold"),
        ANSIEscapes.format("Settings:", "bold"),
        f"\tOutput directory: {ANSIEscapes.format(args.output_dir, 'cyan')}",
        f"\tMax steps: {ANSIEscapes.format(args.max_steps, 'cyan')}",
        f"\tEvaluation frequency: {ANSIEscapes.format(args.eval_frequency, 'cyan')}",
        "\tConfigs: " + ANSIEscapes.format("\n\t          ".join(make_config_name(c)for c in configs), 'cyan'),
        f"\tScenes: {ANSIEscapes.format(scenes, 'cyan')}",
        f"\tEval all iters: {ANSIEscapes.format(eval_all_iters, 'cyan')}",
        sep="\n",
    )

    for scene, config in combinations:
        run_combination(scene, config, args, args_str, eval_all_iters)


if __name__ == "__main__":
    main()
