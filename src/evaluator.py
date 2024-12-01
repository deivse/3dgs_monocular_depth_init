"""
Runs training and evaluation for multiple scenes and initialization strategies, reports results.
"""

from abc import abstractmethod
import abc
from copy import deepcopy
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
from typing import Any, Iterable, Type

from gsplat.strategy import DefaultStrategy
from config import Config
import argparse

import trainer


def append_timestamp_to_dir_name(dir: Path) -> Path:
    """
    Appends a timestamp to the directory name to avoid conflicts
    when the directory already exists. 

    `dir` is not modified, a new Path object is returned.
    """
    last_edit_time = max(
        f.stat().st_mtime for f in dir.rglob("*"))
    last_edit_time_str = datetime.fromtimestamp(
        last_edit_time
    ).strftime('_%d-%m-%Y_%H:%M:%S')
    new_old_dir_name = dir.name + last_edit_time_str
    # This doesn't point combined_tb_dir to the new location
    # (Which is what we want)
    return dir.rename(dir.parent / new_old_dir_name)


def directory_exists_and_has_files(dir: Path) -> bool:
    if not dir.exists():
        return False
    for d in dir.glob("*"):
        if d.is_file():
            return True
    return False


def combine_results(
    combined_tb_dir: Path, result_dirs: list[Path], overwrite: bool = False
):
    if directory_exists_and_has_files(combined_tb_dir):
        if overwrite:
            shutil.rmtree(combined_tb_dir)
        else:
            new_name = append_timestamp_to_dir_name(combined_tb_dir)
            logging.warning(
                f"Target directory already exists and is not empty. Renaming to {new_name}."
            )

    tensorboard_dirs = [d / "tb" for d in result_dirs]
    tensorboard_target_dirs = [combined_tb_dir / d.name for d in result_dirs]

    for tb_dir, target_dir in zip(tensorboard_dirs, tensorboard_target_dirs):
        target_dir.mkdir(parents=True, exist_ok=False)
        for event_file in tb_dir.glob("events*"):
            target_file = target_dir / event_file.name
            shutil.copy(event_file, target_file)
            print(f"Copied {event_file} to {target_file}")

    print(f"Combined results written to {combined_tb_dir}")
    print(f"To view results, run:\n  tensorboard --logdir {combined_tb_dir}")


class RunParam(metaclass=abc.ABCMeta):
    values: Iterable[Any]

    @classmethod
    @abstractmethod
    def apply(cls, config: Config, value):
        """
        Apply the parameter to the config.
        """

    @classmethod
    @abstractmethod
    def get_run_id_str(cls, value):
        """
        Get a string representation to be used as part of run identifier.
        """


class MonoDepthModel(RunParam):
    values = ["metric3d-vit-small", "metric3d-vit-large", "depth_pro", "moge"]

    @classmethod
    def apply(cls, config: Config, value: str):
        if value.startswith("metric3d-vit"):
            config.mono_depth_model = "metric3d"
            suffix = value.split("-")[-1]

            config.metric3d_config = (
                f"third_party/metric3d/mono/configs/HourglassDecoder/vit.raft5.{suffix}.py"
            )
            config.metric3d_weights = (
                f"third_party/metric3d/weight/metric_depth_vit_{suffix}_800k.pth"
            )
            if not Path(config.metric3d_config).exists():
                raise FileNotFoundError(
                    f"Metric3d config file not found: {config.metric3d_config}"
                )
            if not Path(config.metric3d_weights).exists():
                raise FileNotFoundError(
                    f"Metric3d weights file not found: {config.metric3d_weights}"
                )
        elif value == "depth_pro":
            config.mono_depth_model = "depth_pro"
            config.depth_pro_checkpoint = "third_party/apple_depth_pro/checkpoints/depth_pro.pt"
            if not Path(config.depth_pro_checkpoint).exists():
                raise FileNotFoundError(
                    f"DepthPro checkpoint file not found: {config.depth_pro_checkpoint}"
                )
        elif value == "moge":
            config.mono_depth_model = "moge"
        else:
            raise NotImplementedError(f"Unsupported model: {value}")

    @classmethod
    def get_run_id_str(cls, value):
        return value


class DensePointDownsampleFactor(RunParam):
    values = [10, 20]

    @classmethod
    def apply(cls, config: Config, value):
        config.dense_depth_downsample_factor = value

    @classmethod
    def get_run_id_str(cls, value):
        return f"downsample_{value}"


class DataDir(RunParam):
    values = ["garden", "stump", "bicycle",
              "bonsai", "counter", "kitchen", "room", "stump"]

    @classmethod
    def apply(cls, config: Config, value):
        config.data_dir = f"data/360_v2/{value}"

    @classmethod
    def get_run_id_str(cls, value):
        return value


def create_configs_with_params(
    configs: list[tuple[str, Config]], params: list[Type[RunParam]]
) -> list[tuple[str, Config]]:
    if len(params) == 0:
        return configs

    new_configs = []
    param = params[0]
    for curr_name, config in configs:
        for value in param.values:
            if curr_name != "":
                new_name = f"{curr_name}_{param.get_run_id_str(value)}"
            else:
                new_name = param.get_run_id_str(value)
            new_config = deepcopy(config)
            param.apply(new_config, value)
            new_configs.append((new_name, new_config))

    return create_configs_with_params(new_configs, params[1:])


def run_all_combinations(
    base_config: Config,
    params_sfm: list[Type[RunParam]],
    params_monocular_depth: list[Type[RunParam]],
    results_dir: Path,
):
    mono_depth_configs = create_configs_with_params(
        [("", base_config)], params_monocular_depth
    )
    sfm_configs = create_configs_with_params([("", base_config)], params_sfm)

    for _, cfg in mono_depth_configs:
        cfg.init_type = "monocular_depth"

    configs = mono_depth_configs + sfm_configs

    print(f"Running {len(configs)} configurations:")
    print("  -" + "\n  -".join([name for name, _ in configs]))
    print()

    if directory_exists_and_has_files(results_dir):
        new_path = append_timestamp_to_dir_name(results_dir)
        logging.warning(
            f"Result directory already exists and is not empty, moving it to {new_path}"
        )
    elif results_dir.exists():
        # Directory exists, but may contain empty subdirectories
        # Let's clean up just in case
        shutil.rmtree(results_dir)
        logging.info(f"Deleted empty result directory {results_dir}")

    for name, config in configs:
        # Run training and evaluation
        print("=" * (len(name) + 14))
        print(f"== Running {name} ==")
        print("=" * (len(name) + 14))

        config.result_dir = str(results_dir / name)
        try:
            if Path(config.result_dir).exists():
                logging.warning("Result directory already exists, deleting.")
                shutil.rmtree(config.result_dir, ignore_errors=True)
            trainer.run_with_config(config)
        except KeyboardInterrupt:
            print("Interrupted, deleting incomplete result dir.")
            shutil.rmtree(config.result_dir, ignore_errors=True)
            break
        except Exception as e:
            logging.error(f"Error running {name}: {e}")
            shutil.rmtree(config.result_dir, ignore_errors=True)
            logging.info(f"Deleted incomplete result dir {config.result_dir}")


def create_argument_parser():
    parser = argparse.ArgumentParser()

    def add_argument(*args, **kwargs):
        if "default" in kwargs:
            kwargs["help"] = f"(={kwargs['default']})\n{kwargs.get('help', '')}"
        parser.add_argument(*args, **kwargs)

    add_argument(
        "--results-dir",
        type=Path,
        required=False,
        default="results",
        help="Directory containing results to combine.",
    )
    add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default="combined_results",
        help="Directory to copy tensorboard logs to.",
    )
    add_argument(
        "--overwrite-output-dir",
        action="store_true",
        default=False,
        help="Overwrite target directory if it exists.",
    )
    add_argument(
        "--combine-only",
        action="store_true",
        default=False,
        help="Don't run training, just combine results.",
    )
    add_argument(
        "--max-steps",
        type=int,
        default=30_000,
        help="Maximum number of steps to run training for.",
    )
    add_argument(
        "--eval-frequency",
        type=int,
        default=1000,
        help=("How often evaluation is run during training. Evaluation is always run at steps 250, 500, 750, 1000, 1500 and max_steps."
              "After step 1500 the next steps are determined by this value: range(1000, max_steps, eval_frequency)."),
    )
    add_argument(
        "--data-dirs",
        nargs="+",
        default=["garden", "stump", "bicycle", "bonsai",
                 "counter", "kitchen", "room", "stump"],
        help="Data directories to run training and evaluation for.",
    )
    add_argument(
        "--invalidate-mono-depth-cache",
        action="store_true",
        default=False,
        help="Invalidate the cache for monocular depth predictors"
    )
    add_argument(
        "--models",
        nargs="+",
        default=["metric3d-vit-small",
                 "metric3d-vit-large", "depth_pro", "moge"],
        help="Monocular depth models to evaluate.",
    )
    add_argument(
        "--downsample-factors",
        nargs="+",
        default=[10, 20],
        type=int,
        help="Dense points downsample factor for monocular depth initialization.",
    )
    add_argument(
        "--enable-viewer",
        action="store_true",
        default=False,
        help="Enable the viewer for the training process.",
    )
    return parser


def create_base_config(args: argparse.Namespace):
    cfg = Config(strategy=DefaultStrategy(verbose=True))
    cfg.non_blocking_viewer = True
    cfg.disable_viewer = not args.enable_viewer
    cfg.invalidate_mono_depth_cache = args.invalidate_mono_depth_cache

    cfg.max_steps = args.max_steps

    cfg.eval_steps = list(range(250, 1500, 500)) + \
        list(range(1500, args.max_steps, args.eval_frequency))
    cfg.eval_steps.append(args.max_steps)
    cfg.eval_steps = list(set(cfg.eval_steps))
    cfg.eval_steps.sort()

    return cfg


def configure_combinations(args: argparse.Namespace):
    DataDir.values = args.data_dirs
    MonoDepthModel.values = args.models
    DensePointDownsampleFactor.values = args.downsample_factors


def main():
    args = create_argument_parser().parse_args()

    results_dir = Path(args.results_dir)

    if not args.combine_only:
        base_config = create_base_config(args)
        configure_combinations(args)
        run_all_combinations(
            base_config,
            [DataDir],
            [DataDir, MonoDepthModel, DensePointDownsampleFactor],
            results_dir
        )

    combine_results(
        args.output_dir,
        [d for d in results_dir.iterdir() if d.is_dir()],
        args.overwrite_output_dir,
    )


if __name__ == "__main__":
    main()
