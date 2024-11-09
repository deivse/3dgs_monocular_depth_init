"""
Runs training and evaluation for multiple scenes and initialization strategies, reports results.
"""

from abc import abstractmethod
import abc
from copy import deepcopy
import logging
from pathlib import Path
import shutil
from typing import Any, Iterable, Type

from gsplat.strategy import DefaultStrategy
from config import Config
import argparse

import trainer


def combine_results(
    combined_tb_dir: Path, result_dirs: list[Path], overwrite: bool = False
):
    tensorboard_dirs = [d / "tb" for d in result_dirs]
    tensorboard_target_dirs = [combined_tb_dir / d.name for d in result_dirs]

    for tb_dir, target_dir in zip(tensorboard_dirs, tensorboard_target_dirs):
        if target_dir.exists() and any(target_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Target directory {target_dir} already exists.")
            shutil.rmtree(target_dir)  # noqa: F821

        target_dir.mkdir(parents=True, exist_ok=True)
        for event_file in tb_dir.glob("events*"):
            target_file = target_dir / event_file.name
            shutil.copy(event_file, target_file)
            print(f"Copied {event_file} to {target_file}")

    print(f"Combined results written to {combined_tb_dir}")
    print(f"Run `tensorboard --logdir {combined_tb_dir}` to view results.")


class RunParam(metaclass=abc.ABCMeta):
    all_values: Iterable[Any]

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


class Metric3dModel(RunParam):
    all_values = ["small", "large"]

    @classmethod
    def apply(cls, config: Config, value):
        config.metric3d_config = (
            f"metric3d/mono/configs/HourglassDecoder/vit.raft5.{value}.py"
        )
        config.metric3d_weights = f"metric3d/weight/metric_depth_vit_{value}_800k.pth"

    @classmethod
    def get_run_id_str(cls, value):
        return f"vit.raft5.{value}"


class Metric3dPointDownsampleFactor(RunParam):
    all_values = [10, 20]

    @classmethod
    def apply(cls, config: Config, value):
        config.dense_depth_downsample_factor = value

    @classmethod
    def get_run_id_str(cls, value):
        return f"downsample_{value}"


class DataDir(RunParam):
    all_values = [
        "garden",
        "stump",
    ]  # "bicycle", "bonsai", "counter", "kitchen", "room", "stump"]
    # all_values = ["garden", "stump", "bicycle", "bonsai", "counter", "kitchen", "room", "stump"]

    @classmethod
    def apply(cls, config: Config, value):
        config.data_dir = f"data/360_v2/{value}"

    @classmethod
    def get_run_id_str(cls, value):
        return value


def create_base_config(max_steps=30_000):
    cfg = Config(strategy=DefaultStrategy(verbose=True))
    cfg.disable_viewer = True
    cfg.eval_steps = [500, 1000, 7_000, 30_000]
    cfg.max_steps = max_steps
    return cfg


def create_configs_with_params(
    configs: list[tuple[str, Config]], params: list[Type[RunParam]]
) -> list[tuple[str, Config]]:
    if len(params) == 0:
        return configs

    new_configs = []
    param = params[0]
    for curr_name, config in configs:
        for value in param.all_values:
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
    params_metric3d: list[Type[RunParam]],
):
    metric3d_configs = create_configs_with_params([("", base_config)], params_metric3d)
    sfm_configs = create_configs_with_params([("", base_config)], params_sfm)

    for _, cfg in metric3d_configs:
        cfg.init_type = "metric3d"

    configs = metric3d_configs + sfm_configs

    print(f"Running {len(configs)} configurations:")
    print("\n".join([name for name, _ in configs]))

    for name, config in configs:
        # Run training and evaluation
        print("===================================")
        print(f"== Running {name}")
        print("===================================")

        config.result_dir = f"results/{name}"
        try:
            if Path(config.result_dir).exists():
                logging.warning("Result directory already exists, deleting.")
                shutil.rmtree(config.result_dir, ignore_errors=True)
            trainer.run_with_config(config)
        except KeyboardInterrupt:
            print("Interrupted, deleting incomplete result dir.")
            shutil.rmtree(config.result_dir, ignore_errors=True)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=False,
        default="results",
        help="Directory containing results to combine.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default="combined_results",
        help="Directory to copy tensorboard logs to.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite target directory if it exists.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        default=False,
        help="Don't run training, just combine results.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not args.combine_only:
        base_config = create_base_config(max_steps=7000)
        run_all_combinations(
            base_config,
            [DataDir],
            [DataDir, Metric3dModel, Metric3dPointDownsampleFactor],
        )

    combine_results(
        args.output_dir,
        [d for d in results_dir.iterdir() if d.is_dir()],
        args.overwrite,
    )
