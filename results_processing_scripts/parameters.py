import abc
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Callable, Dict
from tensorboard.backend.event_processing import event_accumulator


class TensorboardDataLoader:
    def __init__(self, file):
        self.ea = event_accumulator.EventAccumulator(
            str(file),
            size_guidance={"tensors": 1, "histograms": 1,
                           "images": 1, "scalars": 1},
        )
        self.ea.Reload()

    def read_param(self, param_name, step):
        if param_name not in self.ea.Tags().get("scalars", []):
            raise ValueError(
                f"Parameter {param_name} not found in TensorBoard logs.")

        scalars = self.ea.Scalars(param_name)
        if not scalars:
            raise ValueError(
                f"No scalar data found for parameter {param_name}.")

        for scalar in scalars:
            if scalar.step == step:
                return scalar.value

        raise ValueError(f"Step {step} not found for parameter {param_name}.")


class ParamOrdering(Enum):
    HIGHER_IS_BETTER = 1
    LOWER_IS_BETTER = 2


@dataclass
class ParameterInstance:
    name: str
    value: int | float
    ordering: ParamOrdering
    formatter: Callable[[int | float], str] = str
    should_highlight_best: bool = True

    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise ValueError(
                f"Invalid value {self.value} ({type(self.value)}) for parameter {self.name}"
            )

    def get_formatted_value(self):
        return self.formatter(self.value)

    def __lt__(self, other):
        if other is None:
            return False
        if not isinstance(other, ParameterInstance):
            return NotImplemented
        if self.ordering == ParamOrdering.HIGHER_IS_BETTER:
            return self.value < other.value
        elif self.ordering == ParamOrdering.LOWER_IS_BETTER:
            return self.value > other.value
        else:
            raise ValueError(f"Invalid ordering {self.ordering}")


def default_param_formatter(value):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


@dataclass
class Parameter(abc.ABC):
    name: str
    formatter: Callable[[int | float], str] = default_param_formatter
    ordering: ParamOrdering = ParamOrdering.HIGHER_IS_BETTER
    should_highlight_best: bool = True

    @abc.abstractmethod
    def load(self, results_dir: Path, step: int) -> ParameterInstance:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_patches(self, results_dir: Path, step: int) -> Dict[int, ParameterInstance]:
        raise NotImplementedError()

    def make_instance(self, value):
        return ParameterInstance(
            self.name, value, self.ordering, self.formatter, self.should_highlight_best
        )


class TensorboardParameter(Parameter):
    def __init__(
        self,
        name: str,
        tensorboard_id: str,
        formatter: Callable[[int | float], str] = default_param_formatter,
        ordering: ParamOrdering = ParamOrdering.HIGHER_IS_BETTER,
        should_highlight_best: bool = True,
    ):
        super().__init__(name, formatter, ordering, should_highlight_best)
        self.tensorboard_id = tensorboard_id

    def load(self, results_dir: Path, step: int) -> ParameterInstance:
        try:
            tensorboard_file = next(
                (results_dir / "tensorboard").glob("events.out.tfevents.*")
            )
            data_loader = TensorboardDataLoader(tensorboard_file)
            val = data_loader.read_param(self.tensorboard_id, step)
            return self.make_instance(val)
        except StopIteration:
            raise ValueError(
                f"Tensorboard file not found in {results_dir / 'tensorboard'}"
            )
        except Exception as e:
            raise ValueError(
                f"Error loading tensorboard parameter {self.name} from {tensorboard_file}: {e}"
            )

    def load_patches(self, results_dir, step):
        raise NotImplementedError(
            "TensorboardParameter does not support loading patches.")


class NerfbaselinesJSONParameter(Parameter):
    def __init__(
        self,
        name: str,
        json_name: str,
        formatter: Callable[[int | float], str] = default_param_formatter,
        ordering: ParamOrdering = ParamOrdering.HIGHER_IS_BETTER,
        should_highlight_best: bool = True,
    ):
        super().__init__(name, formatter, ordering, should_highlight_best)
        self.json_name = json_name

    def load(self, results_dir, step) -> ParameterInstance:
        json_file = results_dir / f"results-{step}.json"
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                return self.make_instance(data["metrics"][self.json_name])
        except FileNotFoundError:
            raise ValueError(
                f"JSON file {json_file} not found in {results_dir}")
        except KeyError:
            raise ValueError(
                f"Key metrics.{self.json_name} not found in {json_file}")
        except Exception as e:
            raise ValueError(
                f"Error loading JSON parameter {self.name} from {json_file}: {e}"
            )

    def load_patches(self, results_dir, step, reduce_bins=1) -> Dict[int, ParameterInstance]:
        json_file = results_dir / f"results-{step}.json"
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                patches: dict[str, dict] = data["metrics"]["patches"]

                bins = {int(bin_ix): self.make_instance(
                    bin_vals[self.json_name]) for bin_ix, bin_vals in patches.items()}
                patch_counts = {int(bin_ix): bin_vals["num_patches"]
                                for bin_ix, bin_vals in patches.items()}

                if reduce_bins > 1:
                    reduced_bins: Dict[int, ParameterInstance] = {}
                    reduced_patch_counts: Dict[int, int] = {}
                    sorted_bin_indices = sorted(bins.keys())
                    for i in range(0, len(sorted_bin_indices), reduce_bins):
                        bin_group = sorted_bin_indices[i:i + reduce_bins]
                        if not bin_group:
                            continue

                        if self.json_name == "num_patches":
                            total_count = sum(
                                patch_counts[bin_ix] for bin_ix in bin_group)
                            reduced_bin_index = bin_group[0] // reduce_bins
                            reduced_bins[reduced_bin_index] = self.make_instance(
                                total_count)
                            reduced_patch_counts[reduced_bin_index] = total_count
                        else:
                            avg_value = sum(
                                bins[bin_ix].value * patch_counts[bin_ix] for bin_ix in bin_group
                            ) / sum(patch_counts[bin_ix] for bin_ix in bin_group)
                            reduced_bin_index = bin_group[0] // reduce_bins
                            reduced_bins[reduced_bin_index] = self.make_instance(
                                avg_value)
                            reduced_patch_counts[reduced_bin_index] = sum(
                                patch_counts[bin_ix] for bin_ix in bin_group
                            )
                    return reduced_bins

                return bins

        except FileNotFoundError:
            raise ValueError(
                f"JSON file {json_file} not found in {results_dir}")
        except Exception as e:
            raise ValueError(
                f"Error loading JSON parameter {self.name} from {json_file}: {e}"
            )
