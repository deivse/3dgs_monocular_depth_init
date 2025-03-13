import abc
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Callable
from tensorboard.backend.event_processing import event_accumulator


class TensorboardDataLoader:
    def __init__(self, file):
        self.ea = event_accumulator.EventAccumulator(
            str(file),
        )
        self.ea.Reload()

    def read_param(self, param_name, step):
        if param_name not in self.ea.Tags().get("scalars", []):
            raise ValueError(f"Parameter {param_name} not found in TensorBoard logs.")

        scalars = self.ea.Scalars(param_name)
        if not scalars:
            raise ValueError(f"No scalar data found for parameter {param_name}.")

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

    def _make_instance(self, value):
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
            return self._make_instance(val)
        except StopIteration:
            raise ValueError(
                f"Tensorboard file not found in {results_dir / 'tensorboard'}"
            )
        except Exception as e:
            raise ValueError(
                f"Error loading tensorboard parameter {self.name} from {tensorboard_file}: {e}"
            )


class NerfbaselinesJSONParameter(Parameter):
    def __init__(
        self,
        name: str,
        json_path: str,
        formatter: Callable[[int | float], str] = default_param_formatter,
        ordering: ParamOrdering = ParamOrdering.HIGHER_IS_BETTER,
        should_highlight_best: bool = True,
    ):
        super().__init__(name, formatter, ordering, should_highlight_best)
        self.json_path = json_path.split(".")

    def load(self, results_dir, step) -> ParameterInstance:
        json_file = results_dir / f"results-{step}.json"
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                for key in self.json_path:
                    data = data[key]
                return self._make_instance(data)
        except FileNotFoundError:
            raise ValueError(f"JSON file {json_file} not found in {results_dir}")
        except KeyError:
            raise ValueError(f"Key {'/'.join(self.json_path)} not found in {json_file}")
        except Exception as e:
            raise ValueError(
                f"Error loading JSON parameter {self.name} from {json_file}: {e}"
            )
