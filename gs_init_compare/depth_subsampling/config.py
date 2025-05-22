from dataclasses import dataclass
from typing import Tuple


@dataclass
class AdaptiveSubsamplingConfig:
    """
    Configures which heuristics to use for adaptive subsampling.
    """

    # Range of subsample factors to choose from.
    factor_range: Tuple[int, int] = (5, 15)
