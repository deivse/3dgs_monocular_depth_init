from dataclasses import dataclass


@dataclass
class AdaptiveSubsamplingConfig:
    """
    Configures which heuristics to use for adaptive subsampling.
    """

    # Range of subsample factors to choose from.
    factor_range_min: int = 5
    factor_range_max: int = 15
