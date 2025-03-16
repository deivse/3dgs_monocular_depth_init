from enum import Enum


class DepthAlignmentStrategyEnum(str, Enum):
    lstsqrs = "lstsqrs"
    ransac = "ransac"
    msac = "msac"

    def get_implementation(self):
        if self == self.lstsqrs:
            from .lstsqrs import DepthAlignmentLstSqrs

            return DepthAlignmentLstSqrs
        elif self == self.ransac:
            from .ransacs import DepthAlignmentRansac

            return DepthAlignmentRansac
        elif self == self.msac:
            from .ransacs import DepthAlignmentMsac

            return DepthAlignmentMsac
        else:
            raise NotImplementedError(f"Unknown depth alignment strategy: {self}")
