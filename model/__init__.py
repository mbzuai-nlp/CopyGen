from .labels import CopyTrainingSignals, build_training_signals
from .matcher import MatchFeatures, SourceMatcher

__all__ = [
    "CopyTrainingSignals",
    "MatchFeatures",
    "SourceMatcher",
    "build_training_signals",
]

try:
    from .model import (
        CopyConstraintModel,
        TRIGGER_TO_ID,
        QwenCopyConstraintModel,
        asymmetric_length_regression_loss,
        trigger_nll_loss,
    )

    __all__.extend(
        [
            "CopyConstraintModel",
            "QwenCopyConstraintModel",
            "TRIGGER_TO_ID",
            "asymmetric_length_regression_loss",
            "trigger_nll_loss",
        ]
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
