from enum import Enum
from typing import Any, Callable

import numpy.typing as npt
from tinygrad import Tensor

from tinyexplain.task_functions import (cce_score, drise_score, iou_score,
                                        mse_score)

TinygradModel = Callable[[Tensor], Tensor]
NumpyArray = npt.NDArray[Any]
PostProcessingFunction = Callable[[Tensor], Tensor]
ScoreFunction = Callable[[Any], Tensor]


class TinyExplainTask(Enum):
    OBJECT_DETECTION = 0
    IMAGE_CLASSIFICATION = 1
    SEMANTIC_SEGMENTATION = 2
    TABULAR_DATA = 3
    TIME_SERIES = 4

    @property
    def score_fn(self) -> ScoreFunction:
        if self is TinyExplainTask.OBJECT_DETECTION:
            return drise_score  # type: ignore[return-value]
        if self is TinyExplainTask.IMAGE_CLASSIFICATION:
            return mse_score  # type: ignore[return-value]
        if self is TinyExplainTask.SEMANTIC_SEGMENTATION:
            return iou_score  # type: ignore[return-value]
        if self is TinyExplainTask.TABULAR_DATA:
            return mse_score  # type: ignore[return-value]
        if self is TinyExplainTask.TIME_SERIES:
            return mse_score  # type: ignore[return-value]
        return mse_score
