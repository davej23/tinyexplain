from abc import ABC
from typing import Any, Callable, Optional

from tinygrad import Tensor

from tinyexplain.types import (PostProcessingFunction, TinyExplainTask,
                               TinygradModel)


class Explainer(ABC):
    """Explainer abstract base class"""
    def __init__(self, *args, **kwargs): ...

    def explain(  # pylint: disable=unused-argument
        self,
        inputs: Tensor,
        targets: Tensor,
        postprocess_fn: PostProcessingFunction,
        **kwargs
    ) -> Tensor: ...

    @staticmethod
    def compute_score(
        postprocess_fn: Callable[[Tensor], Tensor],
        score_fn: Optional[Callable[[Any], Tensor]],
        task: TinyExplainTask,
        model: TinygradModel,
        inputs: Tensor,
        targets: Tensor,
        requires_grad: bool = True,
        **kwargs
    ) -> Tensor:
        if requires_grad:
            inputs.requires_grad = True
            targets.requires_grad = True

        out = model(inputs)
        out = postprocess_fn(out)

        if score_fn is None:
            score = task.score_fn(  # type: ignore[call-arg]
                        out, targets, inputs.shape[2], inputs.shape[3]
                    )[0][0] \
                    if task is TinyExplainTask.OBJECT_DETECTION \
                    else task.score_fn(out, targets)  # type: ignore[call-arg]
        else:
            score = score_fn(out, targets)  # type: ignore[call-arg]

        score.backward()

        return score
