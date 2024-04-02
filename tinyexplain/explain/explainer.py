from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from tinygrad import Tensor

from tinyexplain.types import PostProcessingFunction, TinyExplainTask, TinygradModel

from tinyexplain.logging import Logger


class Explainer(ABC):
    """Explainer abstract base class"""

    def __init__(self, *args, **kwargs):
        self._log_prefix = f"{type(self).__name__}:"

    @abstractmethod
    def explain(  # pylint: disable=unused-argument
        self, inputs: Tensor, targets: Tensor, postprocess_fn: PostProcessingFunction, **kwargs
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
        **kwargs,
    ) -> Tensor:
        Logger.debug(f"compute_score: {task=} {inputs=} {targets=}")

        if requires_grad:
            Logger.debug(f"compute_score: Setting requires_grad=True")
            inputs.requires_grad = True
            targets.requires_grad = True

        Logger.debug(f"compute_score: Running forward pass")
        out = model(inputs)
        Logger.debug(f"compute_score: Running postprocess_fn")
        out = postprocess_fn(out)

        if score_fn is None:
            Logger.debug(f"compute_score: Running score_fn")
            score = (
                task.score_fn(out, targets, inputs.shape[2], inputs.shape[3])[0][0]  # type: ignore[call-arg]
                if task is TinyExplainTask.OBJECT_DETECTION
                else task.score_fn(out, targets)
            )  # type: ignore[call-arg]
        else:
            score = score_fn(out, targets)  # type: ignore[call-arg]

        Logger.debug(f"compute_score: Computing gradients")
        score.backward()

        return score
