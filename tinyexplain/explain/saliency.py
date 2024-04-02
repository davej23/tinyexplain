from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import PostProcessingFunction, ScoreFunction, TinyExplainTask, TinygradModel
from tinyexplain.logging import Logger

from .explainer import Explainer


class Saliency(Explainer):
    """Saliency Explainer"""

    def __init__(self, model: TinygradModel, task: TinyExplainTask):
        super().__init__()
        self.model = model
        self.task = task

    def explain(
        self, inputs: Tensor, targets: Tensor, postprocess_fn: PostProcessingFunction,
        score_fn: Optional[ScoreFunction] = None, device: str = "CUDA", **kwargs
    ) -> Tensor:

        Logger.debug(f"{self._log_prefix} {inputs=} {targets=}")

        inputs = inputs.to(device)
        targets = targets.to(device)

        Logger.debug(f"{self._log_prefix} Running score computation")
        Saliency.compute_score(postprocess_fn, score_fn, self.task, self.model, inputs, targets)

        explanations = inputs.grad.abs()

        if len(explanations.shape) == 4:
            explanations = explanations.permute((0, 2, 3, 1))

        Logger.debug(f"{self._log_prefix} {explanations=}")
        return explanations
