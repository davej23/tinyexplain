from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import PostProcessingFunction, ScoreFunction, TinyExplainTask, TinygradModel
from tinyexplain.utils.logging import Logger

from .explainer import Explainer


class GradientInput(Explainer):
    """GradientInput Explainer"""

    def __init__(self, model: TinygradModel, task: TinyExplainTask):
        super().__init__()
        self.model = model
        self.task = task

    def explain(
        self, inputs: Tensor, targets: Tensor, postprocess_fn: PostProcessingFunction, score_fn: Optional[ScoreFunction] = None, **kwargs
    ) -> Tensor:

        Logger.debug(f"{self._log_prefix} Running score computation")
        GradientInput.compute_score(postprocess_fn, score_fn, self.task, self.model, inputs, targets)

        Logger.debug(f"{self._log_prefix} Extracting gradients")
        explanations = inputs.grad * inputs

        if len(explanations.shape) == 4:
            explanations = explanations.permute((0, 2, 3, 1))

        Logger.debug(f"{self._log_prefix} {explanations=}")
        return explanations
