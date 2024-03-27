from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import (PostProcessingFunction, ScoreFunction,
                               TinyExplainTask, TinygradModel)

from .explainer import Explainer


class Saliency(Explainer):
    """Saliency Explainer"""
    def __init__(self, model: TinygradModel, task: TinyExplainTask):
        super().__init__()
        self.model = model
        self.task = task

    def explain(
        self,
        inputs: Tensor,
        targets: Tensor,
        postprocess_fn: PostProcessingFunction,
        score_fn: Optional[ScoreFunction] = None,
        **kwargs
    ) -> Tensor:

        Saliency.compute_score(
            postprocess_fn, score_fn, self.task, self.model, inputs, targets
        )

        explanation = inputs.grad.abs()

        if len(explanation.shape) == 4:
            explanation = explanation.permute((0, 2, 3, 1))

        return explanation
