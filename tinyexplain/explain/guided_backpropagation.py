from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import PostProcessingFunction, ScoreFunction, TinyExplainTask, TinygradModel

from .deconvnet import DeconvNet


class GuidedBackpropagation(DeconvNet):
    """GuidedBackpropagation Explainer"""

    def __init__(self, model: TinygradModel, task: TinyExplainTask):
        super().__init__(model, task)
        self.gbp = True

    def explain(
        self, inputs: Tensor, targets: Tensor, postprocess_fn: PostProcessingFunction, score_fn: Optional[ScoreFunction] = None, **kwargs
    ) -> Tensor:
        return super().explain(inputs, targets, postprocess_fn, score_fn)
