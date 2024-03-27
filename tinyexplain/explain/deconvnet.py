from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import (PostProcessingFunction, ScoreFunction,
                               TinyExplainTask, TinygradModel)
from tinyexplain.utils.overrides import (overwrite_backward, overwrite_relu,
                                         revert_backward, revert_relu)

from .explainer import Explainer


class DeconvNet(Explainer):
    """DeconvNet Explainer"""
    def __init__(self, model: TinygradModel, task: TinyExplainTask):
        super().__init__()
        self.model = model
        self.task = task
        self.gbp = False

    def explain(
        self,
        inputs: Tensor,
        targets: Tensor,
        postprocess_fn: PostProcessingFunction,
        score_fn: Optional[ScoreFunction] = None,
        **kwargs
    ) -> Tensor:

        inputs_c = Tensor(inputs.numpy())

        # Find ReLUs in model
        overwrite_relu()
        self.model(inputs_c)
        revert_relu()  # revert Tensor.relu()

        # ReLU gradients from ReLU
        overwrite_backward(gbp=self.gbp)

        DeconvNet.compute_score(
            postprocess_fn, score_fn, self.task, self.model, inputs, targets
        )

        explanation = inputs.grad

        if len(explanation.shape) == 4:
            explanation = explanation.permute((0, 2, 3, 1))

        revert_backward()  # revert Tensor.backward

        return explanation
