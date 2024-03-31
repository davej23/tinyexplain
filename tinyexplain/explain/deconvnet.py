from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import PostProcessingFunction, ScoreFunction, TinyExplainTask, TinygradModel
from tinyexplain.logging import Logger
from tinyexplain.utils.overrides import overwrite_backward, overwrite_relu, revert_backward, revert_relu

from .explainer import Explainer


class DeconvNet(Explainer):
    """DeconvNet Explainer"""

    def __init__(self, model: TinygradModel, task: TinyExplainTask):
        super().__init__()
        self.model = model
        self.task = task
        self.gbp = False

    def explain(
        self, inputs: Tensor, targets: Tensor, postprocess_fn: PostProcessingFunction,
        score_fn: Optional[ScoreFunction] = None, device: str = "CUDA", **kwargs
    ) -> Tensor:

        Logger.debug(f"{self._log_prefix} {inputs=} {targets=}")

        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs_c = Tensor(inputs.numpy()).to(device)

        Logger.debug(f"{self._log_prefix} Overwriting ReLU")
        overwrite_relu()
        self.model(inputs_c)
        Logger.debug(f"{self._log_prefix} Reverting ReLU")
        revert_relu()

        Logger.debug(f"{self._log_prefix} Overwriting Tensor.backward")
        overwrite_backward(gbp=self.gbp)

        Logger.debug(f"{self._log_prefix} Running score computation")
        DeconvNet.compute_score(postprocess_fn, score_fn, self.task, self.model, inputs, targets)

        Logger.debug(f"{self._log_prefix} Extracting gradients")
        explanations = inputs.grad

        if len(explanations.shape) == 4:
            explanations = explanations.permute((0, 2, 3, 1))

        Logger.debug(f"{self._log_prefix} Revert Tensor.backward")
        revert_backward()

        Logger.debug(f"{self._log_prefix} {explanations=}")

        return explanations
