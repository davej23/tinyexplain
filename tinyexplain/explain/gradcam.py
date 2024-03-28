from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import PostProcessingFunction, ScoreFunction, TinyExplainTask, TinygradModel
from tinyexplain.utils.logging import Logger
from tinyexplain.utils.overrides import TINYGRAD_LAYERS, get_layer, overwrite_call, revert_call

from .explainer import Explainer

import cv2


class GradCam(Explainer):
    """GradCam Explainer"""

    def __init__(self, model: TinygradModel, task: TinyExplainTask, conv_layer: str):
        super().__init__()
        self.model = model
        self.task = task

        # Get layer in model
        self.conv_layer = get_layer(conv_layer, self.model)

        if not isinstance(self.conv_layer, tuple(TINYGRAD_LAYERS)):
            raise ValueError(f"layer {self.conv_layer} is not a Tinygrad layer: {TINYGRAD_LAYERS}")

    def explain(
        self,
        inputs: Tensor,
        targets: Tensor,
        postprocess_fn: PostProcessingFunction,
        score_fn: Optional[ScoreFunction] = None,
        **kwargs,
    ) -> Tensor:

        Logger.debug(f"{self._log_prefix} {inputs=} {targets=}")

        Logger.debug(f"{self._log_prefix} Overwriting {type(self.conv_layer).__call__}")
        overwrite_call(type(self.conv_layer))

        Logger.debug(f"{self._log_prefix} Running score computation")
        GradCam.compute_score(postprocess_fn, score_fn, self.task, self.model, inputs, targets)

        Logger.debug(f"{self._log_prefix} Extracting last layer call")
        output = self.conv_layer.last_call

        Logger.debug(f"{self._log_prefix} Last layer call {output}")

        Logger.debug(f"{self._log_prefix} Extracting gradients")
        feature_maps = output
        gradients = output.grad

        if len(feature_maps.shape) == 4:
            feature_maps = feature_maps.permute((0, 2, 3, 1))

        if len(gradients.shape) == 4:
            gradients = gradients.permute((0, 2, 3, 1))

        mean_gradients = gradients.mean((1, 2) if len(gradients.shape) == 4 else 1, keepdim=True)

        Logger.debug(f"{self._log_prefix} Feature maps {feature_maps}")
        Logger.debug(f"{self._log_prefix} Mean gradients {mean_gradients}")

        if len(gradients.shape) == 4:
            grad_cams = (mean_gradients * feature_maps).sum(-1 if len(gradients.shape) == 4 else 1).relu()
        else:
            grad_cams = (mean_gradients * feature_maps).relu()

        Logger.debug(f"{self._log_prefix} Reverting {type(self.conv_layer).__call__}")
        revert_call(type(self.conv_layer))

        grad_cams = Tensor.stack([Tensor(cv2.resize(grad_cam.numpy(), inputs.shape[len(inputs.shape) - 2 :]).T) for grad_cam in grad_cams])
        Logger.debug(f"{self._log_prefix} GradCams {grad_cams}")
        return grad_cams
