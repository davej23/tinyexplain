from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import (PostProcessingFunction, ScoreFunction,
                               TinyExplainTask, TinygradModel)
from tinyexplain.utils.overrides import (TINYGRAD_LAYERS, get_layer,
                                         overwrite_call, revert_call)

from .explainer import Explainer


class GradCam(Explainer):
    """GradCam Explainer"""
    def __init__(self, model: TinygradModel, task: TinyExplainTask, conv_layer: str):
        super().__init__()
        self.model = model
        self.task = task

        # Get layer in model
        self.conv_layer = get_layer(conv_layer, self.model)

        if not isinstance(self.conv_layer, tuple(TINYGRAD_LAYERS)):
            raise ValueError(
                f"layer {self.conv_layer} is not a Tinygrad layer: {TINYGRAD_LAYERS}"
            )

    def explain(
        self,
        inputs: Tensor,
        targets: Tensor,
        postprocess_fn: PostProcessingFunction,
        score_fn: Optional[ScoreFunction] = None,
        **kwargs,
    ) -> Tensor:

        # Overwrite call dunder of layer class
        overwrite_call(type(self.conv_layer))

        GradCam.compute_score(
            postprocess_fn, score_fn, self.task, self.model, inputs, targets
        )

        output = self.conv_layer.last_call

        feature_maps = output
        gradients = output.grad

        if len(feature_maps.shape) == 4:
            feature_maps = feature_maps.permute((0, 2, 3, 1))

        if len(gradients.shape) == 4:
            gradients = gradients.permute((0, 2, 3, 1))

        mean_gradients = gradients.mean(
            (1, 2) if len(gradients.shape) == 4 else 1, keepdim=True
        )

        if len(gradients.shape) == 4:
            grad_cams = (
                (mean_gradients * feature_maps)
                .sum(-1 if len(gradients.shape) == 4 else 1)
                .relu()
            )
        else:
            grad_cams = (mean_gradients * feature_maps).relu()

        revert_call(type(self.conv_layer))  # revert layer .call

        return grad_cams
