from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import PostProcessingFunction, ScoreFunction, TinyExplainTask, TinygradModel
from tinyexplain.logging import Logger

from .explainer import Explainer


class IntegratedGradients(Explainer):
    """IntegratedGradients Explainer"""

    def __init__(self, model: TinygradModel, task: TinyExplainTask, steps: int = 100):
        super().__init__()
        self.model = model
        self.task = task
        self.steps = steps

    def explain(
        self, inputs: Tensor, targets: Tensor, postprocess_fn: PostProcessingFunction,
        score_fn: Optional[ScoreFunction] = None, device: str = "CUDA", **kwargs
    ) -> Tensor:

        Logger.debug(f"{self._log_prefix} {inputs=} {targets=}")

        inputs = inputs.to(device)
        targets = targets.to(device)

        exp_steps = []

        Logger.debug(f"{self._log_prefix} Getting interpolate images")
        interpolated_images = [Tensor.stack([inputs[i] * (j + 1) / self.steps for j in range(self.steps)]) for i in range(inputs.shape[0])]
        for i, interpolated_image in enumerate(interpolated_images):
            interpolated_image.requires_grad = True
            interpolated_image.to("CUDA")

            Logger.debug(f"{self._log_prefix} Running interpolated image {i+1}/{len(interpolated_images)}")

            Logger.debug(f"{self._log_prefix} Running score computation")
            IntegratedGradients.compute_score(
                postprocess_fn,
                score_fn,
                self.task,
                self.model,
                interpolated_image,
                (targets.repeat((self.steps, 1, 1)) if self.task is TinyExplainTask.OBJECT_DETECTION else targets),
            )

            input_gradients = interpolated_image.grad.mean(0, keepdim=True)
            Logger.debug(f"{self._log_prefix} Extracted gradients {input_gradients}")
            exp_steps.append(input_gradients)

        explanations = inputs * Tensor.stack(exp_steps).mean(0)

        if len(explanations.shape) == 4:
            explanations = explanations.permute((0, 2, 3, 1))

        Logger.debug(f"{self._log_prefix} {explanations=}")

        return explanations
