from typing import Optional

from tinygrad import Tensor

from tinyexplain.types import (PostProcessingFunction, ScoreFunction,
                               TinyExplainTask, TinygradModel)

from .explainer import Explainer


class IntegratedGradients(Explainer):
    """IntegratedGradients Explainer"""
    def __init__(self, model: TinygradModel, task: TinyExplainTask, steps: int = 100):
        super().__init__()
        self.model = model
        self.task = task
        self.steps = steps

    def explain(
        self,
        inputs: Tensor,
        targets: Tensor,
        postprocess_fn: PostProcessingFunction,
        score_fn: Optional[ScoreFunction] = None,
        **kwargs
    ) -> Tensor:

        exp_steps = []

        interpolated_images = [
            Tensor.stack([inputs[i] * (j + 1) / self.steps for j in range(self.steps)])
            for i in range(inputs.shape[0])
        ]
        for interpolated_image in interpolated_images:
            interpolated_image.requires_grad = True
            interpolated_image.to("CUDA")

            IntegratedGradients.compute_score(
                postprocess_fn,
                score_fn,
                self.task,
                self.model,
                interpolated_image,
                (
                    targets.repeat((self.steps, 1, 1))
                    if self.task is TinyExplainTask.OBJECT_DETECTION
                    else targets
                ),
            )

            input_gradients = interpolated_image.grad.mean(0, keepdim=True)
            exp_steps.append(input_gradients)

        explanation = inputs * Tensor.stack(exp_steps).mean(0)

        if len(explanation.shape) == 4:
            explanation = explanation.permute((0, 2, 3, 1))

        return explanation
