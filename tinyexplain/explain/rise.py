from typing import Optional

import cv2
from tinygrad import Tensor
from tqdm import tqdm

from tinyexplain.types import PostProcessingFunction, ScoreFunction, TinyExplainTask, TinygradModel
from tinyexplain.utils.logging import Logger

from .explainer import Explainer


class Rise(Explainer):
    """Rise Explainer"""

    def __init__(
        self,
        model: TinygradModel,
        task: TinyExplainTask,
        samples: int,
        random_mask_shape: tuple[int, int],
    ):
        """Rise Explainer"""
        super().__init__()
        self.model = model
        self.task = task
        self.samples = samples
        self.random_mask_shape = random_mask_shape

    def explain(
        self, inputs: Tensor, targets: Tensor, postprocess_fn: PostProcessingFunction, score_fn: Optional[ScoreFunction] = None, **kwargs
    ) -> Tensor:

        Logger.debug(f"{self._log_prefix} {inputs=} {targets=}")

        explanations = Tensor.zeros((inputs.shape[0], *inputs.shape[len(inputs.shape) - 2 :]))

        Logger.debug(f"{self._log_prefix} {explanations=}")

        for _ in tqdm(range(self.samples)):
            mask = Rise._generate_mask(self.random_mask_shape, inputs.shape[len(inputs.shape) - 2 :])

            Logger.debug(f"{self._log_prefix} {mask=}")

            masked_inputs = inputs * mask.repeat(
                (inputs.shape[0], inputs.shape[1], 1, 1) if len(inputs.shape) == 4 else (inputs.shape[0], inputs.shape[1], 1)
            )

            Logger.debug(f"{self._log_prefix} {masked_inputs=}")

            Logger.debug(f"{self._log_prefix} Running score computation")
            score = Rise.compute_score(
                postprocess_fn,
                score_fn,
                self.task,
                self.model,
                masked_inputs,
                targets,
                requires_grad=False,
            )

            mask = mask.unsqueeze(0)
            explanations += (score * mask).sum(0) / (mask.sum(0) + 1e6)

        Logger.debug(f"{self._log_prefix} {explanations=}")
        return explanations

    @staticmethod
    def _generate_mask(
        random_mask_shape: tuple[int, int],
        upscale_mask_shape: tuple[int, int],
        prob_threshold: float = 0.5,
    ) -> Tensor:
        mask = Tensor.rand(random_mask_shape)
        mask = Tensor(cv2.resize(mask.numpy(), upscale_mask_shape).T)  # pylint: disable=no-member
        return mask > prob_threshold
