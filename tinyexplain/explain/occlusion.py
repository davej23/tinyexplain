from typing import Optional

import numpy as np
from tinygrad import Tensor, dtypes
from tqdm import tqdm

from tinyexplain.types import PostProcessingFunction, ScoreFunction, TinyExplainTask, TinygradModel
from tinyexplain.utils.logging import Logger

from .explainer import Explainer


class Occlusion(Explainer):
    """Occlusion Explainer"""

    def __init__(
        self,
        model: TinygradModel,
        task: TinyExplainTask,
        patch_size: tuple[int, int],
        patch_stride: tuple[int, int],
    ):
        super().__init__()
        self.model = model
        self.task = task
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.postprocess_fn: PostProcessingFunction = None  # type: ignore[assignment]
        self.score_fn: ScoreFunction = None  # type: ignore[assignment]

    def explain(
        self, inputs: Tensor, targets: Tensor, postprocess_fn: PostProcessingFunction, score_fn: Optional[ScoreFunction] = None, **kwargs
    ) -> Tensor:

        Logger.debug(f"{self._log_prefix} {inputs=} {targets=}")

        if score_fn is not None:
            self.score_fn = score_fn
        self.postprocess_fn = postprocess_fn

        Logger.debug(f"{self._log_prefix} {self.score_fn=} {self.postprocess_fn=}")

        x_stride_idxs = [x * self.patch_stride[0] for x in range(int((inputs.shape[2] - self.patch_size[0] + 1) / self.patch_stride[0]))]

        Logger.debug(f"{self._log_prefix} {x_stride_idxs=}")

        if self.task in [
            TinyExplainTask.IMAGE_CLASSIFICATION,
            TinyExplainTask.OBJECT_DETECTION,
            TinyExplainTask.SEMANTIC_SEGMENTATION,
        ]:
            y_stride_idxs = [
                y * self.patch_stride[1] for y in range(int((inputs.shape[3] - self.patch_size[1] + 1) / self.patch_stride[1]))
            ]
            return self._2d_occlusion(inputs, targets, x_stride_idxs, y_stride_idxs)

        return self._1d_occlusion(inputs, targets, x_stride_idxs)

    def _2d_occlusion(
        self,
        inputs: Tensor,
        targets: Tensor,
        x_stride_idxs: list[int],
        y_stride_idxs: list[int],
    ) -> Tensor:
        Logger.debug(f"{self._log_prefix} Running 2D Occlusion {inputs=} {targets=}")
        explanations = Tensor.zeros((inputs.shape[0], *inputs.shape[2:])).to("CUDA")
        for x_stride_idx in tqdm(x_stride_idxs):
            for y_stride_idx in y_stride_idxs:
                mask = np.zeros((inputs.shape[0], *inputs.shape[2:]))
                mask[
                    :,
                    x_stride_idx : x_stride_idx + self.patch_size[0],
                    y_stride_idx : y_stride_idx + self.patch_size[1],
                ] = 1
                mask = Tensor(mask).to("CUDA").cast(dtypes.float32)

                score = Occlusion.compute_score(
                    self.postprocess_fn,
                    self.score_fn,
                    self.task,
                    self.model,
                    inputs * mask,
                    targets,
                    requires_grad=False,
                )

                explanations += (1.0 - score) * mask

        Logger.debug(f"{self._log_prefix} {explanations=}")
        return explanations

    def _1d_occlusion(self, inputs: Tensor, targets: Tensor, stride_idxs: list[int]) -> Tensor:
        Logger.debug(f"{self._log_prefix} Running 1D Occlusion {inputs=} {targets=}")
        explanations = Tensor.zeros(inputs.shape).to("CUDA")
        for stride_idx in tqdm(stride_idxs):
            mask = np.zeros(inputs.shape)
            mask[:, :, stride_idx : stride_idx + self.patch_size[0]] = 1
            mask = Tensor(mask).to("CUDA").cast(dtypes.float32)
            out = self.model(inputs * mask)
            out = self.postprocess_fn(out)

            score = Occlusion.compute_score(
                self.postprocess_fn,
                (TinyExplainTask.TABULAR_DATA.score_fn if self.score_fn is None else self.score_fn),
                self.task,
                self.model,
                inputs * mask,
                targets,
                requires_grad=False,
            )

            explanations += (1.0 - score) * mask

        Logger.debug(f"{self._log_prefix} {explanations=}")
        return explanations
