from typing import Any, Callable, Optional

from skimage.segmentation import felzenszwalb
from sklearn.linear_model import Ridge
from tinygrad import Tensor

from tinyexplain.types import PostProcessingFunction, ScoreFunction, TinyExplainTask, TinygradModel
from tinyexplain.logging import Logger

from .explainer import Explainer


class Lime(Explainer):
    """Lime Explainer"""

    def __init__(
        self,
        model: TinygradModel,
        task: TinyExplainTask,
        samples: int,
        weighting_kernel: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None,
        perturbation_function: Optional[Callable[[int, int], Tensor]] = None,
        interpret_map: Optional[Callable[[Tensor], Tensor]] = None,
        interpretable_model: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.task = task
        self.samples = samples

        self.sim_kernel: Callable[[Tensor, Tensor, Tensor], Tensor] = (
            Lime._get_euclidean_weighting_kernel() if weighting_kernel is None else weighting_kernel
        )
        self.pert_function: Callable[[Any], Tensor] = (
            Lime._default_perturbation_function if perturbation_function is None else perturbation_function
        )  # type: ignore[assignment]
        self.seg_fn = Lime._get_segmentation_function(self.task) if interpret_map is None else interpret_map
        self.interpretable_model = Ridge(2) if interpretable_model is None else interpretable_model

    def explain(
        self, inputs: Tensor, targets: Tensor, postprocess_fn: PostProcessingFunction,
        score_fn: Optional[ScoreFunction] = None, device: str = "CUDA", **kwargs
    ) -> Tensor:

        Logger.debug(f"{self._log_prefix} {inputs=} {targets=}")

        inputs = inputs.to(device)
        targets = targets.to(device)

        explanations = []

        for inp, targ in zip(inputs, targets):
            inp = inp.unsqueeze(0)
            targ = targ.unsqueeze(0)

            Logger.debug(f"{self._log_prefix} Running segmentation function")
            mapper = self.seg_fn(inp)
            num_features = (mapper.max() + Tensor(1)).numpy().item()

            Logger.debug(f"{self._log_prefix} Running perturbation function")
            batch_interpolated_samples = self.pert_function(num_features, self.samples)  # type: ignore[call-arg]

            batch_interpolated_samples = batch_interpolated_samples.to(device)

            perturbed_targets = []
            similarities: list[Tensor] = []
            for interpolated_samples in batch_interpolated_samples:
                interpolated_samples = interpolated_samples.unsqueeze(0)
                Logger.debug(f"{self._log_prefix} Getting mask from interpolated sample")
                masks = interpolated_samples[:, mapper]

                Logger.debug(f"{self._log_prefix} Generating perturbed samples")
                perturbed_samples = Lime._apply_masks(inp[0], masks).unsqueeze(0)

                perturbed_samples = perturbed_samples.to(device)

                Logger.debug(f"{self._log_prefix} Running score computation")
                score = Lime.compute_score(
                    postprocess_fn,
                    score_fn,
                    self.task,
                    self.model,
                    perturbed_samples,
                    targets,
                )

                perturbed_targets.append(score)

                Logger.debug(f"{self._log_prefix} Running similarity kernel")
                similarities.append(self.sim_kernel(inp, interpolated_samples, perturbed_samples))

            perturbed_targets_t = Tensor.stack(perturbed_targets)
            similarities_t = Tensor.stack(similarities)

            Logger.debug(f"{self._log_prefix} {perturbed_targets_t=} {similarities_t=}")

            if similarities_t.max().numpy() == 0 and similarities_t.min().numpy() == 0:
                Logger.info(f"{self._log_prefix} Similarities are zeros")
                explanation = Tensor.zeros(masks.shape)[0]
            else:
                Logger.debug(f"{self._log_prefix} Fitting interpretable model")
                self.interpretable_model.fit(
                    batch_interpolated_samples.numpy(),
                    perturbed_targets_t.numpy(),
                    sample_weight=similarities_t.numpy(),
                )

                explanation = Tensor(self.interpretable_model.coef_).unsqueeze(0)

                explanation = explanation[:, mapper][0]

                Logger.debug(f"{self._log_prefix} {explanation=}")

            explanations.append(explanation)

        explanations = Tensor.stack(explanations)
        Logger.debug(f"{self._log_prefix} {explanations=}")
        return explanations

    @staticmethod
    def _image_segmentation(x: Tensor) -> Tensor:
        return Tensor(felzenszwalb(x[0].permute((1, 2, 0)).numpy()))

    @staticmethod
    def _get_segmentation_function(task: TinyExplainTask) -> Callable[[Tensor], Tensor]:
        if task in [
            TinyExplainTask.OBJECT_DETECTION,
            TinyExplainTask.IMAGE_CLASSIFICATION,
            TinyExplainTask.SEMANTIC_SEGMENTATION,
        ]:
            Logger.debug(f"_get_segmentation_function: Found computer vision task, returning Lime._image_segmentation")
            return Lime._image_segmentation

        # TODO add other tasks
        return lambda x: Tensor(list(range(x.flatten().shape[0])))

    @staticmethod
    def _default_perturbation_function(num_features: int, num_samples: int) -> Tensor:
        Logger.debug(f"_default_perturbation_function: {num_features=} {num_samples=}")
        probs = Tensor.ones(num_features) * 0.5
        sampling = Tensor.uniform((num_samples, num_features))
        return Tensor(probs.numpy() > sampling.numpy())

    @staticmethod
    def _apply_masks(x: Tensor, masks: Tensor) -> Tensor:
        Logger.debug(f"_apply_masks: {x=} {masks=}")
        rep_x = x.repeat((masks.shape[0], 1, 1) if len(masks.shape) == 3 else (masks.shape[0], 1))
        return rep_x * masks

    @staticmethod
    def _get_euclidean_weighting_kernel(
        kernel_width: int = 45,
    ) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        Logger.debug(f"_get_euclidean_weighting_kernel: {kernel_width=}")

        def _euclidean_weighting_kernel(inputs: Tensor, interpolated_samples: Tensor, perturbed_samples: Tensor) -> Tensor:
            rep_x = inputs.repeat((interpolated_samples.shape[0], 1, 1, 1))
            flat_xs = rep_x.flatten()
            flat_samps = perturbed_samples.flatten()

            delta = flat_xs - flat_samps
            distances = (delta**2).sum().sqrt()
            similarities = (-1.0 * (distances**2) / kernel_width**2).exp()

            return similarities

        return _euclidean_weighting_kernel
