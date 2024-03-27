import numpy as np
from sklearn.linear_model import LinearRegression
from tinygrad import Tensor, dtypes

from tinyexplain.types import TinyExplainTask, TinygradModel

from .lime import Lime


class KernelShap(Lime):
    """KernelShap Explainer"""
    def __init__(self, model: TinygradModel, task: TinyExplainTask, samples: int):
        super().__init__(
            model,
            task,
            samples,
            interpretable_model=LinearRegression(),
            weighting_kernel=KernelShap._kernel_shap_weighting_kernel,
            perturbation_function=KernelShap._kernel_shap_perturbation_function,
        )

    @staticmethod
    def _kernel_shap_perturbation_function(
        num_features: int, num_samples: int
    ) -> Tensor:
        # TODO this might not be quite right
        # feature_probs = KernelShap._get_feature_probabilities(num_features)
        sample_idxs = Tensor.randint((num_samples,), high=num_features)

        sample_idxs = sample_idxs.one_hot(num_features)

        random_normal_values = Tensor.normal((num_samples, num_features))
        idx_sorted_values = Tensor(np.argsort(random_normal_values.numpy(), axis=1))

        threshold_idx = idx_sorted_values * sample_idxs
        threshold_idx = threshold_idx.sum(1)
        threshold = random_normal_values * threshold_idx.one_hot(num_features)
        threshold = threshold.sum(1)
        threshold = threshold.unsqueeze(1)
        threshold = threshold.repeat((1, num_features))
        int_samples = Tensor(random_normal_values.numpy() > threshold.numpy())

        return int_samples

    @staticmethod
    def _get_feature_probabilities(n_features: int) -> Tensor:
        list_features_indexes = Tensor(list(range(1, n_features)))
        denom = list_features_indexes * (n_features - list_features_indexes)
        num = n_features - 1
        probs = num / denom
        probs = Tensor([0.0]).cat(probs)
        return probs.cast(dtypes.float32)

    @staticmethod
    def _kernel_shap_weighting_kernel(
        inputs: Tensor, interpolated_samples: Tensor, perturbed_samples: Tensor  # pylint: disable=unused-argument
    ) -> Tensor:
        return Tensor.ones(interpolated_samples.shape[0], dtype=dtypes.float32).sum(0)
