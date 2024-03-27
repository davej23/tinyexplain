import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Tensor, nn

from tinyexplain.explain import Rise
from tinyexplain.extras.models import TimeSeriesModel
from tinyexplain.extras.utils import load_mock_sine_dataset
from tinyexplain.types import TinyExplainTask

model = TimeSeriesModel()
nn.state.load_state_dict(model, nn.state.safe_load("examples/models/saved/time_series_sine.bin"))


x, y = load_mock_sine_dataset()


explainer = Rise(model, TinyExplainTask.TIME_SERIES, 100, (1, 8))
explanation = explainer.explain(x, y, lambda x: x)
explanation = explanation[0].numpy()


explanation = (explanation - np.min(explanation)) / (np.max(explanation) - np.min(explanation))
fig, axs = plt.subplots(2)
axs[0].imshow(x[0].numpy())
axs[1].imshow(explanation)
plt.show()
