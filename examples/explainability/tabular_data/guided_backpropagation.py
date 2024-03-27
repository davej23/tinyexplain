import cv2
import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Tensor, nn

from tinyexplain.explain import GuidedBackpropagation
from tinyexplain.extras.models import TabularDataModel
from tinyexplain.types import TinyExplainTask

model = TabularDataModel()
nn.state.load_state_dict(model, nn.state.safe_load("examples/models/saved/tabular_data_titanic.bin"))


x, y = Tensor([22.0, 7.5, 1.0, 0.0]).reshape((1, 1, 4)), Tensor([0.0]).reshape((1, 1, 1))


explainer = GuidedBackpropagation(model, TinyExplainTask.TABULAR_DATA)
explanation = explainer.explain(x, y, lambda x: x, lambda x, y: (x-y).square().mean())
explanation = explanation[0].numpy()


explanation = (explanation - np.min(explanation)) / (np.max(explanation) - np.min(explanation))
fig, axs = plt.subplots(2)
axs[0].imshow(x[0].numpy())
axs[1].imshow(explanation)
plt.show()
