import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Tensor

from tinyexplain.explain import Rise
from tinyexplain.extras.models import YOLOv8
from tinyexplain.extras.utils import yolov8_tbo
from tinyexplain.types import TinyExplainTask

NC = 10
BS = 1

model = YOLOv8(2, 2, 2, NC)
x = Tensor.rand((BS, 3, 640, 640))
y = Tensor.rand((BS, 4 + NC, 1))


explainer = Rise(model, TinyExplainTask.OBJECT_DETECTION, 100, (10, 10))
explanation = explainer.explain(x, y, yolov8_tbo)
explanation = explanation[0].numpy()


explanation = (explanation - np.min(explanation)) / (np.max(explanation) - np.min(explanation))
plt.imshow(explanation)
plt.show()
