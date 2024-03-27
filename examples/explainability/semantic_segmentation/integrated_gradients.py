import cv2
import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Tensor, nn

from tinyexplain.explain import IntegratedGradients
from tinyexplain.extras.models import UNet
from tinyexplain.extras.utils import process_voc_image, process_voc_label
from tinyexplain.types import TinyExplainTask

# Load trained UNet
model = UNet(out_classes=20)
nn.state.load_state_dict(model, nn.state.safe_load("examples/models/saved/pascal_voc_unet.bin"))


# Load test image and label
image = process_voc_image("examples/data/pascal_voc/2007_000032.jpg")
label = process_voc_label("examples/data/pascal_voc/2007_000032.png")


explainer = IntegratedGradients(model, TinyExplainTask.SEMANTIC_SEGMENTATION, 10)
explanation = explainer.explain(image, label, lambda x: x, lambda x, y: (x-y).square().mean())
explanation = explanation[0].numpy()
explanation = np.array(cv2.resize(explanation, (512, 512)))


explanation = (explanation - np.min(explanation)) / (np.max(explanation) - np.min(explanation))
fig, axs = plt.subplots(2)
axs[0].imshow(image[0].permute((1, 2, 0)).numpy())
axs[1].imshow(explanation)
plt.show()
