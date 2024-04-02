import cv2
import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Tensor, nn

from tinyexplain.explain import GradCam
from tinyexplain.extras.models import FashionMnistCnnClassifier
from tinyexplain.extras.utils import load_fashion_mnist_dataset
from tinyexplain.types import TinyExplainTask

# Load trained Fashion MNIST classifier
model = FashionMnistCnnClassifier()
nn.state.load_state_dict(model, nn.state.safe_load("examples/models/saved/fashion_mnist_cnn_classifier.bin"))


# Load dataset
train_images, train_labels = load_fashion_mnist_dataset()
train_images = train_images.reshape((train_labels.shape[0], 1, 28, 28))
train_images = train_images.repeat((1, 3, 1, 1))


explainer = GradCam(model, TinyExplainTask.IMAGE_CLASSIFICATION, "l2")
explanation = explainer.explain(
    train_images[0].unsqueeze(0), train_labels[0].unsqueeze(0), lambda x: x, lambda x, y: (x - y).square().mean()
)
explanation = explanation[0].numpy()
explanation = np.array(cv2.resize(explanation, (train_images.shape[2], train_images.shape[3])))

explanation = (explanation - np.min(explanation)) / (np.max(explanation) - np.min(explanation))
fig, axs = plt.subplots(2)
axs[0].imshow(train_images[0][0].numpy())
axs[1].imshow(explanation)
plt.show()
