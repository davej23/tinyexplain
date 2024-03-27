import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Tensor, nn

from tinyexplain.explain import Rise
from tinyexplain.extras.models import FashionMnistLinearClassifier
from tinyexplain.extras.utils import load_fashion_mnist_dataset
from tinyexplain.types import TinyExplainTask

# Load trained Fashion MNIST classifier
model = FashionMnistLinearClassifier()
nn.state.load_state_dict(model, nn.state.safe_load("examples/models/saved/fashion_mnist_linear_classifier.bin"))


# Load dataset
train_images, train_labels = load_fashion_mnist_dataset()
train_images = train_images.reshape((train_labels.shape[0], 1, 28, 28))


explainer = Rise(model, TinyExplainTask.IMAGE_CLASSIFICATION, 100, (10, 10))
explanation = explainer.explain(train_images[0].unsqueeze(0), train_labels[0].unsqueeze(0), lambda x: x)
explanation = explanation[0].reshape((28, 28)).numpy()


explanation = (explanation - np.min(explanation)) / (np.max(explanation) - np.min(explanation))
fig, axs = plt.subplots(2)
axs[0].imshow(train_images[0][0].numpy())
axs[1].imshow(explanation)
plt.show()
