import gzip

import numpy as np
from tinygrad import Tensor, dtypes, nn

from tinyexplain.extras.models import FashionMnistCnnClassifier

train_images = "../fashion-mnist/data/fashion/train-images-idx3-ubyte.gz"
train_labels = "../fashion-mnist/data/fashion/train-labels-idx1-ubyte.gz"


with gzip.open(train_labels, "rb") as labels:
    train_labels = np.frombuffer(labels.read(), dtype=np.uint8, offset=8)

with gzip.open(train_images, "rb") as images:
    train_images = np.frombuffer(images.read(), dtype=np.uint8, offset=16).reshape((len(train_labels), 784))


train_images = Tensor(train_images).reshape((len(train_labels), 1, 28, 28))
train_images = train_images.repeat((1, 3, 1, 1)).to("CUDA")
train_labels = Tensor(train_labels).to("CUDA")


model = FashionMnistCnnClassifier()
optim = nn.optim.Adam(nn.state.get_parameters(model), lr=0.001)

for step in range(100):
    optim.zero_grad()
    out = model(train_images)
    loss = out.sparse_categorical_crossentropy(train_labels)
    loss.backward()
    optim.step()
    print("loss ", loss.numpy())

nn.state.safe_save(nn.state.get_state_dict(model), "fashion_mnist_cnn_classifier.bin")
