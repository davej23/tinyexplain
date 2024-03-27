from tinygrad import Tensor, nn


class FashionMnistLinearClassifier:
    def __init__(self):
        self.l1 = nn.Linear(784, 64)
        self.l2 = nn.Linear(64, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        x = self.l1(x)
        x = self.l2(x)
        return x


class FashionMnistCnnClassifier:
    def __init__(self):
        self.l1 = nn.Conv2d(3, 16, 2, 2)
        self.l2 = nn.Conv2d(16, 64, 2, 2)
        self.l3 = lambda x: x.flatten(1)
        self.l4 = nn.Linear(64 * 7 * 7, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x
