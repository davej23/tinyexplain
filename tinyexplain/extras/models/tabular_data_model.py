from tinygrad import Tensor, nn


class TabularDataModel:
    def __init__(self, in_features: int = 4, out_features: int = 1):
        self.l1 = nn.Linear(in_features, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, out_features)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
