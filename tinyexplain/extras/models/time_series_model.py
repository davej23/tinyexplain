from tinygrad import Tensor, nn


class TimeSeriesModel:
    def __init__(
        self, sequence_length: int = 16, filters: int = 32, kernel_size: int = 2
    ):
        self.l1 = nn.Conv1d(1, filters, kernel_size)
        self.l2 = nn.Conv1d(filters, 1, kernel_size)
        self.l3 = nn.Linear(14, sequence_length)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
