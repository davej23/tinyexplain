from tinygrad import Tensor, nn

from tinyexplain.extras.models import TimeSeriesModel
from tinyexplain.utils import count_model_parameters

model = TimeSeriesModel()
print("model param count: ", count_model_parameters(model))

SEQUENCE_LENGTH = 16
BATCH_SIZE = 32

data = Tensor(list(range(100000))).sin()


def create_generator():
    x_batch = None
    y_batch = None

    for i in range(data.shape[0] - SEQUENCE_LENGTH):
        if x_batch is not None and x_batch.shape[0] % BATCH_SIZE == 0:
            yield x_batch.unsqueeze(1), y_batch.unsqueeze(1)
            x_batch = None
            y_batch = None
        else:
            x = data[i : i + SEQUENCE_LENGTH].unsqueeze(0)
            y = data[i + 1 : i + SEQUENCE_LENGTH + 1].unsqueeze(0)
            x_batch = x if x_batch is None else x_batch.cat(x)
            y_batch = y if y_batch is None else y_batch.cat(y)

    yield x_batch.unsqueeze(1), y_batch.unsqueeze(1)


optim = nn.optim.Adam(nn.state.get_parameters(model))

gen = create_generator()

for step in range(100):
    optim.zero_grad()

    x, y = next(gen)
    x.requires_grad = True
    y.requires_grad = True

    out = model(x)
    loss = (out - y).square().mean()
    print(f"{step=} loss={loss.numpy()}")

    loss.backward()
    optim.step()

nn.state.safe_save(nn.state.get_state_dict(model), "time_series_sine.bin")
