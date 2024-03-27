import csv
import random
from dataclasses import dataclass, fields

from tinygrad import Tensor, dtypes, nn

from tinyexplain.extras.models import TabularDataModel
from tinyexplain.utils import count_model_parameters


@dataclass
class TitanicData:
    passenger_id: int
    survived: int
    p_class: int
    name: str
    sex: str
    age: float
    sib_sp: int
    parch: int
    ticket: str
    fare: float
    cabin: str
    embarked: str

    @classmethod
    def _load(cls, data: list[str]) -> "TitanicData":
        memtypes = [n.type for n in fields(cls)]
        return cls(**dict(zip([n.name for n in fields(cls)], [dtype(d) for dtype, d in zip(memtypes, data)])))


model = TabularDataModel()
print("model param count: ", count_model_parameters(model))

BATCH_SIZE = 32


def create_generator():
    x_batch = None
    y_batch = None

    with open("examples/data/titanic/data.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        reader = [n for n in reader if "" not in n][1:]
        random.shuffle(reader)
        for row in reader:
            rowdata = TitanicData._load(row)
            rowdata.sex = 1 if rowdata.sex == "female" else 0
            x = Tensor([rowdata.age, rowdata.fare, rowdata.p_class, rowdata.sex]).cast(dtypes.float32).unsqueeze(0)
            y = Tensor([rowdata.survived]).cast(dtypes.float32).unsqueeze(0)
            if x_batch is not None and x_batch.shape[0] % BATCH_SIZE == 0:
                yield x_batch.unsqueeze(1), y_batch.unsqueeze(1)
                x_batch = None
                y_batch = None
            else:
                x_batch = x if x_batch is None else x_batch.cat(x)
                y_batch = y if y_batch is None else y_batch.cat(y)

        yield x_batch.unsqueeze(1), y_batch.unsqueeze(1)


optim = nn.optim.Adam(nn.state.get_parameters(model))

gen = create_generator()

for step in range(100):
    optim.zero_grad()

    try:
        x, y = next(gen)
    except StopIteration:
        gen = create_generator()
        x, y = next(gen)

    x.requires_grad = True
    y.requires_grad = True

    out = model(x)
    loss = (out - y).square().mean()
    print(f"{step=} loss={loss.numpy()}")

    loss.backward()
    optim.step()

nn.state.safe_save(nn.state.get_state_dict(model), "tabular_data_titanic.bin")
