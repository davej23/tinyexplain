import inspect

from tinygrad import Tensor, nn

from tinyexplain.types import TinygradModel


def count_model_parameters(model: TinygradModel) -> tuple[int, int]:
    layer_info = dict((l.flatten().shape[0], l.requires_grad) for l in nn.state.get_state_dict(model).values())
    return sum(number_params if requires_grad else 0 for number_params, requires_grad in layer_info.items()), sum(
        number_params if not requires_grad else 0 for number_params, requires_grad in layer_info.items()
    )


def summary(model: TinygradModel, x: Tensor) -> None:
    """Create a TensorFlow-style summary

    model   : your tinygrad model
    x       : input for the model

    """

    def _tinyexplain_call_override(self, x: Tensor) -> Tensor:  # Overwriting layer __call__ to track output shape
        out = self.__call_orig__(x)
        self.out = out
        return out

    line_length = 160  # Width of print output

    print("_" * line_length)
    print(f"{'Layer (type)':<110s}{'Shape':<20s}{'Output Shape':<20s}{'Param #':<20s}")
    print("=" * line_length)
    print(f"{'input (Tensor)':<110s}{str(x.shape):<20s}{str(x.shape):<20s}{'N/A':<20s}")

    # Get model layers and replace __call__ with custom ov
    state = nn.state.get_state_dict(model)
    tinygrad_layers = [n[1] for n in inspect.getmembers(nn, inspect.isclass) if "tinygrad.nn" in str(n[1])]

    for l in tinygrad_layers:
        l.__call_orig__ = l.__call__
        l.__call__ = _tinyexplain_call_override

    out = model(x)

    param_counts = count_model_parameters(model)

    # Get individual model layers, shape, out shape, params
    for ln, l in state.items():
        spl = [f"[{n}]" if n.isnumeric() else n for n in ln.split(".")[:-1]]
        spl = [
            (spl[i] + "." if ("[" not in s and "[" not in t) or ("[" in s and "[" not in t) else spl[i])
            for i, (s, t) in enumerate(zip(spl, spl[1:]))
        ] + [spl[-1]]
        layer = eval(f"model.{''.join(spl)}")  # pylint: disable=eval-used
        if ln.split(".")[-1] != "out":
            print(
                f"{f'{ln} ({str(type(layer))})':<110s}"
                + f"{str(l.shape):<20s}"
                + f"{str(layer.out.shape) if hasattr(layer, 'out') else 'N/A':<20s}"
                + f"{str(l.flatten().shape[0]):<20s}"
            )

    print(f"{'output (Tensor)':<110s}{str(out.shape):<20s}{str(out.shape):<20s}{'N/A':<20s}")

    # Revert layer __call__
    for l in tinygrad_layers:
        l.__call__ = l.__call_orig__
        del l.__call_orig__
        del l.out

    print("=" * line_length)
    print(f"Total params: {sum(param_counts)}")
    print(f"Trainable params: {param_counts[0]}")
    print(f"Non-trainable params: {param_counts[1]}")
    print("_" * line_length)
