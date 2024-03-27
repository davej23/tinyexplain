import inspect
from typing import Any

from tinygrad import Tensor, nn

from tinyexplain.utils import TinygradModel

TINYGRAD_LAYERS = [n[1] for n in inspect.getmembers(nn, inspect.isclass)]
RELUS = []


def is_tinygrad_layer(layer: Any) -> bool:
    return layer in TINYGRAD_LAYERS


def get_layer(layer_name: str, model: TinygradModel) -> Any:
    model_var = dict((str(v), k) for k, v in locals().items())[str(model)]
    return eval(f"{model_var}.{layer_name}")


def overwrite_call(obj: Any) -> None:
	# pylint: disable=all
    def _tinyexplain_call_override(self, x: Tensor) -> Tensor:
        out = self.__call_orig__(x)
        self.last_call = out
        return out

    obj.__call_orig__ = obj.__call__
    obj.__call__ = _tinyexplain_call_override


def revert_call(obj: Any) -> None:
	# pylint: disable=all
    if hasattr(obj, "__call_orig__"):
        obj.__call__ = obj.__call_orig__
        del obj.__call_orig__


def get_model_layer_names(model: TinygradModel, obj_type: Any) -> list[str]:
    layers = []
    for name, _ in nn.state.get_state_dict(model, tensor_type=obj_type).items():
        spl = [f"[{n}]" if n.isnumeric() else n for n in name.split(".")]
        spl = [
            (
                spl[i] + "."
                if ("[" not in s and "[" not in t) or ("[" in s and "[" not in t)
                else spl[i]
            )
            for i, (s, t) in enumerate(zip(spl, spl[1:]))
        ] + [spl[-1]]
        layers.append("".join(spl))
    return layers


def get_model_layers(model: TinygradModel, obj_type: Any) -> dict[str, Any]:
    layer_names = get_model_layer_names(model, obj_type)
    layers = [
        eval(
            f"{dict((str(v), k) for k, v in locals().items())[str(model)]}.{layer_name}"
        )
        for layer_name in layer_names
    ]
    return dict(zip(layer_names, layers))


def overwrite_relu() -> None:
	# pylint: disable=all
    def _tinyexplain_relu_override(self):
        global RELUS
        RELUS.append(self)
        return self._relu()

    Tensor._relu = Tensor.relu
    Tensor.relu = _tinyexplain_relu_override


def revert_relu() -> None:
	# pylint: disable=all
    if hasattr(Tensor, "_relu"):
        Tensor.relu = Tensor._relu
        del Tensor._relu


def overwrite_backward(gbp: bool = False) -> None:  # ReLU gradients for backprop
	# pylint: disable=all
	def _tinyexplain_backward_override(self) -> Tensor:
		global RELUS
		assert (
            self.shape == tuple()
        ), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

        # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
        # this is "implicit gradient creation"
		self.grad = Tensor(1.0, device=self.device, requires_grad=False)

		for t0 in reversed(self.deepwalk()):
			assert t0.grad is not None
			grads = t0._ctx.backward(t0.grad.lazydata)
			grads = [
                (
                    Tensor(g, device=self.device, requires_grad=False)
                    if g is not None
                    else None
                )
                for g in ([grads] if len(t0._ctx.parents) == 1 else grads)
            ]
			for t, g in zip(t0._ctx.parents, grads):
				if g is not None and t.requires_grad:
					assert (
                        g.shape == t.shape
                    ), f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
					if str(t) in [str(r) for r in RELUS]:
						grad = (
                            (t.grad.relu() * t.relu()) + g.relu()
                            if gbp
                            else t.grad.relu() + g.relu()
                        )
						t.grad = g.relu() if t.grad is None else grad
					else:
						t.grad = g if t.grad is None else (t.grad + g)
			del t0._ctx
		return self

	Tensor._backward = Tensor.backward
	Tensor.backward = _tinyexplain_backward_override


def revert_backward() -> None:
	# pylint: disable=all
    if hasattr(Tensor, "_backward"):
        Tensor.backward = Tensor._backward
        del Tensor._backward

    if hasattr(Tensor, "last_call"):
        del Tensor.last_call
