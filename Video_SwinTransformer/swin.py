# Modified from 2d Swin Transformers in torchvision:
# https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py

from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# from ...transforms._presets import VideoClassification

from typing import Optional, Tuple, Union
# from ...utils import _log_api_usage_once

from enum import Enum

import fnmatch
import importlib
import inspect
import sys
from dataclasses import dataclass
from enum import Enum
from functools import partial
from inspect import signature
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Type, TypeVar, Union, Sequence

from torch import nn
try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401

import warnings
import math
import functools



class _ModelURLs(dict):
    def __getitem__(self, item):
        warnings.warn(
            "Accessing the model URLs via the internal dictionary of the module is deprecated since 0.13 and may "
            "be removed in the future. Please access them via the appropriate Weights Enum instead."
        )
        return super().__getitem__(item)
    
class WeightsEnum(Enum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `Weights`.

    Args:
        value (Weights): The data class entry with the weight information.
    """

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + ".", "")]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj

    def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    @property
    def url(self):
        return self.value.url

    @property
    def transforms(self):
        return self.value.transforms

    @property
    def meta(self):
        return self.value.meta



W = TypeVar("W", bound=WeightsEnum)
M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _ovewrite_value_param(param: str, actual: Optional[V], expected: V) -> V:
    if actual is not None:
        if actual != expected:
            raise ValueError(f"The parameter '{param}' expected value {expected} but got {actual} instead.")
    return expected
D = TypeVar("D")

def sequence_to_str(seq: Sequence, separate_last: str = "") -> str:
    if not seq:
        return ""
    if len(seq) == 1:
        return f"'{seq[0]}'"

    head = "'" + "', '".join([str(item) for item in seq[:-1]]) + "'"
    tail = f"{'' if separate_last and len(seq) == 2 else ','} {separate_last}'{seq[-1]}'"

    return head + tail

def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]:
    """Decorates a function that uses keyword only parameters to also allow them being passed as positionals.

    For example, consider the use case of changing the signature of ``old_fn`` into the one from ``new_fn``:

    .. code::

        def old_fn(foo, bar, baz=None):
            ...

        def new_fn(foo, *, bar, baz=None):
            ...

    Calling ``old_fn("foo", "bar, "baz")`` was valid, but the same call is no longer valid with ``new_fn``. To keep BC
    and at the same time warn the user of the deprecation, this decorator can be used:

    .. code::

        @kwonly_to_pos_or_kw
        def new_fn(foo, *, bar, baz=None):
            ...

        new_fn("foo", "bar, "baz")
    """
    params = inspect.signature(fn).parameters

    try:
        keyword_only_start_idx = next(
            idx for idx, param in enumerate(params.values()) if param.kind == param.KEYWORD_ONLY
        )
    except StopIteration:
        raise TypeError(f"Found no keyword-only parameter on function '{fn.__name__}'") from None

    keyword_only_params = tuple(inspect.signature(fn).parameters)[keyword_only_start_idx:]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> D:
        args, keyword_only_args = args[:keyword_only_start_idx], args[keyword_only_start_idx:]
        if keyword_only_args:
            keyword_only_kwargs = dict(zip(keyword_only_params, keyword_only_args))
            warnings.warn(
                f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
                f"parameter(s) is deprecated since 0.13 and may be removed in the future. Please use keyword parameter(s) "
                f"instead."
            )
            kwargs.update(keyword_only_kwargs)

        return fn(*args, **kwargs)

    return wrapper
def handle_legacy_interface(**weights: Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]):
    """Decorates a model builder with the new interface to make it compatible with the old.

    In particular this handles two things:

    1. Allows positional parameters again, but emits a deprecation warning in case they are used. See
        :func:`torchvision.prototype.utils._internal.kwonly_to_pos_or_kw` for details.
    2. Handles the default value change from ``pretrained=False`` to ``weights=None`` and ``pretrained=True`` to
        ``weights=Weights`` and emits a deprecation warning with instructions for the new interface.

    Args:
        **weights (Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]): Deprecated parameter
            name and default value for the legacy ``pretrained=True``. The default value can be a callable in which
            case it will be called with a dictionary of the keyword arguments. The only key that is guaranteed to be in
            the dictionary is the deprecated parameter name passed as first element in the tuple. All other parameters
            should be accessed with :meth:`~dict.get`.
    """

    def outer_wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        @kwonly_to_pos_or_kw
        @functools.wraps(builder)
        def inner_wrapper(*args: Any, **kwargs: Any) -> M:
            for weights_param, (pretrained_param, default) in weights.items():  # type: ignore[union-attr]
                # If neither the weights nor the pretrained parameter as passed, or the weights argument already use
                # the new style arguments, there is nothing to do. Note that we cannot use `None` as sentinel for the
                # weight argument, since it is a valid value.
                sentinel = object()
                weights_arg = kwargs.get(weights_param, sentinel)
                if (
                    (weights_param not in kwargs and pretrained_param not in kwargs)
                    or isinstance(weights_arg, WeightsEnum)
                    or (isinstance(weights_arg, str) and weights_arg != "legacy")
                    or weights_arg is None
                ):
                    continue

                # If the pretrained parameter was passed as positional argument, it is now mapped to
                # `kwargs[weights_param]`. This happens because the @kwonly_to_pos_or_kw decorator uses the current
                # signature to infer the names of positionally passed arguments and thus has no knowledge that there
                # used to be a pretrained parameter.
                pretrained_positional = weights_arg is not sentinel
                if pretrained_positional:
                    # We put the pretrained argument under its legacy name in the keyword argument dictionary to have
                    # unified access to the value if the default value is a callable.
                    kwargs[pretrained_param] = pretrained_arg = kwargs.pop(weights_param)
                else:
                    pretrained_arg = kwargs[pretrained_param]

                if pretrained_arg:
                    default_weights_arg = default(kwargs) if callable(default) else default
                    if not isinstance(default_weights_arg, WeightsEnum):
                        raise ValueError(f"No weights available for model {builder.__name__}")
                else:
                    default_weights_arg = None

                if not pretrained_positional:
                    warnings.warn(
                        f"The parameter '{pretrained_param}' is deprecated since 0.13 and may be removed in the future, "
                        f"please use '{weights_param}' instead."
                    )

                msg = (
                    f"Arguments other than a weight enum or `None` for '{weights_param}' are deprecated since 0.13 and "
                    f"may be removed in the future. "
                    f"The current behavior is equivalent to passing `{weights_param}={default_weights_arg}`."
                )
                if pretrained_arg:
                    msg = (
                        f"{msg} You can also use `{weights_param}={type(default_weights_arg).__name__}.DEFAULT` "
                        f"to get the most up-to-date weights."
                    )
                warnings.warn(msg)

                del kwargs[pretrained_param]
                kwargs[weights_param] = default_weights_arg

            return builder(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
    return x


torch.fx.wrap("_patch_merging_pad")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")
class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        # _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention")




class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )
    
class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        # _log_api_usage_once(self)


class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)

def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        # _log_api_usage_once(stochastic_depth)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


torch.fx.wrap("stochastic_depth")


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s
    
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()
        # _log_api_usage_once(self)

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x







__all__ = ["WeightsEnum", "Weights", "get_model", "get_model_builder", "get_model_weights", "get_weight", "list_models"]


@dataclass
class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.

    Args:
        url (str): The location where we find the weights.
        transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
            needed to use the model. The reason we attach a constructor method rather than an already constructed
            object is because the specific object might have memory and thus we want to delay initialization until
            needed.
        meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
            informative attributes (for example the number of parameters/flops, recipe link/methods used in training
            etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
            meta-data (for example the `classes` of a classification model) needed to use the model.
    """

    url: str
    transforms: Callable
    meta: Dict[str, Any]

    def __eq__(self, other: Any) -> bool:
        # We need this custom implementation for correct deep-copy and deserialization behavior.
        # TL;DR: After the definition of an enum, creating a new instance, i.e. by deep-copying or deserializing it,
        # involves an equality check against the defined members. Unfortunately, the `transforms` attribute is often
        # defined with `functools.partial` and `fn = partial(...); assert deepcopy(fn) != fn`. Without custom handling
        # for it, the check against the defined members would fail and effectively prevent the weights from being
        # deep-copied or deserialized.
        # See https://github.com/pytorch/vision/pull/7107 for details.
        if not isinstance(other, Weights):
            return NotImplemented

        if self.url != other.url:
            return False

        if self.meta != other.meta:
            return False

        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return (
                self.transforms.func == other.transforms.func
                and self.transforms.args == other.transforms.args
                and self.transforms.keywords == other.transforms.keywords
            )
        else:
            return self.transforms == other.transforms


class WeightsEnum(Enum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `Weights`.

    Args:
        value (Weights): The data class entry with the weight information.
    """

    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + ".", "")]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj

    def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    @property
    def url(self):
        return self.value.url

    @property
    def transforms(self):
        return self.value.transforms

    @property
    def meta(self):
        return self.value.meta


def get_weight(name: str) -> WeightsEnum:
    """
    Gets the weights enum value by its full name. Example: "ResNet50_Weights.IMAGENET1K_V1"

    Args:
        name (str): The name of the weight enum entry.

    Returns:
        WeightsEnum: The requested weight enum.
    """
    try:
        enum_name, value_name = name.split(".")
    except ValueError:
        raise ValueError(f"Invalid weight name provided: '{name}'.")

    base_module_name = ".".join(sys.modules[__name__].__name__.split(".")[:-1])
    base_module = importlib.import_module(base_module_name)
    model_modules = [base_module] + [
        x[1]
        for x in inspect.getmembers(base_module, inspect.ismodule)
        if x[1].__file__.endswith("__init__.py")  # type: ignore[union-attr]
    ]

    weights_enum = None
    for m in model_modules:
        potential_class = m.__dict__.get(enum_name, None)
        if potential_class is not None and issubclass(potential_class, WeightsEnum):
            weights_enum = potential_class
            break

    if weights_enum is None:
        raise ValueError(f"The weight enum '{enum_name}' for the specific method couldn't be retrieved.")

    return weights_enum[value_name]


def get_model_weights(name: Union[Callable, str]) -> Type[WeightsEnum]:
    """
    Returns the weights enum class associated to the given model.

    Args:
        name (callable or str): The model builder function or the name under which it is registered.

    Returns:
        weights_enum (WeightsEnum): The weights enum class associated with the model.
    """
    model = get_model_builder(name) if isinstance(name, str) else name
    return _get_enum_from_fn(model)


def _get_enum_from_fn(fn: Callable) -> Type[WeightsEnum]:
    """
    Internal method that gets the weight enum of a specific model builder method.

    Args:
        fn (Callable): The builder method used to create the model.
    Returns:
        WeightsEnum: The requested weight enum.
    """
    sig = signature(fn)
    if "weights" not in sig.parameters:
        raise ValueError("The method is missing the 'weights' argument.")

    ann = signature(fn).parameters["weights"].annotation
    weights_enum = None
    if isinstance(ann, type) and issubclass(ann, WeightsEnum):
        weights_enum = ann
    else:
        # handle cases like Union[Optional, T]
        # TODO: Replace ann.__args__ with typing.get_args(ann) after python >= 3.8
        for t in ann.__args__:  # type: ignore[union-attr]
            if isinstance(t, type) and issubclass(t, WeightsEnum):
                weights_enum = t
                break

    if weights_enum is None:
        raise ValueError(
            "The WeightsEnum class for the specific method couldn't be retrieved. Make sure the typing info is correct."
        )

    return weights_enum


M = TypeVar("M", bound=nn.Module)

BUILTIN_MODELS = {}


def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn

    return wrapper


def list_models(
    module: Optional[ModuleType] = None,
    include: Union[Iterable[str], str, None] = None,
    exclude: Union[Iterable[str], str, None] = None,
) -> List[str]:
    """
    Returns a list with the names of registered models.

    Args:
        module (ModuleType, optional): The module from which we want to extract the available models.
        include (str or Iterable[str], optional): Filter(s) for including the models from the set of all models.
            Filters are passed to `fnmatch <https://docs.python.org/3/library/fnmatch.html>`__ to match Unix shell-style
            wildcards. In case of many filters, the results is the union of individual filters.
        exclude (str or Iterable[str], optional): Filter(s) applied after include_filters to remove models.
            Filter are passed to `fnmatch <https://docs.python.org/3/library/fnmatch.html>`__ to match Unix shell-style
            wildcards. In case of many filters, the results is removal of all the models that match any individual filter.

    Returns:
        models (list): A list with the names of available models.
    """
    all_models = {
        k for k, v in BUILTIN_MODELS.items() if module is None or v.__module__.rsplit(".", 1)[0] == module.__name__
    }
    if include:
        models: Set[str] = set()
        if isinstance(include, str):
            include = [include]
        for include_filter in include:
            models = models | set(fnmatch.filter(all_models, include_filter))
    else:
        models = all_models

    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]
        for exclude_filter in exclude:
            models = models - set(fnmatch.filter(all_models, exclude_filter))
    return sorted(models)


def get_model_builder(name: str) -> Callable[..., nn.Module]:
    """
    Gets the model name and returns the model builder method.

    Args:
        name (str): The name under which the model is registered.

    Returns:
        fn (Callable): The model builder method.
    """
    name = name.lower()
    try:
        fn = BUILTIN_MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name}")
    return fn


def get_model(name: str, **config: Any) -> nn.Module:
    """
    Gets the model name and configuration and returns an instantiated model.

    Args:
        name (str): The name under which the model is registered.
        **config (Any): parameters passed to the model builder method.

    Returns:
        model (nn.Module): The initialized model.
    """
    fn = get_model_builder(name)
    return fn(**config)


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``nearest-exact``, ``bilinear``, ``bicubic``, ``box``, ``hamming``,
    and ``lanczos``.
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


class VideoClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: Tuple[int, int],
        resize_size: Union[Tuple[int], Tuple[int, int]],
        mean: Tuple[float, ...] = (0.43216, 0.394666, 0.37645),
        std: Tuple[float, ...] = (0.22803, 0.22145, 0.216989),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.crop_size = list(crop_size)
        self.resize_size = list(resize_size)
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, vid: Tensor) -> Tensor:
        need_squeeze = False
        if vid.ndim < 5:
            vid = vid.unsqueeze(dim=0)
            need_squeeze = True

        N, T, C, H, W = vid.shape
        vid = vid.view(-1, C, H, W)
        # We hard-code antialias=False to preserve results after we changed
        # its default from None to True (see
        # https://github.com/pytorch/vision/pull/7160)
        # TODO: we could re-train the video models with antialias=True?
        vid = F.resize(vid, self.resize_size, interpolation=self.interpolation, antialias=False)
        vid = F.center_crop(vid, self.crop_size)
        vid = F.convert_image_dtype(vid, torch.float)
        vid = F.normalize(vid, mean=self.mean, std=self.std)
        H, W = self.crop_size
        vid = vid.view(N, T, C, H, W)
        vid = vid.permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) => (N, C, T, H, W)

        if need_squeeze:
            vid = vid.squeeze(dim=0)
        return vid

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts batched ``(B, T, C, H, W)`` and single ``(T, C, H, W)`` video frame ``torch.Tensor`` objects. "
            f"The frames are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``. Finally the output "
            "dimensions are permuted to ``(..., C, T, H, W)`` tensors."
        )






__all__ = [
    "SwinTransformer3d",
    "Swin3D_T_Weights",
    "Swin3D_S_Weights",
    "Swin3D_B_Weights",
    "swin3d_t",
    "swin3d_s",
    "swin3d_b",
]


def _get_window_and_shift_size(
    shift_size: List[int], size_dhw: List[int], window_size: List[int]
) -> Tuple[List[int], List[int]]:
    for i in range(3):
        if size_dhw[i] <= window_size[i]:
            # In this case, window_size will adapt to the input size, and no need to shift
            window_size[i] = size_dhw[i]
            shift_size[i] = 0

    return window_size, shift_size


torch.fx.wrap("_get_window_and_shift_size")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> Tensor:
    window_vol = window_size[0] * window_size[1] * window_size[2]
    # In 3d case we flatten the relative_position_bias
    relative_position_bias = relative_position_bias_table[
        relative_position_index[:window_vol, :window_vol].flatten()  # type: ignore[index]
    ]
    relative_position_bias = relative_position_bias.view(window_vol, window_vol, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")


def _compute_pad_size_3d(size_dhw: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    pad_size = [(patch_size[i] - size_dhw[i] % patch_size[i]) % patch_size[i] for i in range(3)]
    return pad_size[0], pad_size[1], pad_size[2]


torch.fx.wrap("_compute_pad_size_3d")


def _compute_attention_mask_3d(
    x: Tensor,
    size_dhw: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    shift_size: Tuple[int, int, int],
) -> Tensor:
    # generate attention mask
    attn_mask = x.new_zeros(*size_dhw)
    num_windows = (size_dhw[0] // window_size[0]) * (size_dhw[1] // window_size[1]) * (size_dhw[2] // window_size[2])
    slices = [
        (
            (0, -window_size[i]),
            (-window_size[i], -shift_size[i]),
            (-shift_size[i], None),
        )
        for i in range(3)
    ]
    count = 0
    for d in slices[0]:
        for h in slices[1]:
            for w in slices[2]:
                attn_mask[d[0] : d[1], h[0] : h[1], w[0] : w[1]] = count
                count += 1

    # Partition window on attn_mask
    attn_mask = attn_mask.view(
        size_dhw[0] // window_size[0],
        window_size[0],
        size_dhw[1] // window_size[1],
        window_size[1],
        size_dhw[2] // window_size[2],
        window_size[2],
    )
    attn_mask = attn_mask.permute(0, 2, 4, 1, 3, 5).reshape(
        num_windows, window_size[0] * window_size[1] * window_size[2]
    )
    attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


torch.fx.wrap("_compute_attention_mask_3d")


def shifted_window_attention_3d(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[B, T, H, W, C]): The input tensor, 5-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): 3-dimensions window size, T, H, W .
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention (T, H, W).
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[B, T, H, W, C]: The output tensor after shifted window attention.
    """
    b, t, h, w, c = input.shape
    # pad feature maps to multiples of window size
    pad_size = _compute_pad_size_3d((t, h, w), (window_size[0], window_size[1], window_size[2]))
    x = F.pad(input, (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
    _, tp, hp, wp, _ = x.shape
    padded_size = (tp, hp, wp)

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    # partition windows
    num_windows = (
        (padded_size[0] // window_size[0]) * (padded_size[1] // window_size[1]) * (padded_size[2] // window_size[2])
    )
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        window_size[0],
        padded_size[1] // window_size[1],
        window_size[1],
        padded_size[2] // window_size[2],
        window_size[2],
        c,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        b * num_windows, window_size[0] * window_size[1] * window_size[2], c
    )  # B*nW, Wd*Wh*Ww, C

    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (c // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask to handle shifted windows with varying size
        attn_mask = _compute_attention_mask_3d(
            x,
            (padded_size[0], padded_size[1], padded_size[2]),
            (window_size[0], window_size[1], window_size[2]),
            (shift_size[0], shift_size[1], shift_size[2]),
        )
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        padded_size[1] // window_size[1],
        padded_size[2] // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        c,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, tp, hp, wp, c)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

    # unpad features
    x = x[:, :t, :h, :w, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention_3d")


class ShiftedWindowAttention3d(nn.Module):
    """
    See :func:`shifted_window_attention_3d`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(window_size) != 3 or len(shift_size) != 3:
            raise ValueError("window_size and shift_size must be of length 2")

        self.window_size = window_size  # Wd, Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self) -> None:
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                self.num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self) -> None:
        # get pair-wise relative position index for each token inside the window
        coords_dhw = [torch.arange(self.window_size[i]) for i in range(3)]
        coords = torch.stack(
            torch.meshgrid(coords_dhw[0], coords_dhw[1], coords_dhw[2], indexing="ij")
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        # We don't flatten the relative_position_index here in 3d case.
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self, window_size: List[int]) -> torch.Tensor:
        return _get_relative_position_bias(self.relative_position_bias_table, self.relative_position_index, window_size)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        _, t, h, w, _ = x.shape
        size_dhw = [t, h, w]
        window_size, shift_size = self.window_size.copy(), self.shift_size.copy()
        # Handle case where window_size is larger than the input tensor
        window_size, shift_size = _get_window_and_shift_size(shift_size, size_dhw, window_size)

        relative_position_bias = self.get_relative_position_bias(window_size)

        return shifted_window_attention_3d(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            window_size,
            self.num_heads,
            shift_size=shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


# Modified from:
# https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py
class PatchEmbed3d(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (List[int]): Patch token size.
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size: List[int],
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        # padding
        _, _, t, h, w = x.size()
        pad_size = _compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
        x = self.proj(x)  # B C T Wh Ww
        x = x.permute(0, 2, 3, 4, 1)  # B T Wh Ww C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformer3d(nn.Module):
    """
    Implements 3D Swin Transformer from the `"Video Swin Transformer" <https://arxiv.org/abs/2106.13230>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 400.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
        patch_embed (nn.Module, optional): Patch Embedding layer. Default: None.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 400,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        patch_embed: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.num_classes = num_classes

        if block is None:
            block = partial(SwinTransformerBlock, attn_layer=ShiftedWindowAttention3d)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        if patch_embed is None:
            patch_embed = PatchEmbed3d

        # split image into non-overlapping patches
        self.patch_embed = patch_embed(patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=dropout)

        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        attn_layer=ShiftedWindowAttention3d,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: B C T H W
        x = self.patch_embed(x)  # B _T _H _W C
        x = self.pos_drop(x)
        x = self.features(x)  # B _T _H _W C
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, _T, _H, _W
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def _swin_transformer3d(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SwinTransformer3d:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SwinTransformer3d(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_KINETICS400_CATEGORIES = [
    "abseiling",
    "air drumming",
    "answering questions",
    "applauding",
    "applying cream",
    "archery",
    "arm wrestling",
    "arranging flowers",
    "assembling computer",
    "auctioning",
    "baby waking up",
    "baking cookies",
    "balloon blowing",
    "bandaging",
    "barbequing",
    "bartending",
    "beatboxing",
    "bee keeping",
    "belly dancing",
    "bench pressing",
    "bending back",
    "bending metal",
    "biking through snow",
    "blasting sand",
    "blowing glass",
    "blowing leaves",
    "blowing nose",
    "blowing out candles",
    "bobsledding",
    "bookbinding",
    "bouncing on trampoline",
    "bowling",
    "braiding hair",
    "breading or breadcrumbing",
    "breakdancing",
    "brush painting",
    "brushing hair",
    "brushing teeth",
    "building cabinet",
    "building shed",
    "bungee jumping",
    "busking",
    "canoeing or kayaking",
    "capoeira",
    "carrying baby",
    "cartwheeling",
    "carving pumpkin",
    "catching fish",
    "catching or throwing baseball",
    "catching or throwing frisbee",
    "catching or throwing softball",
    "celebrating",
    "changing oil",
    "changing wheel",
    "checking tires",
    "cheerleading",
    "chopping wood",
    "clapping",
    "clay pottery making",
    "clean and jerk",
    "cleaning floor",
    "cleaning gutters",
    "cleaning pool",
    "cleaning shoes",
    "cleaning toilet",
    "cleaning windows",
    "climbing a rope",
    "climbing ladder",
    "climbing tree",
    "contact juggling",
    "cooking chicken",
    "cooking egg",
    "cooking on campfire",
    "cooking sausages",
    "counting money",
    "country line dancing",
    "cracking neck",
    "crawling baby",
    "crossing river",
    "crying",
    "curling hair",
    "cutting nails",
    "cutting pineapple",
    "cutting watermelon",
    "dancing ballet",
    "dancing charleston",
    "dancing gangnam style",
    "dancing macarena",
    "deadlifting",
    "decorating the christmas tree",
    "digging",
    "dining",
    "disc golfing",
    "diving cliff",
    "dodgeball",
    "doing aerobics",
    "doing laundry",
    "doing nails",
    "drawing",
    "dribbling basketball",
    "drinking",
    "drinking beer",
    "drinking shots",
    "driving car",
    "driving tractor",
    "drop kicking",
    "drumming fingers",
    "dunking basketball",
    "dying hair",
    "eating burger",
    "eating cake",
    "eating carrots",
    "eating chips",
    "eating doughnuts",
    "eating hotdog",
    "eating ice cream",
    "eating spaghetti",
    "eating watermelon",
    "egg hunting",
    "exercising arm",
    "exercising with an exercise ball",
    "extinguishing fire",
    "faceplanting",
    "feeding birds",
    "feeding fish",
    "feeding goats",
    "filling eyebrows",
    "finger snapping",
    "fixing hair",
    "flipping pancake",
    "flying kite",
    "folding clothes",
    "folding napkins",
    "folding paper",
    "front raises",
    "frying vegetables",
    "garbage collecting",
    "gargling",
    "getting a haircut",
    "getting a tattoo",
    "giving or receiving award",
    "golf chipping",
    "golf driving",
    "golf putting",
    "grinding meat",
    "grooming dog",
    "grooming horse",
    "gymnastics tumbling",
    "hammer throw",
    "headbanging",
    "headbutting",
    "high jump",
    "high kick",
    "hitting baseball",
    "hockey stop",
    "holding snake",
    "hopscotch",
    "hoverboarding",
    "hugging",
    "hula hooping",
    "hurdling",
    "hurling (sport)",
    "ice climbing",
    "ice fishing",
    "ice skating",
    "ironing",
    "javelin throw",
    "jetskiing",
    "jogging",
    "juggling balls",
    "juggling fire",
    "juggling soccer ball",
    "jumping into pool",
    "jumpstyle dancing",
    "kicking field goal",
    "kicking soccer ball",
    "kissing",
    "kitesurfing",
    "knitting",
    "krumping",
    "laughing",
    "laying bricks",
    "long jump",
    "lunge",
    "making a cake",
    "making a sandwich",
    "making bed",
    "making jewelry",
    "making pizza",
    "making snowman",
    "making sushi",
    "making tea",
    "marching",
    "massaging back",
    "massaging feet",
    "massaging legs",
    "massaging person's head",
    "milking cow",
    "mopping floor",
    "motorcycling",
    "moving furniture",
    "mowing lawn",
    "news anchoring",
    "opening bottle",
    "opening present",
    "paragliding",
    "parasailing",
    "parkour",
    "passing American football (in game)",
    "passing American football (not in game)",
    "peeling apples",
    "peeling potatoes",
    "petting animal (not cat)",
    "petting cat",
    "picking fruit",
    "planting trees",
    "plastering",
    "playing accordion",
    "playing badminton",
    "playing bagpipes",
    "playing basketball",
    "playing bass guitar",
    "playing cards",
    "playing cello",
    "playing chess",
    "playing clarinet",
    "playing controller",
    "playing cricket",
    "playing cymbals",
    "playing didgeridoo",
    "playing drums",
    "playing flute",
    "playing guitar",
    "playing harmonica",
    "playing harp",
    "playing ice hockey",
    "playing keyboard",
    "playing kickball",
    "playing monopoly",
    "playing organ",
    "playing paintball",
    "playing piano",
    "playing poker",
    "playing recorder",
    "playing saxophone",
    "playing squash or racquetball",
    "playing tennis",
    "playing trombone",
    "playing trumpet",
    "playing ukulele",
    "playing violin",
    "playing volleyball",
    "playing xylophone",
    "pole vault",
    "presenting weather forecast",
    "pull ups",
    "pumping fist",
    "pumping gas",
    "punching bag",
    "punching person (boxing)",
    "push up",
    "pushing car",
    "pushing cart",
    "pushing wheelchair",
    "reading book",
    "reading newspaper",
    "recording music",
    "riding a bike",
    "riding camel",
    "riding elephant",
    "riding mechanical bull",
    "riding mountain bike",
    "riding mule",
    "riding or walking with horse",
    "riding scooter",
    "riding unicycle",
    "ripping paper",
    "robot dancing",
    "rock climbing",
    "rock scissors paper",
    "roller skating",
    "running on treadmill",
    "sailing",
    "salsa dancing",
    "sanding floor",
    "scrambling eggs",
    "scuba diving",
    "setting table",
    "shaking hands",
    "shaking head",
    "sharpening knives",
    "sharpening pencil",
    "shaving head",
    "shaving legs",
    "shearing sheep",
    "shining shoes",
    "shooting basketball",
    "shooting goal (soccer)",
    "shot put",
    "shoveling snow",
    "shredding paper",
    "shuffling cards",
    "side kick",
    "sign language interpreting",
    "singing",
    "situp",
    "skateboarding",
    "ski jumping",
    "skiing (not slalom or crosscountry)",
    "skiing crosscountry",
    "skiing slalom",
    "skipping rope",
    "skydiving",
    "slacklining",
    "slapping",
    "sled dog racing",
    "smoking",
    "smoking hookah",
    "snatch weight lifting",
    "sneezing",
    "sniffing",
    "snorkeling",
    "snowboarding",
    "snowkiting",
    "snowmobiling",
    "somersaulting",
    "spinning poi",
    "spray painting",
    "spraying",
    "springboard diving",
    "squat",
    "sticking tongue out",
    "stomping grapes",
    "stretching arm",
    "stretching leg",
    "strumming guitar",
    "surfing crowd",
    "surfing water",
    "sweeping floor",
    "swimming backstroke",
    "swimming breast stroke",
    "swimming butterfly stroke",
    "swing dancing",
    "swinging legs",
    "swinging on something",
    "sword fighting",
    "tai chi",
    "taking a shower",
    "tango dancing",
    "tap dancing",
    "tapping guitar",
    "tapping pen",
    "tasting beer",
    "tasting food",
    "testifying",
    "texting",
    "throwing axe",
    "throwing ball",
    "throwing discus",
    "tickling",
    "tobogganing",
    "tossing coin",
    "tossing salad",
    "training dog",
    "trapezing",
    "trimming or shaving beard",
    "trimming trees",
    "triple jump",
    "tying bow tie",
    "tying knot (not on a tie)",
    "tying tie",
    "unboxing",
    "unloading truck",
    "using computer",
    "using remote controller (not gaming)",
    "using segway",
    "vault",
    "waiting in line",
    "walking the dog",
    "washing dishes",
    "washing feet",
    "washing hair",
    "washing hands",
    "water skiing",
    "water sliding",
    "watering plants",
    "waxing back",
    "waxing chest",
    "waxing eyebrows",
    "waxing legs",
    "weaving basket",
    "welding",
    "whistling",
    "windsurfing",
    "wrapping present",
    "wrestling",
    "writing",
    "yawning",
    "yoga",
    "zumba",
]

_COMMON_META = {
    "categories": _KINETICS400_CATEGORIES,
    "min_size": (1, 1),
    "min_temporal_size": 1,
}


class Swin3D_T_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(
        url="https://download.pytorch.org/models/swin3d_t-7615ae03.pth",
        transforms=partial(
            VideoClassification,
            crop_size=(224, 224),
            resize_size=(256,),
            mean=(0.4850, 0.4560, 0.4060),
            std=(0.2290, 0.2240, 0.2250),
        ),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/SwinTransformer/Video-Swin-Transformer#kinetics-400",
            "_docs": (
                "The weights were ported from the paper. The accuracies are estimated on video-level "
                "with parameters `frame_rate=15`, `clips_per_video=12`, and `clip_len=32`"
            ),
            "num_params": 28158070,
            "_metrics": {
                "Kinetics-400": {
                    "acc@1": 77.715,
                    "acc@5": 93.519,
                }
            },
            "_ops": 43.882,
            "_file_size": 121.543,
        },
    )
    DEFAULT = KINETICS400_V1


class Swin3D_S_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(
        url="https://download.pytorch.org/models/swin3d_s-da41c237.pth",
        transforms=partial(
            VideoClassification,
            crop_size=(224, 224),
            resize_size=(256,),
            mean=(0.4850, 0.4560, 0.4060),
            std=(0.2290, 0.2240, 0.2250),
        ),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/SwinTransformer/Video-Swin-Transformer#kinetics-400",
            "_docs": (
                "The weights were ported from the paper. The accuracies are estimated on video-level "
                "with parameters `frame_rate=15`, `clips_per_video=12`, and `clip_len=32`"
            ),
            "num_params": 49816678,
            "_metrics": {
                "Kinetics-400": {
                    "acc@1": 79.521,
                    "acc@5": 94.158,
                }
            },
            "_ops": 82.841,
            "_file_size": 218.288,
        },
    )
    DEFAULT = KINETICS400_V1


class Swin3D_B_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(
        url="https://download.pytorch.org/models/swin3d_b_1k-24f7c7c6.pth",
        transforms=partial(
            VideoClassification,
            crop_size=(224, 224),
            resize_size=(256,),
            mean=(0.4850, 0.4560, 0.4060),
            std=(0.2290, 0.2240, 0.2250),
        ),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/SwinTransformer/Video-Swin-Transformer#kinetics-400",
            "_docs": (
                "The weights were ported from the paper. The accuracies are estimated on video-level "
                "with parameters `frame_rate=15`, `clips_per_video=12`, and `clip_len=32`"
            ),
            "num_params": 88048984,
            "_metrics": {
                "Kinetics-400": {
                    "acc@1": 79.427,
                    "acc@5": 94.386,
                }
            },
            "_ops": 140.667,
            "_file_size": 364.134,
        },
    )
    KINETICS400_IMAGENET22K_V1 = Weights(
        url="https://download.pytorch.org/models/swin3d_b_22k-7c6ae6fa.pth",
        transforms=partial(
            VideoClassification,
            crop_size=(224, 224),
            resize_size=(256,),
            mean=(0.4850, 0.4560, 0.4060),
            std=(0.2290, 0.2240, 0.2250),
        ),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/SwinTransformer/Video-Swin-Transformer#kinetics-400",
            "_docs": (
                "The weights were ported from the paper. The accuracies are estimated on video-level "
                "with parameters `frame_rate=15`, `clips_per_video=12`, and `clip_len=32`"
            ),
            "num_params": 88048984,
            "_metrics": {
                "Kinetics-400": {
                    "acc@1": 81.643,
                    "acc@5": 95.574,
                }
            },
            "_ops": 140.667,
            "_file_size": 364.134,
        },
    )
    DEFAULT = KINETICS400_V1


@register_model()
@handle_legacy_interface(weights=("pretrained", Swin3D_T_Weights.KINETICS400_V1))
def swin3d_t(*, weights: Optional[Swin3D_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer3d:
    """
    Constructs a swin_tiny architecture from
    `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`_.

    Args:
        weights (:class:`~torchvision.models.video.Swin3D_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.Swin3D_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.Swin3D_T_Weights
        :members:
    """
    weights = Swin3D_T_Weights.verify(weights)

    return _swin_transformer3d(
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        stochastic_depth_prob=0.1,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", Swin3D_S_Weights.KINETICS400_V1))
def swin3d_s(*, weights: Optional[Swin3D_S_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer3d:
    """
    Constructs a swin_small architecture from
    `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`_.

    Args:
        weights (:class:`~torchvision.models.video.Swin3D_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.Swin3D_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.Swin3D_S_Weights
        :members:
    """
    weights = Swin3D_S_Weights.verify(weights)

    return _swin_transformer3d(
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        stochastic_depth_prob=0.1,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", Swin3D_B_Weights.KINETICS400_V1))
def swin3d_b(*, weights: Optional[Swin3D_B_Weights] =  Swin3D_B_Weights.KINETICS400_V1, progress: bool = True, **kwargs: Any) -> SwinTransformer3d:
    """
    Constructs a swin_base architecture from
    `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`_.

    Args:
        weights (:class:`~torchvision.models.video.Swin3D_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.Swin3D_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.Swin3D_B_Weights
        :members:
    """
    weights = Swin3D_B_Weights.verify(weights)

    return _swin_transformer3d(
        patch_size=[2, 4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 7, 7],
        stochastic_depth_prob=0.1,
        weights=weights,
        progress=progress,
        **kwargs,
    )


swin_base = swin3d_b()
swin_base.head = nn.Identity()
print(swin_base(torch.zeros(4, 3, 30, 224, 224)).shape)
print("param", sum(p.numel() for p in swin_base.parameters() if p.requires_grad))