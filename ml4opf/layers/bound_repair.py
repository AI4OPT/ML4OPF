"""Differentiable repair layer for satisfying bound constraints x̲≤x≤x̅."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class BoundRepair(nn.Module):
    """An activation function that clips the output to a given range."""

    SUPPORTED_METHODS = ["relu", "sigmoid", "clamp", "softplus", "tanh", "none"]

    def __init__(
        self,
        xmin: Optional[Tensor],
        xmax: Optional[Tensor],
        method: str = "relu",
        sanity_check: bool = True,
        memory_efficient: int = 0,
    ):
        """Initializes the BoundRepair module.

        If both `xmin` and `xmax` are None, per-sample bounds must be provided as input to the forward method.
        In this case `memory_efficient` is ignored (it is set to 2 regardless).

        Args:
            xmin (Tensor): Lower bounds for clipping.
            xmax (Tensor): Upper bounds for clipping.
            method (str): The method to use for clipping. One of ["relu", "sigmoid", "clamp", "softplus", "tanh", "none"].
            sanity_check (bool): If True, performs sanity checks on the input.
            memory_efficient (int): 0: pre-compute masks and pre-index bounds, 1: pre-compute masks, 2: do not pre-compute anything
        """
        super().__init__()

        if sanity_check:
            assert (xmin is None and xmax is None) or (
                xmin is not None and xmax is not None
            ), "BoundRepair got only one of xmin or xmax"

            if xmin is not None and xmax is not None:
                assert (
                    xmin.shape == xmax.shape
                ), f"BoundRepair got unequal shapes for lower & upper bounds: {xmin.shape=} {xmax.shape=}"
                assert (
                    method.lower() in self.SUPPORTED_METHODS
                ), f"BoundRepair got {method=} but only supports {self.SUPPORTED_METHODS=}"

                for l, u in zip(xmin, xmax):
                    assert l <= u, f"BoundRepair got xmin > xmax: xmin={l} xmax={u}"
                    assert l != torch.inf, f"BoundRepair got xmin with wrong-signed infinity: xmin={l}"
                    assert u != -torch.inf, f"BoundRepair got xmax with wrong-signed infinity: xmax={u}"
                    if method in ["sigmoid", "tanh"]:
                        assert ((l == -torch.inf) and (u == torch.inf)) or (
                            (l != -torch.inf) and (u != torch.inf)
                        ), f"BoundRepair with {method=} only supports double-sided or no bounds"

            self.sanity_check = True

        self.register_buffer("method_idx", torch.as_tensor(self.SUPPORTED_METHODS.index(method.lower())))

        if xmin is None and xmax is None:
            memory_efficient = 2
        else:
            self.register_buffer("xmin", xmin)
            self.register_buffer("xmax", xmax)

        self.register_buffer("memory_efficient", torch.as_tensor(memory_efficient))
        self.preprocess_bounds(self.memory_efficient)

        self._forward = getattr(self, self.SUPPORTED_METHODS[self.method_idx])

    def __repr__(self):
        return f"BoundRepair(method={self.SUPPORTED_METHODS[self.method_idx]})"

    def forward(self, x: Tensor, xmin: Optional[Tensor] = None, xmax: Optional[Tensor] = None):
        """Applies the bound clipping function to the input."""
        if self.sanity_check:
            if xmin is not None and xmax is not None:
                assert (xmin <= xmax).all(), f"BoundRepair got input with xmin > xmax: {xmin=} {xmax=}"
                assert (
                    xmin.shape == xmax.shape
                ), f"BoundRepair got input with unequal shapes for lower & upper bounds: {xmin.shape=} {xmax.shape=}"
                assert (xmin.shape == x.shape[1:]) or (
                    xmin.shape == x.shape
                ), f"BoundRepair got input with wrong shape for lower & upper bounds: {xmin.shape=} {xmax.shape=} {x.shape=}"
            else:
                assert (
                    x.shape[-1] == self.xmin.shape[-1]
                ), f"BoundRepair got input of shape {x.shape=} but expected {self.xmin.shape=}"

        if xmin is not None and xmax is not None:
            self.xmin = xmin
            self.xmax = xmax
            self.preprocess_bounds(2)

        return self._forward(x)

    def none(self, x: Tensor):
        """no-op, just return x"""
        return x

    def clamp(self, x: Tensor):
        r"""Bound repair function that uses `torch.clamp`.

        \[ \text{clamp}(x, \underline{x}, \overline{x}) \]

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        return torch.clamp(x, self.xmin, self.xmax)

    @staticmethod
    @torch.jit.script
    def double_relu(x: Tensor, xmin: Tensor, xmax: Tensor):
        r"""ReLU bound repair function for double-sided bounds.

        \[ \text{relu}(x - \underline{x}) - \text{relu}(x - \overline{x}) + \underline{x} \]

        Args:
            x (Tensor): Input tensor.
            xmin (Tensor): Lower bound.
            xmax (Tensor): Upper bound.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        return torch.relu(x - xmin) - torch.relu(x - xmax) + xmin

    @staticmethod
    @torch.jit.script
    def lower_relu(x: Tensor, xmin: Tensor):
        r"""ReLU bound repair function for lower bounds.

        \[ \text{relu}(x - \underline{x}) + \underline{x} \]

        Args:
            x (Tensor): Input tensor.
            xmin (Tensor): Lower bound.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        return torch.relu(x - xmin) + xmin

    @staticmethod
    @torch.jit.script
    def upper_relu(x: Tensor, xmax: Tensor):
        r"""ReLU bound repair function for upper bounds.

        \[ -\text{relu}(\overline{x} - x) + \overline{x} \]

        Args:
            x (Tensor): Input tensor.
            xmax (Tensor): Upper bound.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        return -torch.relu(xmax - x) + xmax

    def relu(self, x: Tensor):
        r"""Apply the ReLU-based bound repair functions to the input, supporting any combination of single- or double-sided bounds.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        y = torch.clone(x)
        y[..., self.lower_mask] = self.lower_relu(x[..., self.lower_mask], self.xmin_lower)
        y[..., self.upper_mask] = self.upper_relu(x[..., self.upper_mask], self.xmax_upper)
        y[..., self.double_mask] = self.double_relu(x[..., self.double_mask], self.xmin_double, self.xmax_double)
        return y

    @staticmethod
    @torch.jit.script
    def double_sigmoid(x: Tensor, xmin: Tensor, xmax: Tensor):
        r"""Sigmoid bound repair function for double-sided bounds.

        \[ \text{sigmoid}(x) \cdot (\overline{x} - \underline{x}) + \underline{x} \]

        Args:
            x (Tensor): Input tensor.
            xmin (Tensor): Lower bound.
            xmax (Tensor): Upper bound.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        return torch.sigmoid(x) * (xmax - xmin) + xmin

    def sigmoid(self, x: Tensor):
        r"""Apply the sigmoid bound repair function to the input, supporting only unbounded or double-sided bounds.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        y = torch.clone(x)
        y[..., self.double_mask] = self.double_sigmoid(x[..., self.double_mask], self.xmin_double, self.xmax_double)
        return y

    @staticmethod
    @torch.jit.script
    def double_softplus(x: Tensor, xmin: Tensor, xmax: Tensor):
        r"""Softplus bound repair function for double-sided bounds.

        \[ \text{softplus}(x - \underline{x}) - \text{softplus}(x - \overline{x}) + \underline{x} \]

        Args:
            x (Tensor): Input tensor.
            xmin (Tensor): Lower bound.
            xmax (Tensor): Upper bound.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        return F.softplus(x - xmin) - F.softplus(x - xmax) + xmin

    @staticmethod
    @torch.jit.script
    def lower_softplus(x: Tensor, xmin: Tensor):
        r"""Softplus bound repair function for lower bounds.

        \[ \text{softplus}(x - \underline{x}) + \underline{x} \]

        Args:
            x (Tensor): Input tensor.
            xmin (Tensor): Lower bound.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        return F.softplus(x - xmin) + xmin

    @staticmethod
    @torch.jit.script
    def upper_softplus(x: Tensor, xmax: Tensor):
        r"""Softplus bound repair function for upper bounds.

        \[ -\text{softplus}(\overline{x} - x) + \overline{x} \]

        Args:
            x (Tensor): Input tensor.
            xmax (Tensor): Upper bound.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        return -F.softplus(xmax - x) + xmax

    def softplus(self, x: Tensor):
        r"""Apply the softplus bound-clipping function to the input, supporting any combination of single- or double-sided bounds.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor satisfying the bounds."""
        y = torch.clone(x)
        y[..., self.lower_mask] = self.lower_softplus(x[..., self.lower_mask], self.xmin_lower)
        y[..., self.upper_mask] = self.upper_softplus(x[..., self.upper_mask], self.xmax_upper)
        y[..., self.double_mask] = self.double_softplus(x[..., self.double_mask], self.xmin_double, self.xmax_double)
        return y

    @staticmethod
    @torch.jit.script
    def double_tanh(x: Tensor, xmin: Tensor, xmax: Tensor):
        r"""Tanh bound repair function for double-sided bounds.

        \[ (\frac{1}{2} \tanh(x) + \frac{1}{2}) \cdot (\overline{x} - \underline{x}) + \underline{x} \]

        Args:
            x (Tensor): Input tensor.
            xmin (Tensor): Lower bound.
            xmax (Tensor): Upper bound.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        return (0.5 * torch.tanh(x) + 0.5) * (xmax - xmin) + xmin

    def tanh(self, x: Tensor):
        r"""Apply the tanh bound-clipping function to the input, supporting only unbounded or double-sided bounds.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor satisfying the bounds.
        """
        y = torch.clone(x)
        y[..., self.double_mask] = self.double_tanh(x[..., self.double_mask], self.xmin_double, self.xmax_double)
        return y

    def preprocess_bounds(self, memory_efficient: int):
        """Pre-computes masks and pre-indexes bounds depending on `memory_efficient` level.

        Args:
            memory_efficient (int):

            `0`: (fastest, most memory) pre-compute masks and index bounds

            `1`: pre-compute masks only

            `2`: (slowest, least memory) do not pre-compute anything
        """
        if hasattr(self, "_memory_mode") and ((self._memory_mode == 2) and (memory_efficient == 2)):
            return

        self._properties = {}

        for k, v in {
            "lower_mask": lambda self: (self.xmin.isfinite() & ~self.xmax.isfinite()),
            "upper_mask": lambda self: (~self.xmin.isfinite() & self.xmax.isfinite()),
            "double_mask": lambda self: (self.xmin.isfinite() & self.xmax.isfinite()),
            "none_mask": lambda self: (~self.xmin.isfinite() & ~self.xmax.isfinite()),
        }.items():
            if memory_efficient == 1 or memory_efficient == 0:
                self.register_buffer(k, v(self), persistent=False)
            else:
                self._properties[k] = v

        for k, v in {
            "xmin_lower": lambda self: self.xmin[self.lower_mask],
            "xmax_upper": lambda self: self.xmax[self.upper_mask],
            "xmin_double": lambda self: self.xmin[self.double_mask],
            "xmax_double": lambda self: self.xmax[self.double_mask],
        }.items():
            if memory_efficient == 0:
                self.register_buffer(k, v(self), persistent=False)
            else:
                self._properties[k] = v

        self._memory_mode = memory_efficient

    def __getattr__(self, name: str):
        if "_properties" in self.__dict__:
            _properties = self.__dict__["_properties"]
            if name in _properties:
                return _properties[name](self)
        return super().__getattr__(name)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Loads the state dictionary and re-initializes the pre-computed quantities."""
        unmatched = super().load_state_dict(state_dict, False)
        if set(unmatched.unexpected_keys) == {"xmin", "xmax"}:
            self.register_buffer("xmin", state_dict["xmin"])
            self.register_buffer("xmax", state_dict["xmax"])
        elif set(unmatched.missing_keys) == {"xmin", "xmax"}:
            pass
        elif strict and unmatched.unexpected_keys or unmatched.missing_keys:
            raise ValueError(
                f"Unexpected keys in state_dict. Unexpected: {unmatched.unexpected_keys}, missing: {unmatched.missing_keys}"
            )
        self._forward = getattr(self, self.SUPPORTED_METHODS[self.method_idx])
        self.preprocess_bounds(state_dict["memory_efficient"])
        return self
