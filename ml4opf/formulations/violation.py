""" Abstract base class for OPFViolation classes.

Each formulation should define a class that inherits from `OPFViolation` and implements the following methods:

- `calc_violations`: Calculate the violations of all the constraints. Returns a dictionary of tensors.

- `violations_shapes`: Return the shapes of the violations as a dictionary.

- `objective`: Compute the objective value for a batch of samples.
"""

from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor

import ml4opf.functional as MOF


class OPFViolation(nn.Module, ABC):
    """
    The `OPFViolation` class is where all the problem expressions (objective, constraints, etc.) are defined.
    The classes provide a convenient interface for when the only varying quantity in the formulation
    per sample is the load demand `pd`. If other quantities vary, the user should use the functional
    interface at `ml4opf.functional`.

    When `clamp` is true, the values of g(x) are clamped to be non-negative and the values of h(x) are absolute-valued.
    Otherwise the raw values are returned.

    `OPFViolation` is a `torch.nn.Module`; all tensors used in computation are registered as non-persistent buffers.
    To move the module to a different device, use `.to(device)` as you would with any other `nn.Module`.
    Make sure that when you pass data to `OPFViolation`, it is on the same device as `OPFViolation`.
    """

    SUPPORTED_REDUCTIONS = ("none", "mean", "sum", "max")

    def __init__(self, data: dict[str, Tensor]):
        super().__init__()

        for k, v in data.items():
            if k in ["config", "ref_solutions"]:
                continue

            self.register_buffer(k, v, persistent=False)

        self.n_bus = self.N
        self.n_gen = self.G
        self.n_load = self.L
        self.n_branch = self.E

        gen_bus = [None] * self.n_gen
        for i in range(self.n_bus):
            for g in self.bus_gens[i]:
                if g >= self.n_gen:
                    continue
                gen_bus[g] = i

        self.register_buffer("gen_bus", torch.as_tensor(gen_bus), persistent=False)

        load_bus = [None] * self.n_load
        for i in range(self.n_bus):
            for l in self.bus_loads[i]:
                if l >= self.n_load:
                    continue
                load_bus[l] = i

        self.register_buffer("load_bus", torch.as_tensor(load_bus), persistent=False)

    @abstractmethod
    def calc_violations(self, *args, reduction: str = "mean", clamp: bool = True) -> dict[str, Tensor]:
        """Calculate the violations of the constraints. Returns a dictionary of tensors."""

    @abstractmethod
    def violation_shapes(self) -> dict[str, int]:
        """Return the shapes of the violations returned by `OPFViolation.calc_violations`."""

    @abstractmethod
    def objective(self, *args) -> Tensor:
        """Compute the objective value for a batch of samples."""

    @staticmethod
    def reduce_violations(violations: dict[str, Tensor], reduction: str, dim: int = 1):
        """Apply mean/sum/max to every value in a dictionary."""
        for k, v in violations.items():
            violations[k] = OPFViolation.reduce_violation(v, reduction, dim=dim)

        return violations

    @staticmethod
    def reduce_violation(violation: Tensor, reduction: str, dim: int = 1):
        """Apply mean/sum/max to a single tensor."""
        if reduction == "mean":
            return violation.mean(dim=dim)
        if reduction == "sum":
            return violation.sum(dim=dim)
        if reduction == "max":
            return violation.max(dim=dim).values
        if reduction is None or reduction == "none":
            return violation
        raise ValueError(f"Invalid {reduction=}. Supported: {OPFViolation.SUPPORTED_REDUCTIONS}")

    def forward(self, *args, **kwargs):
        """Pass-through for `OPFViolation.calc_violations`"""
        return self.calc_violations(*args, **kwargs)

    @staticmethod
    def clamped_bound_residual(x: Tensor, xmin: Tensor, xmax: Tensor, clamp: bool):
        g_lower, g_upper = MOF.bound_residual(x, xmin, xmax)

        return (
            MOF.inequality_violation(g_lower, clamp=clamp),
            MOF.inequality_violation(g_upper, clamp=clamp),
        )

    @property
    def branch_incidence(self) -> Tensor:
        """Sparse branch incidence matrix.

        Each row corresponds to a bus and each column corresponds to a branch.
        The value is 1 if the branch is from the bus, -1 if the branch is to the bus, and 0 otherwise.

        Returns:
            Tensor: Sparse branch incidence matrix. (nbus, nbranch)

        """
        if "_branch_incidence_buffer" in self._buffers:
            return self._branch_incidence_buffer

        self.register_buffer(
            "_branch_incidence_buffer",
            MOF.branch_incidence(self.bus_fr, self.bus_to, self.N, self.E),
            persistent=False,
        )

        return self._branch_incidence_buffer

    @property
    def branch_from_incidence(self) -> Tensor:
        """Sparse branch from incidence matrix.

        Each row corresponds to a bus and each column corresponds to a branch.
        The value is 1 if the branch is from the bus, and 0 otherwise.

        Returns:
            Tensor: Sparse branch from incidence matrix. (nbus, nbranch)
        """
        if "_branch_from_incidence_buffer" in self._buffers:
            return self._branch_from_incidence_buffer

        self.register_buffer(
            "_branch_from_incidence_buffer",
            MOF.branch_from_incidence(self.bus_fr, self.N, self.E),
            persistent=False,
        )

        return self._branch_from_incidence_buffer

    @property
    def branch_to_incidence(self) -> Tensor:
        """Sparse branch to incidence matrix.

        Each row corresponds to a bus and each column corresponds to a branch.
        The value is 1 if the branch is to the bus, and 0 otherwise.

        Returns:
            Tensor: Sparse branch to incidence matrix. (nbus, nbranch)
        """
        if "_branch_to_incidence_buffer" in self._buffers:
            return self._branch_to_incidence_buffer

        self.register_buffer(
            "_branch_to_incidence_buffer",
            MOF.branch_to_incidence(self.bus_to, self.N, self.E),
            persistent=False,
        )

        return self._branch_to_incidence_buffer

    @property
    def generator_incidence(self) -> Tensor:
        """Sparse generator incidence matrix.

        Each row corresponds to a bus and each column corresponds to a generator.
        The value is 1 if the generator is at the bus, and 0 otherwise.

        Returns:
            Tensor: Sparse generator incidence matrix. (nbus, ngen)
        """
        if "_generator_incidence_buffer" in self._buffers:
            return self._generator_incidence_buffer

        self.register_buffer(
            "_generator_incidence_buffer",
            MOF.generator_incidence(self.gen_bus, self.n_bus, self.n_gen),
            persistent=False,
        )

        return self._generator_incidence_buffer

    @property
    def load_incidence(self) -> Tensor:
        """Sparse load incidence matrix.

        Each row corresponds to a bus and each column corresponds to a load.
        The value is 1 if the load is at the bus, and 0 otherwise.

        Returns:
            Tensor: Sparse load incidence matrix. (nbus, nload)
        """
        if "_load_incidence_buffer" in self._buffers:
            return self._load_incidence_buffer

        self.register_buffer(
            "_load_incidence_buffer",
            MOF.load_incidence(self.load_bus, self.n_bus, self.n_load),
            persistent=False,
        )

        return self._load_incidence_buffer

    @property
    def adjacency_matrix(self) -> Tensor:
        """Sparse adjacency matrix.

        Each row corresponds to a bus and each column corresponds to a bus.
        The value is 1 if there is a branch between the buses, and 0 otherwise.

        Args:
            fbus (Tensor): From bus indices. (nbranch,)
            tbus (Tensor): To bus indices. (nbranch,)
            n_bus (int): Number of buses.
            n_branch (int): Number of branches.

        Returns:
            Tensor: Sparse adjacency matrix. (nbus, nbus)
        """
        if "_adjacency_buffer" in self._buffers:
            return self._adjacency_buffer

        self.register_buffer(
            "_adjacency_buffer",
            MOF.adjacency(
                self.bus_fr,
                self.bus_to,
                self.N,
                self.E,
            ),
            persistent=False,
        )

        return self._adjacency_buffer

    @property
    def branch_incidence_dense(self) -> Tensor:
        if "_branch_incidence_dense_buffer" in self._buffers:
            return self._branch_incidence_dense_buffer

        self.register_buffer(
            "_branch_incidence_dense_buffer",
            self.branch_incidence.to_dense(),
            persistent=False,
        )
        return self._branch_incidence_dense_buffer

    @property
    def branch_from_incidence_dense(self) -> Tensor:
        if "_branch_from_incidence_dense_buffer" in self._buffers:
            return self._branch_from_incidence_dense_buffer

        self.register_buffer(
            "_branch_from_incidence_dense_buffer",
            self.branch_from_incidence.to_dense(),
            persistent=False,
        )
        return self._branch_from_incidence_dense_buffer

    @property
    def branch_to_incidence_dense(self) -> Tensor:
        if "_branch_to_incidence_dense_buffer" in self._buffers:
            return self._branch_to_incidence_dense_buffer

        self.register_buffer(
            "_branch_to_incidence_dense_buffer",
            self.branch_to_incidence.to_dense(),
            persistent=False,
        )
        return self._branch_to_incidence_dense_buffer

    @property
    def generator_incidence_dense(self) -> Tensor:
        if "_generator_incidence_dense_buffer" in self._buffers:
            return self._generator_incidence_dense_buffer

        self.register_buffer(
            "_generator_incidence_dense_buffer",
            self.generator_incidence.to_dense(),
            persistent=False,
        )
        return self._generator_incidence_dense_buffer

    @property
    def load_incidence_dense(self) -> Tensor:
        if "_load_incidence_dense_buffer" in self._buffers:
            return self._load_incidence_dense_buffer

        self.register_buffer(
            "_load_incidence_dense_buffer",
            self.load_incidence.to_dense(),
            persistent=False,
        )
        return self._load_incidence_dense_buffer

    @property
    def adjacency_matrix_dense(self) -> Tensor:
        if "_adjacency_matrix_dense" in self._buffers:
            return self._adjacency_matrix_dense

        self.register_buffer(
            "_adjacency_matrix_dense",
            self.adjacency_matrix.to_dense(),
            persistent=False,
        )
        return self._adjacency_matrix_dense

    def gen_to_bus(self, pg_or_qg: Tensor, method: str = "pad") -> Tensor:
        """Embed generator-wise values to bus-wise.

        The default method "pad" sums over any generators at the same bus.
        The matrix methods "dense_matrix" and "matrix" use the incidence matrix.

        Args:
            pg_or_qg (Tensor): Generator-wise values. (batch, ngen)
            method (str): Method to use. Supported: ['pad', 'dense_matrix', 'matrix']

        Returns:
            Tensor: Bus-wise values. (batch, nbus)
        """
        if method == "dense_matrix":
            return MOF.map_to_bus_matrix(pg_or_qg, self.generator_incidence_dense)
        elif method == "matrix":
            return MOF.map_to_bus_matrix(pg_or_qg, self.generator_incidence)
        elif method == "pad":
            return MOF.map_to_bus_pad(pg_or_qg, self.bus_gens)
        else:
            raise ValueError(f"Invalid {method=}. Supported: 'pad', 'dense_matrix', 'matrix'")

    def load_to_bus(self, pd_or_qd: Tensor, method: str = "pad") -> Tensor:
        """Embed load-wise values to bus-wise.

        The default method "pad" sums over any loads at the same bus.
        The matrix methods "dense_matrix" and "matrix" use the incidence matrix.

        Args:
            pd_or_qd (Tensor): Load-wise values. (batch, ngen)
            method (str): Method to use. Supported: ['pad', 'dense_matrix', 'matrix']

        Returns:
            Tensor: Bus-wise values. (batch, nbus)
        """
        if method == "dense_matrix":
            return MOF.map_to_bus_matrix(pd_or_qd, self.load_incidence_dense)
        elif method == "matrix":
            return MOF.map_to_bus_matrix(pd_or_qd, self.load_incidence)
        elif method == "pad":
            return MOF.map_to_bus_pad(pd_or_qd, self.bus_loads)
        else:
            raise ValueError(f"Invalid {method=}. Supported: 'pad', 'dense_matrix', 'matrix'")

    def branch_from_to_bus(self, pf_or_qf: Tensor, method: str = "pad") -> Tensor:
        """Embed batched branch-wise values to batched bus-wise.

        The default method "pad" sums over any flows on branches from the same bus.
        The matrix methods "dense_matrix" and "matrix" use the incidence matrix.

        Args:
            pf_or_qf (Tensor): Branch-wise values. (batch, nbranch)
            method (str): Method to use. Supported: ['pad', 'dense_matrix', 'matrix']

        Returns:
            Tensor: Bus-wise values. (batch, nbus)
        """
        if method == "dense_matrix":
            return MOF.map_to_bus_matrix(pf_or_qf, self.branch_from_incidence_dense)
        elif method == "matrix":
            return MOF.map_to_bus_matrix(pf_or_qf, self.branch_from_incidence)
        elif method == "pad":
            return MOF.map_to_bus_pad(pf_or_qf, self.bus_arcs_fr)
        else:
            raise ValueError(f"Invalid {method=}. Supported: 'pad', 'dense_matrix', 'matrix'")

    def branch_to_to_bus(self, pt_or_qt: Tensor, method: str = "pad") -> Tensor:
        """Embed batched branch-wise values to batched bus-wise.

        The default method "pad" sums over any flows on branches to the same bus.
        The matrix methods "dense_matrix" and "matrix" use the incidence matrix.

        Args:
            pt_or_qt (Tensor): Branch-wise values. (batch, nbranch)
            method (str): Method to use. Supported: ['pad', 'dense_matrix', 'matrix']

        Returns:
            Tensor: Bus-wise values. (batch, nbus)
        """
        if method == "dense_matrix":
            return MOF.map_to_bus_matrix(pt_or_qt, self.branch_to_incidence_dense)
        elif method == "matrix":
            return MOF.map_to_bus_matrix(pt_or_qt, self.branch_to_incidence)
        elif method == "pad":
            return MOF.map_to_bus_pad(pt_or_qt, self.bus_arcs_to)
        else:
            raise ValueError(f"Invalid {method=}. Supported: 'pad', 'dense_matrix', 'matrix'")
