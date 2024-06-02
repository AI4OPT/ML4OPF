import torch.nn as nn
from torch import Tensor
from abc import ABC

import ml4opf.functional as MOF


class IncidenceMixin(nn.Module, ABC):

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
            MOF.branch_incidence(self.fbus_per_branch, self.tbus_per_branch, self.n_bus, self.n_branch),
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
            MOF.branch_from_incidence(self.fbus_per_branch, self.n_bus, self.n_branch),
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
            MOF.branch_to_incidence(self.tbus_per_branch, self.n_bus, self.n_branch),
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
            MOF.adjacency_matrix(
                self.fbus_per_branch,
                self.tbus_per_branch,
                self.n_bus,
                self.n_branch,
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
            return MOF.map_to_bus_pad(pg_or_qg, self.gen_per_bus)
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
            return MOF.map_to_bus_pad(pd_or_qd, self.load_per_bus)
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
            return MOF.map_to_bus_pad(pf_or_qf, self.branch_from_per_bus)
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
            return MOF.map_to_bus_pad(pt_or_qt, self.branch_to_per_bus)
        else:
            raise ValueError(f"Invalid {method=}. Supported: 'pad', 'dense_matrix', 'matrix'")
