import torch

from torch import Tensor


def branch_incidence(fbus: Tensor, tbus: Tensor, n_bus: int, n_branch: int) -> Tensor:
    """Sparse branch incidence matrix.

    Each row corresponds to a bus and each column corresponds to a branch.
    The value is 1 if the branch is from the bus, -1 if the branch is to the bus, and 0 otherwise.

    Args:
        fbus (Tensor): From bus indices. (nbranch,)
        tbus (Tensor): To bus indices. (nbranch,)
        n_bus (int): Number of buses.
        n_branch (int): Number of branches.

    Returns:
        Tensor: Sparse branch incidence matrix. (nbus, nbranch)

    """
    i = torch.cat((fbus, tbus), dim=0)

    j1 = torch.arange(n_branch, dtype=torch.long)
    j = torch.cat((j1, j1), dim=0)

    indices = torch.stack((i, j), dim=0)
    values = torch.cat((torch.ones(n_branch), -torch.ones(n_branch)), dim=0)
    size = (n_bus, n_branch)

    branch_incidence_matrix = torch.sparse_coo_tensor(indices, values, size).to(fbus.device)

    return branch_incidence_matrix


def branch_from_incidence(fbus: Tensor, n_bus: int, n_branch: int) -> Tensor:
    """Sparse branch from incidence matrix.

    Each row corresponds to a bus and each column corresponds to a branch.
    The value is 1 if the branch is from the bus, and 0 otherwise.

    Args:
        fbus (Tensor): From bus indices. (nbranch,)
        n_bus (int): Number of buses.
        n_branch (int): Number of branches.

    Returns:
        Tensor: Sparse branch from incidence matrix. (nbus, nbranch)
    """
    indices = torch.stack((fbus, torch.arange(n_branch, dtype=torch.long)), dim=0)
    values = torch.ones(n_branch)
    size = (n_bus, n_branch)

    branch_from_incidence_matrix = torch.sparse_coo_tensor(indices, values, size).to(fbus.device)

    return branch_from_incidence_matrix


def branch_to_incidence(tbus: Tensor, n_bus: int, n_branch: int) -> Tensor:
    """Sparse branch to incidence matrix.

    Each row corresponds to a bus and each column corresponds to a branch.
    The value is 1 if the branch is to the bus, and 0 otherwise.

    Args:
        tbus (Tensor): To bus indices. (nbranch,)
        n_bus (int): Number of buses.
        n_branch (int): Number of branches.

    Returns:
        Tensor: Sparse branch to incidence matrix. (nbus, nbranch)
    """
    indices = torch.stack((tbus, torch.arange(n_branch, dtype=torch.long)), dim=0)
    values = torch.ones(n_branch)
    size = (n_bus, n_branch)

    branch_to_incidence_matrix = torch.sparse_coo_tensor(indices, values, size).to(tbus.device)

    return branch_to_incidence_matrix


def generator_incidence(gen_bus: Tensor, n_bus: int, n_gen: int) -> Tensor:
    """Sparse generator incidence matrix.

    Each row corresponds to a bus and each column corresponds to a generator.
    The value is 1 if the generator is at the bus, and 0 otherwise.

    Args:
        gen_bus (Tensor): Generator bus indices. (ngen,)
        n_bus (int): Number of buses.
        n_gen (int): Number of generators.

    Returns:
        Tensor: Sparse generator incidence matrix. (nbus, ngen)
    """
    indices = torch.stack((gen_bus, torch.arange(n_gen, dtype=torch.long)), dim=0)
    values = torch.ones(n_gen)

    generator_incidence_matrix = torch.sparse_coo_tensor(indices, values, (n_bus, n_gen))

    return generator_incidence_matrix


def load_incidence(load_bus: Tensor, n_bus: int, n_load: int) -> Tensor:
    """Sparse load incidence matrix.

    Each row corresponds to a bus and each column corresponds to a load.
    The value is 1 if the load is at the bus, and 0 otherwise.

    Args:
        load_bus (Tensor): Load bus indices. (nload,)
        n_bus (int): Number of buses.
        n_load (int): Number of loads.

    Returns:
        Tensor: Sparse load incidence matrix. (nbus, nload)
    """
    indices = torch.stack((load_bus, torch.arange(n_load, dtype=torch.long)), dim=0)
    values = torch.ones(n_load)
    size = (n_bus, n_load)

    load_incidence_matrix = torch.sparse_coo_tensor(indices, values, size)

    return load_incidence_matrix


def adjacency_matrix(fbus: Tensor, tbus: Tensor, n_bus: int, n_branch: int) -> Tensor:
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

    indices = torch.stack((fbus, tbus), dim=0)
    values = torch.ones(n_branch)
    size = (n_bus, n_bus)

    adjacency_matrix = torch.sparse_coo_tensor(indices, values, size)

    return adjacency_matrix


@torch.jit.script
def map_to_bus_pad(x: Tensor, x_per_bus: Tensor):
    x_: Tensor = torch.nn.functional.pad(x, pad=(0, 1, 0, 0), mode="constant", value=0.0)
    x_bus = x_[:, x_per_bus].sum(dim=2)
    return x_bus


@torch.jit.script
def map_to_bus_matrix(x: Tensor, x_matrix: Tensor):
    return x @ x_matrix.swapaxes(0, 1)
