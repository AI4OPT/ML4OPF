"""
ML4OPF: Machine Learning for OPF
------------------------------------

This repository contains a collection of tools for applying machine learning
to the optimal power flow (OPF) problem. Below are some common usage patterns:


## Loading data
This is probably the most common usage, especially for those who
already have their own models and wish to evaluate on the PGLearn
datasets. ML4OPF makes loading data and splitting training/testing sets
easy and reproducible.
```python
from ml4opf import DCProblem

data_dir = ... # path to folder containing the data

problem = DCProblem(data_dir, **kwargs)

# extract tensors
train_pd = problem.train_data["input/pd"]
train_pg = problem.train_data["primal/pg"]
train_va = problem.train_data["primal/va"]

test_pd = problem.test_data["input/pd"]
test_pg = problem.test_data["primal/pg"]
test_va = problem.test_data["primal/va"]

# create a PyTorch dataset
torch_dataset, slices = problem.make_dataset()
```

## Computing residuals
The ML4OPF OPFViolation modules provide a fast (using `torch.jit`),
standard, and convenient way to: calculate the residuals/violations
of the OPF constraints, compute the objective function,
and other useful problem data such as incidence matrices.

```python
v = problem.violation
pg_lower, pg_upper = v.pg_bound_residual(train_pg) # supply clamp=True to report violations only
obj = v.objective(train_pg)
gen_incidence = v.generator_incidence
```

Note that you can use the underlying functions directly without instantiating
the OPFViolation class by accessing `ml4opf.functional`.
This allows to perform the calculations without using the data parsing or caching logic,
but requires the user to adopt the functional interface (`ml4opf.functional`) vs. the object-oriented interface (`ml4opf.formulations`).

```python
import ml4opf.functional as MOF
gen_incidence = MOF.generator_incidence(v.gen_bus, v.n_bus, v.n_gen)
obj = MOF.DCP.objective(train_pg, v.c0, v.c1)
```

## Implementing an OPFModel
In order to use the ML4OPF evaluation tools, you need to subclass the
`OPFModel` class and implement a few methods. The typical pattern is to
first write your model in the typical PyTorch fashion - subclassing `torch.nn.Module`.
Then, subclass `OPFModel` and implement the required methods. Below is an example
where the original model is `MyPyTorchModel` and the wrapper is `MyDCPModel`.
```python
import torch
from ml4opf import DCPModel

N_LOADS = problem.violation.n_load
N_GEN = problem.violation.n_gen
N_BUS = problem.violation.n_bus

class MyPyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(N_LOADS, 16)
        self.fc2 = torch.nn.Linear(16, N_GEN)
        self.fc3 = torch.nn.Linear(16, N_BUS)

    def forward(self, pd):
        x = torch.relu(self.fc1(pd))
        pg_pred = self.fc2(x)
        va_pred = self.fc3(x)
        return pg_pred, va_pred

class MyDCPModel(DCPModel):
    def __init__(self, pytorch_model, problem):
        super().__init__()
        self.model = pytorch_model
        self.problem = problem

    def save_checkpoint(self, path_to_folder):
        torch.save(self.model.state_dict(), f"{path_to_folder}/model.pth")

    @classmethod
    def load_from_checkpoint(cls, path_to_folder, problem):
        pytorch_model = MyPyTorchModel()
        pytorch_model.load_state_dict(torch.load(f"{path_to_folder}/model.pth"))
        return cls(pytorch_model, problem)

    def predict(self, pd):
        pg, va = self.model(pd)
        return {"pg": pg, "va": va}
```

## Using repair layers
A common issue with learning OPF is that the model may predict
infeasible solutions. The `ml4opf.layers` module provides a collection
of differentiable layers that can be used to repair infeasible solutions. For example,
the `BoundRepair` layer can be used to repair solutions that violate
bound constraints. The output of `BoundRepair` is guaranteed to be within
the specified bounds.

```python
from ml4opf.layers import BoundRepair

class BoundRepairPyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(N_LOADS, 16)
        self.fc2 = torch.nn.Linear(16, N_GEN)
        self.fc3 = torch.nn.Linear(16, N_BUS)
        self.bound_repair = BoundRepair(xmin=v.pmin, xmax=v.pmax, method="softplus")

    def forward(self, pd):
        x = torch.relu(self.fc1(pd))
        pg_pred = self.bound_repair(self.fc2(x))
        va_pred = self.fc3(x)
        return pg_pred, va_pred
```

The source code is organized into
several submodules:

"""

## setup logging
import logging, os

logger = logging.getLogger(__name__)
LOGLEVEL = os.environ.get("ML4OPF_LOGLEVEL", "WARNING").upper()
logger.setLevel(LOGLEVEL)
try:
    from rich.logging import RichHandler

    handler = RichHandler(rich_tracebacks=True, enable_link_path=False)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
except ImportError:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s", "%m-%d %H:%M:%S")
    )
handler.setLevel(LOGLEVEL)
logger.addHandler(handler)


def warn(msg: str, stacklevel: int = 2, **kwargs):
    logger.warning(msg, stacklevel=stacklevel, **kwargs)


def info(msg: str, stacklevel: int = 2, **kwargs):
    logger.info(msg, stacklevel=stacklevel, **kwargs)


def debug(msg: str, stacklevel: int = 2, **kwargs):
    logger.debug(msg, stacklevel=stacklevel, **kwargs)


## import torch
try:
    debug("Importing PyTorch...")
    import torch

    debug("Imported PyTorch.")
except ImportError as e:
    raise ImportError(
        "Could not import PyTorch. Please follow the installation instructions in the ML4OPF README."
    ) from e

__all__ = []

import ml4opf.functional as MOF
from ml4opf import layers
from ml4opf import loss_functions
from ml4opf import formulations
from ml4opf import models
from ml4opf import parsers
from ml4opf import viz

__all__.extend(["MOF", "layers", "loss_functions", "formulations", "models", "parsers", "viz"])

from ml4opf.layers import *
from ml4opf.loss_functions import *
from ml4opf.formulations import *
from ml4opf.parsers import *

# NOTE: ml4opf.functional.__all__ is not exported
__all__.extend(layers.__all__)
__all__.extend(loss_functions.__all__)
__all__.extend(formulations.__all__)
__all__.extend(parsers.__all__)

debug("Imported ml4opf.")
