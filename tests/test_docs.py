from pathlib import Path
from ml4opf import __path__ as ml4opf_path


def test_docs_exist():
    import ml4opf

    for i in ml4opf.__all__:
        obj = getattr(ml4opf, i)
        assert obj.__doc__ is not None, f"Missing docstring for {i}"


def test_docs_index():
    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"

    from ml4opf import DCProblem

    problem = DCProblem(data_dir)

    # extract tensors
    train_pd = problem.train_data["input/pd"]
    train_pg = problem.train_data["primal/pg"]
    train_va = problem.train_data["primal/va"]

    test_pd = problem.test_data["input/pd"]
    test_pg = problem.test_data["primal/pg"]
    test_va = problem.test_data["primal/va"]

    # create a PyTorch dataset
    torch_dataset, slices = problem.make_dataset()

    v = problem.violation
    pg_lower, pg_upper = v.pg_bound_residual(train_pg)  # supply clamp=True to report violations only
    obj = v.objective(train_pg)
    gen_incidence = v.generator_incidence

    import torch
    from ml4opf import DCModel

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

    class MyDCPModel(DCModel):
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

    import ml4opf.functional as MOF

    gen_incidence = MOF.generator_incidence(v.gen_bus, v.n_bus, v.n_gen)
    obj = MOF.DC.objective(train_pg, v.c0, v.c1)
