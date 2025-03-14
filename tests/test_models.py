from typing import Optional


def test_models():

    import torch

    from ml4opf import ACProblem, DCProblem
    from ml4opf.formulations.ed.problem import EDProblem

    from pathlib import Path
    from ml4opf import __path__ as ml4opf_path

    data_dir = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"

    ac_problem = ACProblem(data_directory=data_dir)
    dc_problem = DCProblem(data_directory=data_dir)
    ed_problem = EDProblem(data_directory=data_dir, ptdf_path=data_dir / "ptdf.h5")

    from ml4opf.formulations import OPFProblem
    from ml4opf.models.basic_nn import ACBasicNeuralNet, DCBasicNeuralNet
    from ml4opf.models.basic_nn.ed_basic_nn import EDBasicNeuralNet
    from ml4opf.models.penalty_nn import ACPenaltyNeuralNet, DCPenaltyNeuralNet
    from ml4opf.models.penalty_nn.ed_penalty_nn import EDPenaltyNeuralNet
    from ml4opf.models.ldf_nn import ACLDFNeuralNet, DCLDFNeuralNet
    from ml4opf.models.ldf_nn.ed_ldf_nn import EDLDFNeuralNet
    from ml4opf.models.e2elr.e2elr import EDE2ELRNeuralNet
    from ml4opf.models.pca_nn import ACPCANeuralNet, DCPCANeuralNet, EDPCANeuralNet

    def make_model(problem: OPFProblem, config: dict, kind="basic", loss_config: Optional[dict] = None):
        if loss_config is None:
            loss_config = dict()

        if kind == "basic":
            if isinstance(problem, ACProblem):
                cls = ACBasicNeuralNet
            elif isinstance(problem, DCProblem):
                cls = DCBasicNeuralNet
            elif isinstance(problem, EDProblem):
                cls = EDBasicNeuralNet
        elif kind == "penalty":
            if isinstance(problem, ACProblem):
                cls = ACPenaltyNeuralNet
            elif isinstance(problem, DCProblem):
                cls = DCPenaltyNeuralNet
            elif isinstance(problem, EDProblem):
                cls = EDPenaltyNeuralNet
        elif kind == "ldf":
            if isinstance(problem, ACProblem):
                cls = ACLDFNeuralNet
            elif isinstance(problem, DCProblem):
                cls = DCLDFNeuralNet
            elif isinstance(problem, EDProblem):
                cls = EDLDFNeuralNet
        elif kind == "e2elr":
            assert isinstance(problem, EDProblem)
            cls = EDE2ELRNeuralNet
        elif kind == "pca":
            if isinstance(problem, ACProblem):
                cls = ACPCANeuralNet
            elif isinstance(problem, DCProblem):
                cls = DCPCANeuralNet
            elif isinstance(problem, EDProblem):
                cls = EDPCANeuralNet
        else:
            raise ValueError(f"Unknown model kind: {kind}. Must be one of 'basic', 'ldf', or 'penalty'.")

        return cls(config=dict(**config, **loss_config), problem=problem)

    from ml4opf.models.basic_nn.basic_nn import BasicNeuralNet  # all the models are subclasses of this
    from pathlib import Path

    def train(model: BasicNeuralNet, epochs: int = 100):
        from datetime import datetime
        from lightning.pytorch.loggers import CSVLogger

        start = datetime.now()
        print(f"Starting training {model.__class__.__name__} at {start}")

        if hasattr(model, "trainer"):
            model.trainer.fit_loop.max_epochs += epochs
        else:
            model.make_dataset(dl_kwargs=dict(persistent_workers=True, num_workers=1))
            model.make_trainer(
                accelerator="auto" if torch.cuda.is_available() else "cpu",  # mps is slower than cpu!
                max_epochs=epochs,
                enable_checkpointing=False,
                logger=CSVLogger("logs", name=model.__class__.__name__),
                enable_progress_bar=False,
            )

        model.train()
        end = datetime.now()
        print(f"Finished training {model.__class__.__name__} at {end} -- took {end - start}")

    def evaluate_model(model: BasicNeuralNet):
        return {
            k: v.item() for k, v in model.evaluate_model(reduction="mean", inner_reduction="sum").items()
        }  # sum over components, mean over samples

    def save_checkpoint(model: BasicNeuralNet):
        ckpt_dir = Path(model.trainer.logger.log_dir) / "checkpoint"
        model.save_checkpoint(ckpt_dir)

    def load_checkpoint(model: BasicNeuralNet):
        ckpt_dir = Path(model.trainer.logger.log_dir) / "checkpoint"
        return model.__class__.load_from_checkpoint(ckpt_dir, model.problem)

    def ac_predict(model: BasicNeuralNet):
        pred = model.predict(ac_problem.test_data["input/pd"], ac_problem.test_data["input/qd"])

        pred_pg, pred_qg, pred_vm, pred_va = pred["pg"], pred["qg"], pred["vm"], pred["va"]
        pred_pf, pred_pt, pred_qf, pred_qt = ac_problem.violation.flows_from_voltage_bus(pred["vm"], pred["va"])

        return {
            "pg": pred_pg,
            "qg": pred_qg,
            "vm": pred_vm,
            "va": pred_va,
            "pf": pred_pf,
            "pt": pred_pt,
            "qf": pred_qf,
            "qt": pred_qt,
        }

    def dc_predict(model: BasicNeuralNet):
        pred = model.predict(dc_problem.test_data["input/pd"])

        pred_pg, pred_va = pred["pg"], pred["va"]
        pred_pf = model.violation.pf_from_va(pred["va"])

        return {
            "pg": pred_pg,
            "va": pred_va,
            "pf": pred_pf,
        }

    def ed_predict(model: BasicNeuralNet):
        pred = model.predict(ed_problem.test_data["input/pd"])

        pred_pg = pred["pg"]
        pred_pf = model.violation.pf_from_pdpg(ed_problem.test_data["input/pd"], pred["pg"])

        return {
            "pg": pred_pg,
            "pf": pred_pf,
        }

    config = {
        "optimizer": "adam",
        "loss": "l1",
        "hidden_sizes": [8, 8],
        "activation": "sigmoid",
        "boundrepair": "softplus",
        "learning_rate": 1e-3,
    }

    ac_basic_nn = make_model(ac_problem, config, kind="basic")
    dc_basic_nn = make_model(dc_problem, config, kind="basic")
    ed_basic_nn = make_model(ed_problem, config, kind="basic")

    e2elr_nn = make_model(ed_problem, config, kind="e2elr")
    ac_pca_nn = make_model(ac_problem, config, kind="pca")
    dc_pca_nn = make_model(dc_problem, config, kind="pca")
    ed_pca_nn = make_model(ed_problem, config, kind="pca")

    ldf_config = {
        "step_size": 1e-2,
        "kickin": 0,
        "update_freq": 1,
        "divide_by_counter": True,
        "exclude_keys": [],
    }

    ac_ldf_nn = make_model(ac_problem, config, kind="ldf", loss_config=ldf_config)
    dc_ldf_nn = make_model(dc_problem, config, kind="ldf", loss_config=ldf_config)
    ed_ldf_nn = make_model(ed_problem, config, kind="ldf", loss_config=ldf_config)

    penalty_config = {
        "exclude_keys": [],
        "multipliers": 1e-2,
    }

    ac_penalty_nn = make_model(ac_problem, config, kind="penalty", loss_config=penalty_config)
    dc_penalty_nn = make_model(dc_problem, config, kind="penalty", loss_config=penalty_config)
    ed_penalty_nn = make_model(ed_problem, config, kind="penalty", loss_config=penalty_config)

    models = [
        ac_basic_nn,
        dc_basic_nn,
        ed_basic_nn,
        e2elr_nn,
        ac_pca_nn,
        dc_pca_nn,
        ed_pca_nn,
        ac_ldf_nn,
        dc_ldf_nn,
        ed_ldf_nn,
        ac_penalty_nn,
        dc_penalty_nn,
        ed_penalty_nn,
    ]

    evals = {}

    for model in models:
        train(model, epochs=2)
        save_checkpoint(model)
        evals[model.__class__.__name__] = evaluate_model(model)

    ac_predict(ac_basic_nn)
    dc_predict(dc_basic_nn)
    ed_predict(ed_basic_nn)
    ed_predict(e2elr_nn)

    ac_predict(ac_pca_nn)
    dc_predict(dc_pca_nn)
    ed_predict(ed_pca_nn)

    ac_predict(ac_ldf_nn)
    dc_predict(dc_ldf_nn)
    ed_predict(ed_ldf_nn)

    load_checkpoint(ac_basic_nn)
    load_checkpoint(dc_ldf_nn)
    load_checkpoint(ed_penalty_nn)

    ac_predict(ac_penalty_nn)
    dc_predict(dc_penalty_nn)
    ed_predict(ed_penalty_nn)

    config["optimizer"] = "adamw"
    make_model(ac_problem, config, kind="basic")

    config["optimizer"] = "sgd"
    make_model(dc_problem, config, kind="ldf", loss_config=ldf_config)

    config["optimizer"] = "nonexistent"
    make_model(ed_problem, config, kind="penalty", loss_config=penalty_config)
