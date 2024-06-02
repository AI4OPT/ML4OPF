import sys, pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path


@pytest.mark.skip("Does not run on CI")
def test_basic_usage():

    import torch

    # load data
    from ml4opf import ACPProblem

    data_path = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    network = "300_ieee"

    problem = ACPProblem(data_path, network, dataset_name="ACOPF", test_set_size=10)

    # make a basic neural network model
    from ml4opf.models.acp.basic_nn import BasicNeuralNet  # requires pytorch-lightning

    config = {
        "optimizer": "adam",
        "init_lr": 1e-3,
        "loss": "mse",
        "hidden_sizes": [50, 30, 50],  # encoder-decoder structure
        "activation": "sigmoid",
        "boundrepair": "none",  # optionally clamp outputs to bounds (choices: "sigmoid", "relu", "clamp")
    }

    model = BasicNeuralNet(config, problem)

    model.train(trainer_kwargs={"max_epochs": 5, "accelerator": "auto"})

    evals = model.evaluate_model()

    from ml4opf.viz import make_stats_df

    print(make_stats_df(evals))

    model.save_checkpoint(
        "basic_300bus"
    )  # creates a folder called "basic_300bus" with two files in it, trainer.ckpt and config.json


def test_advanced_usage():
    import logging

    logging.basicConfig(level=logging.INFO)
    import torch

    from ml4opf import ACPProblem

    data_path = Path(ml4opf_path[0]).parent / "tests" / "test_data"
    network = "300_ieee"

    problem = ACPProblem(data_path, network, dataset_name="ACOPF", test_set_size=10)

    # get train/test set:
    train_data = problem.train_data
    test_data = problem.test_data

    train_data["input/pd"].shape  # torch.Size([52863, 201])
    test_data["input/pd"].shape  # torch.Size([5000, 201])

    # if needed, convert the HDF5 data to a tree dictionary instead of a flat dictionary:
    from ml4opf.parsers import H5Parser

    h5_tree = H5Parser.make_tree(train_data)  # this tree structure should
    # exactly mimic the
    # structure of the HDF5 file.
    h5_tree["input"]["pd"].shape  # torch.Size([52863, 201])
