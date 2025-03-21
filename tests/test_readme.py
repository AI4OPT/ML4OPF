import pytest

from pathlib import Path
from ml4opf import __path__ as ml4opf_path


def test_basic_usage():

    import torch

    # load data
    from ml4opf import ACProblem

    data_path = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"

    problem = ACProblem(data_path)

    # make a basic neural network model
    from ml4opf.models.basic_nn import ACBasicNeuralNet  # requires pytorch-lightning

    config = {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "loss": "mse",
        "hidden_sizes": [10, 5, 10],  # encoder-decoder structure
        "activation": "sigmoid",
        "boundrepair": "none",  # optionally clamp outputs to bounds (choices: "sigmoid", "relu", "clamp")
    }

    model = ACBasicNeuralNet(config, problem)

    model.train(trainer_kwargs={"max_epochs": 2, "accelerator": "cpu"})

    evals = model.evaluate_model()

    from ml4opf.viz import make_stats_df

    print(make_stats_df(evals))

    model.save_checkpoint(
        "basic_300bus"
    )  # creates a folder called "basic_300bus" with two files in it, trainer.ckpt and config.json

    # delete it
    import shutil; shutil.rmtree("basic_300bus")

def test_advanced_usage():
    import logging

    logging.basicConfig(level=logging.INFO)
    import torch

    from ml4opf import ACProblem

    data_path = Path(ml4opf_path[0]).parent / "tests" / "test_data" / "89_pegase"

    problem = ACProblem(data_path)

    # get train/test set:
    train_data = problem.train_data
    test_data = problem.test_data

    train_data["input/pd"].shape  # torch.Size([52863, 201])
    test_data["input/pd"].shape  # torch.Size([5000, 201])

    # if needed, convert the HDF5 data to a tree dictionary instead of a flat dictionary:
    from ml4opf.parsers import PGLearnParser

    h5_tree = PGLearnParser.make_tree(train_data)  # this tree structure should
    # exactly mimic the
    # structure of the HDF5 file.
    h5_tree["input"]["pd"].shape  # torch.Size([52863, 201])
