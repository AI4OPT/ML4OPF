import torch
from ml4opf.viz.table import make_stats_df


def test_table():

    data = {
        "loss": torch.as_tensor([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]).view(-1, 1),
        "val_loss": torch.as_tensor([10.0, 9, 8, 7, 6, 5, 4, 3, 2, 10]).view(-1, 1),
    }

    df = make_stats_df(data)
    assert df.shape == (4, 2)

    assert df.columns.tolist() == ["loss", "val_loss"]
    assert df.index.tolist() == ["mean", "max", "min", "std"]

    df.to_latex()

    df = make_stats_df(data, round_to=None)
    assert df.shape == (4, 2)

    assert df.columns.tolist() == ["loss", "val_loss"]
    assert df.index.tolist() == ["mean", "max", "min", "std"]
