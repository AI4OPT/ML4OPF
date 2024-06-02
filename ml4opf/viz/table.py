import torch

from pandas import DataFrame
from torch import Tensor

from typing import Optional


def make_stats_df(data: dict[str, Tensor], round_to: Optional[int] = 5):
    """Make a pandas DataFrame with mean/max/min/std stats.

    Args:
        data (dict[str, Tensor]): Each tensor should be of shape [batch, 1]
        round_to (Optional[int]): Number of decimals to round each Tensor to (default: 5)

    Returns:
        DataFrame: mean/max/min/std along batch dimension.

    When used with `DataFrame.to_latex()`, the output can be copied directly into a LaTeX table.
    """

    stats, names = [], []
    for k, v in data.items():

        if round_to is not None:
            v_ = torch.round(v, decimals=round_to)
        else:
            v_ = v

        stats.append(
            {
                "mean": v_.mean().item(),
                "max": v_.max().item(),
                "min": v_.min().item(),
                "std": v_.std().item(),
            }
        )

        names.append(k)

    df = DataFrame(stats).T
    df.columns = names

    return df
