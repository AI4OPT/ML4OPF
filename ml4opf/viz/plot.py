import numpy as np

from logging import info, warning as warn
from torch import Tensor
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame


def plot_wrt_total_load(
    pd: Tensor,
    metrics: list[Tensor],
    labels: list[str],
    title: str,
    ylabel: str,
    logy: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    subplots_kwargs: dict = None, #{"figsize": (8, 5), "dpi": 300},
    plot_kwargs: dict = None, #{"ls": "", "marker": ".", "ms": 0.75},
    legend: bool = True,
    show: bool = True,
):
    """Plot metrics vs total active load

    Args:
        pd (Tensor): Active load tensor of shape [batch, n_loads]. pd.sum(dim=1) becomes the x-axis.
        metrics (list[Tensor]): List of metrics tensors of shape [batch, 1] <- one number per sample
        labels (list[str], optional): List of labels for each metric
        title (str, optional): Plot title
        ylabel (str, optional): Y-axis label
        logy (bool, optional): Whether to use a log scale for the y-axis. Default: False
        xlim (tuple[float, float], optional): X-axis limits. Default: None
        ylim (tuple[float, float], optional): Y-axis limits. Default: None
        subplots_kwargs (dict, optional): Keyword arguments for plt.subplots(). Default: {'figsize':(8,5), 'dpi':300}
        plot_kwargs (dict, optional): Keyword arguments for ax.plot() (same for every metric). Default: {'ls':'', 'marker': '.', 'ms':0.75}
        show (bool, optional): Whether to show the plot. Default: True
    """
    try:
        import matplotlib.pyplot as plt
        from pandas import DataFrame
    except ImportError as e:
        raise ImportError(
            "plot_wrt_total_load requires matplotlib and pandas. Install them using:\n\tpip install matplotlib pandas"
        ) from e
    
    if subplots_kwargs is None:
        subplots_kwargs = {"figsize": (8, 5), "dpi": 300}
    if plot_kwargs is None:
        plot_kwargs = {"ls": "", "marker": ".", "ms": 0.75}

    assert set([len(m) for m in metrics]) == {len(pd)}, "Metrics must have the same length as pd"
    assert len(metrics) == len(labels), "Metrics and labels must have the same length"

    info("Setting matplotlib style to 'tableau-colorblind10'")
    plt.style.use("tableau-colorblind10")

    tl = pd.sum(dim=1)
    tl_sorted, tl_idx = tl.sort()

    fig, ax = plt.subplots(**subplots_kwargs)

    for metric, label in zip(metrics, labels):
        ax.plot(tl_sorted, metric[tl_idx], label=label, **plot_kwargs)

    ax.set_xlabel("Total active load (p.u.)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)

    if logy:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if legend:
        leg = ax.legend()
        for lh in leg.legendHandles:
            lh.set_markersize(10)
    if show:
        plt.show()

    return fig, ax


def plot_pca(
    tensor: Tensor,
    label: str,
    color_by: Optional[Tensor] = None,
    logy: bool = False,
    logx: bool = False,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    subplots_kwargs: dict = None, #{"figsize": (8, 5), "dpi": 300},
    scatter_kwargs: dict = None, #{"marker": "."},
    show: bool = True,
):
    """Plot PCA of a 2D tensor.

    Args:
        tensor (Tensor): Tensor of shape [batch, n_features]
        label (str): Label for the plot
        color_by (Tensor, optional): Tensor of shape [batch, 1] to color the points by. Default: None
        logy (bool, optional): Whether to use a log scale for the y-axis. Default: False
        logx (bool, optional): Whether to use a log scale for the x-axis. Default: False
        xlim (tuple[float, float], optional): X-axis limits. Default: None
        ylim (tuple[float, float], optional): Y-axis limits. Default: None
        subplots_kwargs (dict, optional): Keyword arguments for plt.subplots(). Default: {'figsize':(8,5), 'dpi':300}
        scatter_kwargs (dict, optional): Keyword arguments for ax.scatter(). Default: {'marker': '.'}
        show (bool, optional): Whether to show the plot. Default: True
    """

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "plot_wrt_total_load requires matplotlib and pandas. Install them using:\n\tpip install matplotlib pandas"
        ) from e

    if color_by is not None:
        assert len(color_by) == len(tensor), "color_by must have the same length as tensor"
        assert color_by.ndim == 1, "color_by must be a 1D tensor"

    if subplots_kwargs is None:
        subplots_kwargs = {"figsize": (8, 5), "dpi": 300}
    if scatter_kwargs is None:
        scatter_kwargs = {"marker": "."}

    assert tensor.ndim == 2, "tensor must be a 2D tensor"

    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("plot_pca requires scikit-learn. Install it using:\n\tpip install scikit-learn") from e

    pca = PCA(n_components=2)
    tensor_pca = pca.fit_transform(tensor)

    fig, ax = plt.subplots(**subplots_kwargs)
    if color_by is not None:
        ax.scatter(tensor_pca[:, 0], tensor_pca[:, 1], c=color_by, **scatter_kwargs)
        fig.colorbar(ax.collections[0], ax=ax, label="Total Load")
    else:
        ax.scatter(tensor_pca[:, 0], tensor_pca[:, 1], **scatter_kwargs)

    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance expl.)", fontsize=14)
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance expl.)", fontsize=14)

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    title = f"PCA of {label}"

    ax.set_title(title, fontsize=16)

    if show:
        plt.show()

    return fig, ax


def interp_curves(
    convergence_curves: list["DataFrame"],
    column: str,
    max_time: Optional[int] = None,
    n_time_steps: int = 2000,
    kind: str = "previous",
    fill_value: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate convergence curves to have the same start/end/number of time steps.

    Args:
        convergence_curves (list[DataFrame]): List of DataFrames with columns 'time' and `column`
        column (str): Column to interpolate. All convergence_curves must have this column.
        max_time (int, optional): Maximum time to interpolate to. If None, use the maximum time of all convergence curves. Default: None
        n_time_steps (int, optional): Number of time steps to interpolate to. Default: 2000
        kind (str, optional): Interpolation kind. Default: 'previous'
        fill_value (str, optional): Fill value for time steps before the first point.
                If None, return None for any time steps before the first point and the last point for any time steps after the last point. Default: None

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Interpolated times, means, and standard deviations
    """

    [warn("interp_curves has not been tested!") for _ in range(3)]

    assert "time" in convergence_curves[0].columns, "convergence_curves must have a 'time' column"
    assert all(column in c.columns for c in convergence_curves), f"all convergence_curves must have a '{column}' column"

    try:
        from scipy.interpolate import interp1d
    except ImportError as e:
        raise ImportError("interp_curves requires scipy. Install it using:\n\tpip install scipy") from e
    # Interpolate each trial's convergence curve to have n_time_steps.
    # Return None for any time steps below the first point, and return the latest point before the time step queried.

    if max_time is None:
        max_time = max(c["time"].values[-1] for c in convergence_curves)

    times = np.linspace(0, max_time, n_time_steps)

    interp_curves = []
    for curve in convergence_curves:
        if fill_value is None:
            fill_value = (None, curve[column].values[-1])
        interp_curve = interp1d(curve["time"].values, curve[column].values, kind=kind, fill_value=fill_value)
        new_ts = interp_curve(times)
        interp_curves.append(new_ts)

    interp_curves = np.array(interp_curves)

    # Take the mean and std across trials
    mean = interp_curves.mean(axis=0)
    std = interp_curves.std(axis=0)

    return times, mean, std
