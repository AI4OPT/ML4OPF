import matplotlib.pyplot as plt
import torch

from torch import Tensor
from typing import Optional

from ml4opf import ACPProblem, ACPModel
from ml4opf.viz.plot import plot_wrt_total_load


class ModelReport:
    """Create an evaluation report for an ACPModel."""

    def __init__(self, problem: ACPProblem, model: ACPModel):
        self.problem = problem
        self.model = model

    def compute_predictions(self):
        """Compute predictions for the model."""
        pd, qd = self.problem.test_data["input/pd"], self.problem.test_data["input/qd"]
        self.predictions = self.model.predict(pd, qd)

    def pg_qg_fig(self, generator_idx: int, sort_by: Optional[Tensor] = None):
        if sort_by is None:
            sort_by = self.problem.test_data["input/pd"].sum(dim=1)

        assert generator_idx >= 0 and generator_idx < self.problem.violation.n_gen, "Invalid generator id"

        if not hasattr(self, "predictions"):
            self.compute_predictions()

        sort_idx = sort_by.argsort()

        pg = self.predictions["pg"]
        pg_gt = self.problem.test_data["ACPPowerModel/primal/pg"]
        pmin = self.problem.violation.pmin[generator_idx]
        pmax = self.problem.violation.pmax[generator_idx]
        pg_sorted = pg[sort_idx, generator_idx]
        pg_gt_sorted = pg_gt[sort_idx, generator_idx]

        qg = self.predictions["qg"]
        qg_gt = self.problem.test_data["ACPPowerModel/primal/qg"]
        qmin = self.problem.violation.qmin[generator_idx]
        qmax = self.problem.violation.qmax[generator_idx]
        qg_sorted = qg[sort_idx, generator_idx]
        qg_gt_sorted = qg_gt[sort_idx, generator_idx]

        fig, ax = plt.subplots(2, 1, figsize=(8, 8), dpi=300)
        ax[0].plot(sort_by[sort_idx], pg_sorted, color="red", ls="", marker=".", ms=0.75)
        ax[0].plot(sort_by[sort_idx], pg_gt_sorted, color="black", ls="", marker=".", ms=0.75)
        ax[0].axhline(pmin, color="k", ls="--", lw=0.5, label="Bounds")
        ax[0].axhline(pmax, color="k", ls="--", lw=0.5)
        ax[0].set_ylabel("Active generation (p.u.)", fontsize=14)

        ax[1].plot(sort_by[sort_idx], qg_sorted, color="red", ls="", marker=".", ms=0.75)
        ax[1].plot(sort_by[sort_idx], qg_gt_sorted, color="black", ls="", marker=".", ms=0.75)
        ax[1].axhline(qmin, color="k", ls="--", lw=0.5, label="Bounds")
        ax[1].axhline(qmax, color="k", ls="--", lw=0.5)
        ax[1].set_ylabel("Reactive generation (p.u.)", fontsize=14)
        ax[1].set_xlabel("Total active load (p.u.)", fontsize=14)
        ax[0].set_title(f"Generator {generator_idx}", fontsize=16)

        # manual legend in between subplots
        import matplotlib.lines as mlines

        red_patch = mlines.Line2D([], [], color="red", marker=".", linestyle="None", markersize=20, label="Predicted")
        black_patch = mlines.Line2D(
            [], [], color="black", marker=".", linestyle="None", markersize=20, label="Ground truth"
        )
        plt.subplots_adjust(hspace=0.25)
        ax[1].legend(
            handles=[red_patch, black_patch], loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=14
        )

        return fig

    def vm_va_fig(self, bus_idx: int, sort_by: Optional[Tensor] = None):
        if sort_by is None:
            sort_by = self.problem.test_data["input/pd"].sum(dim=1)

        assert bus_idx >= 0 and bus_idx < self.problem.violation.n_bus, "Invalid bus id"

        if not hasattr(self, "predictions"):
            self.compute_predictions()

        sort_idx = sort_by.argsort()

        vm = self.predictions["vm"]
        vm_gt = self.problem.test_data["ACPPowerModel/primal/vm"]
        vmin = self.problem.violation.vmin[bus_idx]
        vmax = self.problem.violation.vmax[bus_idx]
        vm_sorted = vm[sort_idx, bus_idx]
        vm_gt_sorted = vm_gt[sort_idx, bus_idx]

        va = self.predictions["va"]
        va_gt = self.problem.test_data["ACPPowerModel/primal/va"]
        va_sorted = va[sort_idx, bus_idx]
        va_gt_sorted = va_gt[sort_idx, bus_idx]

        fig, ax = plt.subplots(2, 1, figsize=(8, 8), dpi=300)
        ax[0].plot(sort_by[sort_idx], vm_sorted, color="red", ls="", marker=".", ms=0.75)
        ax[0].plot(sort_by[sort_idx], vm_gt_sorted, color="black", ls="", marker=".", ms=0.75)
        ax[0].axhline(vmin, color="k", ls="--", lw=0.5, label="Bounds")
        ax[0].axhline(vmax, color="k", ls="--", lw=0.5)
        ax[0].set_ylabel("Voltage magnitude (p.u.)", fontsize=14)

        ax[1].plot(sort_by[sort_idx], va_sorted, color="red", ls="", marker=".", ms=0.75)
        ax[1].plot(sort_by[sort_idx], va_gt_sorted, color="black", ls="", marker=".", ms=0.75)
        ax[1].set_ylabel("Voltage angle (p.u.)", fontsize=14)
        ax[1].set_xlabel("Total active load (p.u.)", fontsize=14)
        ax[0].set_title(f"Bus {bus_idx}", fontsize=16)

        # manual legend in between subplots
        import matplotlib.lines as mlines

        red_patch = mlines.Line2D([], [], color="red", marker=".", linestyle="None", markersize=20, label="Predicted")
        black_patch = mlines.Line2D(
            [], [], color="black", marker=".", linestyle="None", markersize=20, label="Ground truth"
        )
        plt.subplots_adjust(hspace=0.25)
        ax[1].legend(
            handles=[red_patch, black_patch], loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=14
        )

        return fig

    def loss_wrt_total_load_fig(self, which: str = "pg"):
        if not hasattr(self, "predictions"):
            self.compute_predictions()

        pd = self.problem.test_data["input/pd"]

        loss = torch.abs(
            self.predictions[which] - self.problem.test_data[f"{self.solution_prefix}/primal/" + which]
        ).mean(dim=1)

        title = f"Absolute difference between ground truth and predicted {which}"
        ylabel = f"L1 loss on test set"

        return plot_wrt_total_load(pd, [loss], [f"{which} loss"], title, ylabel, logy=True)
