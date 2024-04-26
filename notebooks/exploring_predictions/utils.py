import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SeabornFig2Grid:
    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(
            self.sg, sns.axisgrid.PairGrid
        ):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """Move PairGrid or Facetgrid"""
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """Move Jointgrid"""
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r + 1, r + 1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


# Make a function to plot the scatter plot for each model
def plot_scatter(
    model_df: pd.DataFrame,
    model_name: str,
    ccs_label: str,
    pred_ccs_label: str,
    color_by: str,
    adduct_label: str = "adduct",
):
    if color_by == "dimer":
        model_df["color"] = model_df["dimer"]
        palette = {"Monomer": "#dfa693", "Dimer": "#a5d3eb"}

    elif color_by == "mol_type":
        model_df["color"] = model_df["mol_type"]
        palette = {
            "small molecule": "#dfa693",
            "lipid": "#a5d3eb",
            "peptide": "#5f5f5f",
            "carbohydrate": "#f2b5d4",
        }

    elif color_by == "adduct":
        model_df["color"] = model_df[adduct_label].copy()
        palette = {
            "[M+H]+": "#1f77b4",
            "[2M+H]+": "#ff7f0e",
            "[M+Na]+": "#d62728",
            "[2M+Na]+": "#2ca02c",
            "[M-H]-": "#8c564b",
            "[2M-H]-": "#9467bd",
            "[M+K]+": "#e377c2",
            "[M+H-H2O]+": "#7f7f7f",
            "[M+NH4]+": "#bcbd22",
        }

    else:
        raise ValueError("color_by should be either 'dimer' or 'mol_type'")

    g = sns.jointplot(
        data=model_df,
        x=model_df[ccs_label],
        y=model_df[pred_ccs_label],
        hue=model_df["color"],
        palette=palette,
        # add transparency to the points
        alpha=0.4,
    )

    # set title
    g.fig.suptitle(f"Scatter plot of {model_name} predictions")

    # Add a line to show the perfect correlation
    g.ax_joint.plot(
        [model_df[ccs_label].min(), model_df[ccs_label].max()],
        [model_df[ccs_label].min(), model_df[ccs_label].max()],
        "k--",
        lw=2,
    )

    mae = mean_absolute_error(model_df[ccs_label], model_df[pred_ccs_label])
    rmse = mean_squared_error(model_df[ccs_label], model_df[pred_ccs_label])
    r2 = linregress(model_df[ccs_label], model_df[pred_ccs_label]).rvalue ** 2

    g.ax_joint.text(
        0.5,
        0.1,
        f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=g.ax_joint.transAxes,
    )

    g.ax_joint.set_xlabel("Experimental CCS")
    g.ax_joint.set_ylabel(f"Predicted CCS by {model_name}")

    # set legend title
    g.ax_joint.get_legend().set_title(f"{color_by} n={model_df.shape[0]}")

    return g
