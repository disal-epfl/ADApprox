# MIT License
#
# Copyright (c) 2025 Nicolaj Bösel-Schmid
# Contact: nicolaj.schmid@epfl.ch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


KERNEL_NAME = "Kernel DM+V/W"
LS_NAME = "ADApprox"
FONT_SIZE = 20
ARROW_UP = "↑"
ARROW_DOWN = "↓"
ARROW_RIGHT = "→"
ARROW_LEFT = "←"

METHOD_COLORS = {
    KERNEL_NAME: "royalblue", 
    LS_NAME: "orangered",
    LS_NAME+" (0.1)": "orangered",
    LS_NAME+" (0.2)": "coral",
    LS_NAME+" (0.4)": "peachpuff",
}

plt.rcParams.update({
    'font.size': 16,          # General font size
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 14,      # X and Y label font size
    'xtick.labelsize': 14,     # X tick labels font size
    'ytick.labelsize': 14,     # Y tick labels font size
    'legend.fontsize': 16,     # Legend font size
    'figure.titlesize': 16     # Figure title font size
})

def load_data(
    file_path: str,
):
    """
    Load the CSV file into a Pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        (DataFrame): The loaded data.
    """
    return pd.read_csv(file_path, index_col=0, sep=", ", engine='python')

def add_fig_legend(
    fig: plt.Figure,
    method_names: List[str],
    ncol: int = 2,
    loc: str = "lower center",
):
    handles = []
    method_names = [KERNEL_NAME, LS_NAME+" (0.1)", LS_NAME+" (0.2)", LS_NAME+" (0.4)"]
    for m in method_names:
        handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=METHOD_COLORS[m], markersize=10))
    fig.legend(handles, method_names, loc=loc, bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False)
    return fig

def plot_boxplots(
    ax: plt.Axes,
    cathegories: list,
    method_dfs: List[pd.DataFrame],
    cathegories_names: List[str] = None,
    method_names: List[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    y_min_max: tuple = (None, None),
    put_legend_below: bool = False,
    show_legend: bool = True,
    show_yaxis_ticks: bool = True,
    spacing_between_cathegories: float = 0.06,
    spacing_between_methods: float = 0.02,
):
    """
    Plot one boxplot for each method.
    Args:
        ax (plt.Axes): Matplotlib axes object.
        cathegories (list): List of metric names.
        method_dfs (list): List of DataFrames containing the metric values for each method.
        cathegories_names (list): List of metric names to be displayed on the x-axis.
        method_names (list): List of method names to be displayed in the legend.
        title (str): Title of the plot.
        x_label (str): Label of the x-axis.
        y_label (str): Label of the y-axis.
        y_min_max (tuple): Tuple containing the minimum and maximum values of the y-axis.
        put_legend_below (bool): Whether to put the legend below the x-axis.
        show_legend (bool): Whether to show the legend.
        show_yaxis_ticks (bool): Whether to show the y-axis ticks.
        spacing_between_cathegories (float): Spacing between the boxplots of different metrics.
        spacing_between_methods (float): Spacing between the boxplots of different methods.
    Returns:
        ax (plt.Axes): Matplotlib axes object.
    """
    if method_names is None:
        method_names = [f"Method {i}" for i in range(len(method_dfs))]
    if cathegories_names is None:
        cathegories_names = cathegories

    # Data: list of methods containing lists of metric values
    data = [] 
    box_mask = np.zeros((len(method_dfs), len(cathegories)))
    for i, df in enumerate(method_dfs):

        data.append([])
        for j, metric in enumerate(cathegories):
            if metric not in df.columns:
                continue
            
            data[i].append(df[metric].values)
            box_mask[i, j] = True

    # Position of each boxplot
    positions = [] # list of methods containing lists of metric positions
    for i, df in enumerate(method_dfs):

        positions.append([])
        for j, metric in enumerate(cathegories):
            if metric not in df.columns:
                continue

            # number of boxes added along the x-axis
            boxes_added = np.sum(box_mask[:, :j]) + np.sum(box_mask[:i, j])

            pos = boxes_added / np.sum(box_mask)
            pos += j*spacing_between_cathegories
            pos += (boxes_added-j)*spacing_between_methods
            positions[i].append(pos)

    # Box plot
    for i, df in enumerate(method_dfs):
        
        color = METHOD_COLORS.get(method_names[i], "gray")  # Default to gray if method not in palette

        bp = ax.boxplot(
            x=data[i], 
            positions=positions[i],
            widths=1 / np.sum(box_mask), 
            patch_artist=True,
            boxprops=dict(facecolor=color, color="black"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )

        # set edge color
        for box in bp['boxes']:
            box.set(edgecolor=color, linewidth=1)

    # Set x-ticks and y-ticks
    x_ticks_poss = []
    for j, metric in enumerate(cathegories):
        boxes_added = np.sum(box_mask[:, :j]) + (np.sum(box_mask[:, j])-1)/2
        pos = boxes_added / np.sum(box_mask)
        pos += j*spacing_between_cathegories
        pos += (boxes_added-j)*spacing_between_methods
        x_ticks_poss.append(pos)
    ax.set_xticks(x_ticks_poss)
    ax.set_xticklabels(cathegories_names)

    if not show_yaxis_ticks:
        ax.yaxis.set_tick_params(labelleft=False)

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Create legend
    if show_legend:
        handles = []
        for m in method_names:
            handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=METHOD_COLORS[m], markersize=10))
        if put_legend_below:
            ax.legend(handles, method_names, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
        else:
            ax.legend(handles, method_names)

    # Set x-axis limits
    ax.set_xlim(
        left = - spacing_between_methods - (1 / np.sum(box_mask))/2, 
        right = positions[-1][-1] + spacing_between_methods + (1 / np.sum(box_mask))/2,
    )

    # Set y-axis limits
    if y_min_max[0] is not None:
        ax.set_ylim(bottom=y_min_max[0])
    if y_min_max[1] is not None:
        ax.set_ylim(top=y_min_max[1])

    # create an annotation for all data points that are cutoff
    ymin, ymax = ax.get_ylim()
    arrow_length = 0.1 * (ymax - ymin)
    offset_step = 0.13
    for i, df in enumerate(method_dfs):
        for j, metric in enumerate(cathegories):
            if metric not in df.columns:
                continue

            outlier_count = 0
            vals = df[metric].sort_values(ascending=False).values
            for d in vals:
                if d > ymax:
                    y_pos = ymax - outlier_count * offset_step

                    ax.annotate(
                        f'{d:.1f}',
                        xy=(positions[i][j], y_pos),
                        xytext=(positions[i][j], y_pos - arrow_length),
                        arrowprops=dict(arrowstyle='->', color='black') if outlier_count == 0 else None,
                        ha='center',
                        color='black',
                    )

                    outlier_count += 1

            outlier_count = 0
            vals = df[metric].sort_values(ascending=True).values
            for d in vals:
                if d < ymin:
                    y_pos = ymin + outlier_count * offset_step

                    ax.annotate(
                        f'{d:.1f}',
                        xy=(positions[i][j], y_pos),
                        xytext=(positions[i][j], y_pos + arrow_length),
                        arrowprops=dict(arrowstyle='->', color='black') if outlier_count == 0 else None,
                        ha='center',
                        color='black',
                    )

                    outlier_count += 1

    return ax

   

def plot_gdm(
    experiment_folder: str, 
):
    # Parameters Experiment
    file_kernel = "results_kernel.csv"
    file_ls_0_1 = "results_ls_0_1.csv"
    file_ls_0_2 = "results_ls_0_2.csv"
    file_ls_0_4 = "results_ls_0_4.csv"

    # Load data
    df_kernel = load_data(experiment_folder + file_kernel)
    df_ls_0_1 = load_data(experiment_folder + file_ls_0_1)
    df_ls_0_2 = load_data(experiment_folder + file_ls_0_2)
    df_ls_0_4 = load_data(experiment_folder + file_ls_0_4)

    # Plot 
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(10, 5.5), tight_layout=True, width_ratios=[4, 4, 4], height_ratios=[5, 0.5])
    axs = axs.flatten()

    ax = axs[0]
    plot_boxplots(
        ax=ax,
        cathegories=['time'],
        method_dfs=[df_kernel, df_ls_0_1, df_ls_0_2, df_ls_0_4],
        cathegories_names=[''],
        method_names=[KERNEL_NAME, LS_NAME+" (0.1)", LS_NAME+" (0.2)", LS_NAME+" (0.4)"],
        y_min_max=(0, None),
        y_label="Time [s] " + ARROW_LEFT,
        show_legend=False,
    )

    y_min_max = (0, None)
    if "simulation" in experiment_folder:
        y_min_max = (0, None)
    elif "realworld" in experiment_folder:
        y_min_max = (0, 35)

    ax = axs[1]
    plot_boxplots(
        ax=ax,
        cathegories=['mae'],
        method_dfs=[df_kernel, df_ls_0_1, df_ls_0_2, df_ls_0_4],
        cathegories_names=[''],
        method_names=[KERNEL_NAME, LS_NAME+" (0.1)", LS_NAME+" (0.2)", LS_NAME+" (0.4)"],
        y_min_max=y_min_max,
        y_label="Mean Absolute Error " + ARROW_LEFT,
        show_legend=False,
    )

    y_min_max = (None, 1.0)
    if "simulation" in experiment_folder:
        y_min_max = (0.85, 1)
    elif "realworld" in experiment_folder:
        y_min_max = (0.65, 1)

    ax = axs[2]
    plot_boxplots(
        ax=ax,
        cathegories=['shape'],
        method_dfs=[df_kernel, df_ls_0_1, df_ls_0_2, df_ls_0_4],
        cathegories_names=[''],
        method_names=[KERNEL_NAME, LS_NAME+" (0.1)", LS_NAME+" (0.2)", LS_NAME+" (0.4)"],
        y_min_max=y_min_max,
        y_label="Shape [%] " + ARROW_RIGHT,
        show_legend=False,
    )

    fig = add_fig_legend(
        fig=fig,
        method_names=[KERNEL_NAME, LS_NAME+" (0.1)", LS_NAME+" (0.2)", LS_NAME+" (0.4)"],
    )
    
    axs[3].remove()
    axs[4].remove()
    axs[5].remove()

    plt.savefig(experiment_folder + "boxplots.png", dpi=300)
    plt.show()

def plot_simu_wind(
    experiment_folder: str,
):
    # Parameters Experiment
    file_kernel_0_3 = "wind_0_3/rect_1/results_kernel.csv"
    file_kernel_0_7 = "wind_0_7/rect_1/results_kernel.csv"
    file_kernel_1_3 = "wind_1_3/rect_1/results_kernel.csv"
    file_ls_0_3 = "wind_0_3/rect_1/results_ls_0_1.csv"
    file_ls_0_7 = "wind_0_7/rect_1/results_ls_0_1.csv"
    file_ls_1_3 = "wind_1_3/rect_1/results_ls_0_1.csv"

    # Load data
    df_kernel_0_3 = load_data(experiment_folder + file_kernel_0_3)
    df_kernel_0_7 = load_data(experiment_folder + file_kernel_0_7)
    df_kernel_1_3 = load_data(experiment_folder + file_kernel_1_3)
    df_ls_0_3 = load_data(experiment_folder + file_ls_0_3)
    df_ls_0_7 = load_data(experiment_folder + file_ls_0_7)
    df_ls_1_3 = load_data(experiment_folder + file_ls_1_3)

    cathegories = ["0_3", "0_7", "1_3"]
    cathegory_names = ["0.3", "0.7", "1.3"]

    df_kernel_mae = pd.DataFrame({
        "0_3": df_kernel_0_3["mae"],
        "0_7": df_kernel_0_7["mae"],
        "1_3": df_kernel_1_3["mae"],
    })
    df_ls_mae = pd.DataFrame({
        "0_3": df_ls_0_3["mae"],
        "0_7": df_ls_0_7["mae"],
        "1_3": df_ls_1_3["mae"],
    })
    df_kernel_shape = pd.DataFrame({
        "0_3": df_kernel_0_3["shape"],
        "0_7": df_kernel_0_7["shape"],
        "1_3": df_kernel_1_3["shape"],
    })
    df_ls_shape = pd.DataFrame({
        "0_3": df_ls_0_3["shape"],
        "0_7": df_ls_0_7["shape"],
        "1_3": df_ls_1_3["shape"],
    })

    # print median
    print("Kernel MAE")
    print(df_kernel_mae.median())
    print("LS MAE")
    print(df_ls_mae.median())
    print("Kernel Shape")
    print(df_kernel_shape.median())
    print("LS Shape")
    print(df_ls_shape.median())

    # Plot 
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), tight_layout=True, width_ratios=[3, 3])
    axs = axs.flatten()

    ax = axs[0]
    plot_boxplots(
        ax=ax,
        cathegories=cathegories,
        cathegories_names=cathegory_names,
        method_dfs=[df_kernel_mae, df_ls_mae],
        method_names=[KERNEL_NAME, LS_NAME],
        x_label="Wind Speed [m/s]",
        y_label="Mean Absolute Error " + ARROW_LEFT,
        y_min_max=(0, None),
        show_legend=False,
    )

    ax = axs[1]
    plot_boxplots(
        ax=ax,
        cathegories=cathegories,
        cathegories_names=cathegory_names,
        method_dfs=[df_kernel_shape, df_ls_shape],
        method_names=[KERNEL_NAME, LS_NAME],
        x_label="Wind Speed [m/s]",
        y_label="Shape [%] " + ARROW_RIGHT,
        y_min_max=(None, 1),
    )

    plt.savefig(experiment_folder + "wind_boxplots.png", dpi=300)
    plt.show()

def plot_simu_spacing(
    experiment_folder: str,
):
    # Parameters Experiment    
    file_kernel_1 = "wind_0_7/rect_1/results_kernel.csv"
    file_kernel_1_5 = "wind_0_7/rect_1_5/results_kernel.csv"
    file_kernel_2 = "wind_0_7/rect_2/results_kernel.csv"
    file_ls_1 = "wind_0_7/rect_1/results_ls_0_1.csv"
    file_ls_1_5 = "wind_0_7/rect_1_5/results_ls_0_1.csv"
    file_ls_2 = "wind_0_7/rect_2/results_ls_0_1.csv"

    # Load data
    df_kernel_0_3 = load_data(experiment_folder + file_kernel_1)
    df_kernel_0_7 = load_data(experiment_folder + file_kernel_1_5)
    df_kernel_1_3 = load_data(experiment_folder + file_kernel_2)
    df_ls_0_3 = load_data(experiment_folder + file_ls_1)
    df_ls_0_7 = load_data(experiment_folder + file_ls_1_5)
    df_ls_1_3 = load_data(experiment_folder + file_ls_2)

    cathegories = ["1", "1_5", "2"]
    cathegory_names = ["1.0", "1.5", "2.0"]

    df_kernel_mae = pd.DataFrame({
        "1": df_kernel_0_3["mae"],
        "1_5": df_kernel_0_7["mae"],
        "2": df_kernel_1_3["mae"],
    })
    df_ls_mae = pd.DataFrame({
        "1": df_ls_0_3["mae"],
        "1_5": df_ls_0_7["mae"],
        "2": df_ls_1_3["mae"],
    })
    df_kernel_shape = pd.DataFrame({
        "1": df_kernel_0_3["shape"],
        "1_5": df_kernel_0_7["shape"],
        "2": df_kernel_1_3["shape"],
    })
    df_ls_shape = pd.DataFrame({
        "1": df_ls_0_3["shape"],
        "1_5": df_ls_0_7["shape"],
        "2": df_ls_1_3["shape"],
    })

    # print median
    print("Kernel MAE")
    print(df_kernel_mae.median())
    print("LS MAE")
    print(df_ls_mae.median())
    print("Kernel Shape")
    print(df_kernel_shape.median())
    print("LS Shape")
    print(df_ls_shape.median())

    # Plot 
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), tight_layout=True, width_ratios=[3, 3])
    axs = axs.flatten()

    ax = axs[0]
    plot_boxplots(
        ax=ax,
        cathegories=cathegories,
        cathegories_names=cathegory_names,
        method_dfs=[df_kernel_mae, df_ls_mae],
        method_names=[KERNEL_NAME, LS_NAME],
        x_label="Spacing [m]",
        y_label="Mean Absolute Error " + ARROW_LEFT,
        y_min_max=(0, None),
        show_legend=False,
    )

    ax = axs[1]
    plot_boxplots(
        ax=ax,
        cathegories=cathegories,
        cathegories_names=cathegory_names,
        method_dfs=[df_kernel_shape, df_ls_shape],
        method_names=[KERNEL_NAME, LS_NAME],
        x_label="Spacing [m]",
        y_label="Shape [%] " + ARROW_RIGHT,
        y_min_max=(None, 1),
    )

    plt.savefig(experiment_folder + "spacing_boxplots.png", dpi=300)
    plt.show()

def plot_gsl(
    experiment_folder:str,
):
    # Parameters Experiment
    file_kernel = "results_kernel.csv"
    file_ls_0_1 = "results_ls_0_1.csv"
    file_ls_0_2 = "results_ls_0_2.csv"
    file_ls_0_4 = "results_ls_0_4.csv"

    # Load data
    df_kernel = load_data(experiment_folder + file_kernel)
    df_ls_0_1 = load_data(experiment_folder + file_ls_0_1)
    df_ls_0_2 = load_data(experiment_folder + file_ls_0_2)
    df_ls_0_4 = load_data(experiment_folder + file_ls_0_4)

    # print median
    print("Kernel")
    print(df_kernel.median())
    print("LS 0.1")
    print(df_ls_0_1.median())

    # Plot 
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 5.5), tight_layout=True, height_ratios=[5, 0.5])
    axs = np.array(axs).flatten()

    y_min_max = (None, 1.0)
    if "simulation" in experiment_folder:
        y_min_max = (0, 1.75)
    elif "realworld" in experiment_folder:
        y_min_max = (0, 2.75)

    ax = axs[0]
    plot_boxplots(
        ax=ax,
        cathegories=['gsl_max', 'gsl_var', 'gsl_wind'],
        method_dfs=[df_kernel, df_ls_0_1, df_ls_0_2, df_ls_0_4],
        cathegories_names=['Concentration', 'Variance', 'Release Rate Parameter'],
        method_names=[KERNEL_NAME, LS_NAME+" (0.1)", LS_NAME+" (0.2)", LS_NAME+" (0.4)"],
        y_min_max=y_min_max,
        y_label="Localization Error [m] " + ARROW_LEFT,
        show_legend=False,
    )

    fig = add_fig_legend(
        fig=fig,
        method_names=[KERNEL_NAME, LS_NAME+" (0.1)", LS_NAME+" (0.2)", LS_NAME+" (0.4)"],
    )
    
    axs[1].remove()

    plt.savefig(experiment_folder + "boxplots.png", dpi=300)
    plt.show()