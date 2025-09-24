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

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from args.args import Args
from mapping.kernelDMV import KernelDMV
from mapping.mean_meas import MeanMeas
from mapping.least_square_variance import LeastSquareVariance
from evaluation.metrics.metrics import Metrics
from mapping.map import Map

def maps_min_max(
    maps:list,
    features:int=None,
) -> tuple:
    """
    Get the minimum and maximum values of a list of maps.
    Args:
        maps (list): List of maps.
        feature (int): Feature index. If None, get the minimum and maximum values of all features.
    Returns:
        min_max (tuple): Minimum and maximum values.
    """
    min_max = [np.inf, -np.inf]
    for map_, feature_ in zip(maps, features):
        
        min_ = np.min(map_.get_vals_min(
            feature=feature_,
        ))
        if min_ < min_max[0]:
            min_max[0] = min_

        max_ = np.max(map_.get_vals_max(
            feature=feature_,
        ))
        if max_ > min_max[1]:
            min_max[1] = max_
    
    return tuple(min_max)

def plot(
    args:Args,
    filter_gt: MeanMeas,
    filter_gt_var: MeanMeas,
    filter_mm: MeanMeas,
    filter_km: KernelDMV,
    simu: LeastSquareVariance,
    map_ls: Map,
    map_ls_var: Map,
    metrics: Metrics,
    cf_poss: np.ndarray,
    exp: str,
    print_results:bool = False,
):
    if print_results:
        print("\nPlotting")

    plt.rcParams.update({
        'font.size': 16,          # General font size
        'axes.titlesize': 16,      # Title font size
        'axes.labelsize': 14,      # X and Y label font size
        'xtick.labelsize': 14,     # X tick labels font size
        'ytick.labelsize': 14,     # Y tick labels font size
        'legend.fontsize': 16,     # Legend font size
        'figure.titlesize': 16     # Figure title font size
    })

    # Plotting
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(17.5, 14), tight_layout=True)
    axes = axes.flatten()
    cmap_min_max = maps_min_max(
        maps=[filter_gt.map_c, filter_mm.map_c, filter_km.map_c, map_ls],
        features=[0, 0, 0, 0],
    )
    
    ax = axes[0]
    ax = filter_gt.map_c.plot2D(
        ax=ax,
        cmap_name='cividis',
        cmap_min_max=cmap_min_max,
        title="a) Ground Truth",
        feature=0,
        x_label=None,
    )
    ax = filter_gt.map_c.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )
    ax = metrics.plot_segmentation(
        ax=ax,
        map=filter_gt.map_c,
    )

    ax = axes[1]
    ax = filter_gt_var.map_c.plot2D(
        ax=ax,
        cmap_name='gray',
        title="b) Ground Truth Variance",
        feature=0,
        x_label=None,
        y_lable=None,
    )
    ax = filter_gt_var.map_c.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )

    ax = axes[2]
    ax = filter_km.map_c.plot2D(
        ax=ax,
        cmap_name='cividis',
        cmap_min_max=cmap_min_max,
        title=f"c) Kernel DM+V/W",
        feature=0,
        x_label=None,
    )
    ax = filter_km.map_c.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )
    ax = metrics.plot_segmentation(
        ax=ax,
        map=filter_km.map_c,
    )
    ax = metrics.plot_max(
        ax=ax,
        map=filter_km.map_c,
        map_mask=filter_km.map_c,
    )

    ax = axes[3]
    ax = filter_km.map_var.plot2D(
        ax=ax,
        cmap_name='gray',
        title=f"d) Kernel DM+V/W Variance",
        feature=0,
        x_label=None,
        y_lable=None,
    )
    ax = filter_km.map_var.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )
    ax = metrics.plot_max(
        ax=ax,
        map=filter_km.map_var,
        map_mask=filter_km.map_c,
    )

    ax = axes[4]
    ax = map_ls.plot2D(
        ax=ax,
        cmap_name='cividis',
        cmap_min_max=cmap_min_max,
        title=f"e) ADApprox",
        feature=0,
        x_label=None,
    )
    ax = map_ls.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )
    ax = metrics.plot_segmentation(
        ax=ax,
        map=map_ls,
    )
    ax = metrics.plot_max(
        ax=ax,
        map=map_ls,
        map_mask=map_ls,
    )

    ax = axes[5]
    ax = map_ls_var.plot2D(
        ax=ax,
        cmap_name='gray',
        title=f"f) ADApprox Variance",
        feature=0,
        x_label=None,
        y_lable=None,
    )
    ax = map_ls_var.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )
    ax = metrics.plot_max(
        ax=ax,
        map=map_ls_var,
        map_mask=map_ls,
    )

    ax = axes[6]
    ax = filter_mm.map_c.plot2D(
        ax=ax,
        cmap_name='cividis',
        cmap_min_max=cmap_min_max,
        title=f"g) Measurements",
        feature=0,
        x_label=None,
    )
    ax = filter_mm.map_c.plot2D_traj(
        traj=cf_poss,
        ax=ax,
    )
    ax = filter_mm.map_c.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )

    ax = axes[7]
    abs_max = simu.map_params.get_vals_abs_max(feature=1)
    ax = simu.map_params.plot2D(
        ax=ax,
        cmap_name='coolwarm',
        cmap_min_max=(-abs_max, abs_max),
        title=f"h) Cross-Wind Parameter",
        feature=1,
        x_label=None,
        y_lable=None,
    )
    ax = simu.map_params.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )

    ax = axes[8]
    abs_max = simu.map_params.get_vals_abs_max(feature=0)
    ax = simu.map_params.plot2D(
        ax=ax,
        cmap_name='coolwarm',
        cmap_min_max=(-abs_max, abs_max),
        title=f"i) Bias Parameter",
        feature=0,
    )
    ax = simu.map_params.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )

    ax = axes[9]
    abs_max = simu.map_params.get_vals_abs_max(feature=2)
    ax = simu.map_params.plot2D(
        ax=ax,
        cmap_name='coolwarm',
        cmap_min_max=(-abs_max, abs_max),
        title=f"j) Release Rate Parameter",
        feature=2,
        y_lable=None,
    )
    ax = simu.map_params.plot2D_poss(
        ax=ax,
        poss = args(["eval", "source_pos"]),
        color='red',
        marker='*',
        markersize=120,
        label='Source',
    )
    ax = metrics.plot_max(
        ax=ax,
        map=simu.map_params,
        feature=2,
        map_mask=map_ls,
    )

    plt.savefig(args(["data", "data_folder"]) + args(["data", "folder_cf"]) + exp + 'maps.png', dpi=300)
    plt.show()

def plot3D(
    args:Args,
    filter_gt: MeanMeas,
    filter_mm: MeanMeas,
    filter_km: KernelDMV,
    simu: LeastSquareVariance,
    map_ls: Map,
    map_ls_var: Map,
    metrics: Metrics,
    cf_poss: np.ndarray,
    exp: str,
    print_results:bool = False,
):
    if print_results:
        print("\nPlotting")

    heights = [0.55, 0.65, 0.75, 0.85, 0.95]

    # Plotting
    fig, axes = plt.subplots(nrows=8, ncols=len(heights), figsize=(40, 20), layout='compressed')
    axes = axes.flatten()
    cmap_min_max = maps_min_max(
        maps=[filter_gt.map_c, filter_mm.map_c, filter_km.map_c, map_ls],
        features=[0, 0, 0, 0],
    )

    
    for i, h in enumerate(heights):
    
        ax = axes[0*len(heights) + i]
        ax = filter_gt.map_c.plot2D(
            ax=ax,
            cmap_name='cividis',
            cmap_min_max=cmap_min_max,
            title="Ground Truth",
            feature=0,
            height=h,
        )
        ax = filter_gt.map_c.plot2D_poss(
            ax=ax,
            poss = args(["eval", "source_pos"]),
            color='red',
            marker='*',
            markersize=120,
            label='Source',
        )
        # ax = metrics.plot_segmentation(
        #     ax=ax,
        #     map=filter_gt.map_c,
        # )

        ax = axes[1*len(heights) + i]
        ax = filter_mm.map_c.plot2D(
            ax=ax,
            cmap_name='cividis',
            cmap_min_max=cmap_min_max,
            title=f"Measurements",
            feature=0,
            height=h,
        )
        ax = filter_mm.map_c.plot2D_traj(
            traj=cf_poss,
            ax=ax,
        )
        ax = filter_mm.map_c.plot2D_poss(
            ax=ax,
            poss = args(["eval", "source_pos"]),
            color='red',
            marker='*',
            markersize=120,
            label='Source',
        )

        ax = axes[2*len(heights) + i]
        ax = filter_km.map_c.plot2D(
            ax=ax,
            cmap_name='cividis',
            cmap_min_max=cmap_min_max,
            title=f"Kernel DM+V/W",
            feature=0,
            height=h,
        )
        ax = filter_km.map_c.plot2D_poss(
            ax=ax,
            poss = args(["eval", "source_pos"]),
            color='red',
            marker='*',
            markersize=120,
            label='Source',
        )
        # ax = metrics.plot_segmentation(
        #     ax=ax,
        #     map=filter_km.map_c,
        # )
        ax = metrics.plot_max(
            ax=ax,
            map=filter_km.map_c,
            map_mask=filter_km.map_c,
        )

        # ax = axes[4]
        # ax = filter_km.map_var.plot2D(
        #     ax=ax,
        #     cmap_name='gray',
        #     title=f"Kernel DM+V/W Variance",
        #     feature=0,
        # )
        # ax = filter_km.map_var.plot2D_poss(
        #     ax=ax,
        #     poss = args(["eval", "source_pos"]),
        #     color='red',
        #     marker='*',
        #     markersize=120,
        #     label='Source',
        # )
        # ax = metrics.plot_max(
        #     ax=ax,
        #     map=filter_km.map_var,
        #     map_mask=filter_km.map_c,
        # )

        ax = axes[3*len(heights) + i]
        ax = map_ls.plot2D(
            ax=ax,
            cmap_name='cividis',
            cmap_min_max=cmap_min_max,
            title=f"Least Square",
            feature=0,
            height=h,
        )
        ax = map_ls.plot2D_poss(
            ax=ax,
            poss = args(["eval", "source_pos"]),
            color='red',
            marker='*',
            markersize=120,
            label='Source',
        )
        # ax = metrics.plot_segmentation(
        #     ax=ax,
        #     map=map_ls,
        # )
        ax = metrics.plot_max(
            ax=ax,
            map=map_ls,
            map_mask=map_ls,
        )

        # ax = axes[7]
        # ax = map_ls_var.plot2D(
        #     ax=ax,
        #     cmap_name='gray',
        #     title=f"Least Square Variance",
        #     feature=0,
        # )
        # ax = map_ls_var.plot2D_poss(
        #     ax=ax,
        #     poss = args(["eval", "source_pos"]),
        #     color='red',
        #     marker='*',
        #     markersize=120,
        #     label='Source',
        # )
        # ax = metrics.plot_max(
        #     ax=ax,
        #     map=map_ls_var,
        #     map_mask=map_ls,
        # )

        ax = axes[4*len(heights) + i]
        abs_max = simu.map_params.get_vals_abs_max(feature=0)
        ax = simu.map_params.plot2D(
            ax=ax,
            cmap_name='coolwarm',
            # cmap_min_max=(-abs_max, abs_max),
            title=f"Least Square Bias Weight",
            feature=0,
            height=h,
        )
        ax = simu.map_params.plot2D_poss(
            ax=ax,
            poss = args(["eval", "source_pos"]),
            color='red',
            marker='*',
            markersize=120,
            label='Source',
        )

        ax = axes[5*len(heights) + i]
        abs_max = simu.map_params.get_vals_abs_max(feature=1)
        ax = simu.map_params.plot2D(
            ax=ax,
            cmap_name='coolwarm',
            title=f"Least Square Weight y",
            feature=1,
            height=h,
        )
        ax = simu.map_params.plot2D_poss(
            ax=ax,
            poss = args(["eval", "source_pos"]),
            color='red',
            marker='*',
            markersize=120,
            label='Source',
        )

        ax = axes[6*len(heights) + i]
        abs_max = simu.map_params.get_vals_abs_max(feature=2)
        ax = simu.map_params.plot2D(
            ax=ax,
            cmap_name='coolwarm',
            title=f"Least Square Weight z",
            feature=2,
            height=h,
        )
        ax = simu.map_params.plot2D_poss(
            ax=ax,
            poss = args(["eval", "source_pos"]),
            color='red',
            marker='*',
            markersize=120,
            label='Source',
        )
        ax = metrics.plot_max(
            ax=ax,
            map=simu.map_params,
            feature=2,
            map_mask=map_ls,
        )

        ax = axes[7*len(heights) + i]
        abs_max = simu.map_params.get_vals_abs_max(feature=2)
        ax = simu.map_params.plot2D(
            ax=ax,
            cmap_name='coolwarm',
            title=f"Least Square Weight Wind",
            feature=3,
            height=h,
        )
        ax = simu.map_params.plot2D_poss(
            ax=ax,
            poss = args(["eval", "source_pos"]),
            color='red',
            marker='*',
            markersize=120,
            label='Source',
        )
        ax = metrics.plot_max(
            ax=ax,
            map=simu.map_params,
            feature=3,
            map_mask=map_ls,
        )

    plt.savefig(args(["data", "data_folder"]) + args(["data", "folder_cf"]) + exp + 'maps_3D.png')
    plt.show()