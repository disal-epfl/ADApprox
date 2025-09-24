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

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from mapping.map import Map


class Metrics():
    def __init__(
        self,
        map_gt: Map = None,
        source_pos: np.ndarray = None,
        metric_names: List[str] = ["rmse", "shape"],
    ):
        
        # Parameters
        self.metric_names = metric_names

        # Ground truth
        self.map_gt = map_gt
        self.source_pos = source_pos
        if self.map_gt is not None:
            self.thr_gt = self._otsu_thr(
                grid = self.map_gt.get_map(),
            )

    def __call__(
        self,
        map_c: Map,
        feature: int = 0,
        map_mask: Map = None,
        feature_mask: int = 0,
        metric_names: List[str] = None,
    ):
        """
        Compute the metrics.
        Args:
            
            map_c (Map): Map to evaluate.
            feature (int): Feature to evaluate.
            map_mask (Map): Map to extract shape used as mask
            feature_mask (int): Feature to evaluate map_mask.
            metric_names (list): List of metric names.
        Returns:
            results (dict): Dictionary with metric names and values pairs.
        """
        if metric_names is None:
            metric_names = self.metric_names
        
        results = {}
        for name in metric_names:
            metric_fct = getattr(self, f"_{name}")
            results[name] = metric_fct(
                map_c=map_c,
                feature=feature,
                map_mask=map_mask,
                feature_mask=feature_mask,
            )

        return results
    
    def _rmse(
        self,
        map_c: Map,
        feature: int,
        map_mask: Map = None,
        feature_mask: int = 0,
    ):
        """
        Compute the Root Mean Squared Error (RMSE).
        Args:
            map_c (Map): Map to evaluate.
            feature (int): Feature to evaluate.
            map_mask (Map): Not used.
            feature_mask (int): Not used.
        Returns:
            (float): RMSE.
        """
        grid_c = map_c.get_map(feature=feature)
        grid_gt = self.map_gt.get_map().flatten()

        return np.sqrt(np.mean((grid_c - grid_gt)**2)) # (1,)
    
    def _mae(
        self,
        map_c: Map,
        feature: int,
        map_mask: Map = None,
        feature_mask: int = 0,
    ):
        """
        Compute the Mean Absolute Error (MAE).
        Args:
            map_c (Map): Map to evaluate.
            feature (int): Feature to evaluate.
            map_mask (Map): Not used.
            feature_mask (int): Not used.
        Returns:
            (float): MAE.
        """
        grid_c = map_c.get_map(feature=feature)
        grid_gt = self.map_gt.get_map().flatten()

        return np.mean(np.abs(grid_c - grid_gt)) # (1,)
    
    def _shape(
        self,
        map_c: Map,
        feature: int,
        map_mask: Map = None,
        feature_mask: int = 0,
    ):
        """
        Compute the Shape metric.
        Args:
            map_c (Map): Map to evaluate.
            feature (int): Feature to evaluate.
            map_mask (Map): Not used.
            feature_mask (int): Not used.
        Returns:
            (float): Shape metric [0, 1], larger is better.
        """
        grid_c = map_c.get_map(feature=feature)
        grid_gt = self.map_gt.get_map().flatten()

        # Compute optimal threshold using Otsu's method
        thr_c = self._otsu_thr(
            grid = grid_c,
        )

        # Compute binary maps
        mask_c = grid_c > thr_c
        mask_gt = grid_gt > self.thr_gt

        # Compute score
        return (np.sum(mask_c & mask_gt) + np.sum(~mask_c & ~mask_gt)) / grid_c.size
    
    def _gsl_max(
        self,
        map_c: Map,
        feature: int,
        map_mask: Map = None,
        feature_mask: int = 0,
    ):
        """
        Compute the Gas Source Localization error based on the maximum value.
        Args:
            map_c (Map): Map to evaluate.
            feature (int): Feature to evaluate.
            map_mask (Map): Map to extract shape used as mask
            feature_mask (int): Feature to evaluate map_mask.
        Returns:
            (float): GSL error.
        """
        idx_max = self._max_idx(
            map_c=map_c,
            feature=feature,
            map_mask=map_mask,
            feature_mask=feature_mask,
        )

        pos_max = map_c.get_cell_poss(
            idxs_flat=[idx_max],
        )[0]

        return np.linalg.norm(pos_max - self.source_pos)
    
    def _max_idx(
        self,
        map_c: Map,
        feature: int,
        map_mask: Map = None,
        feature_mask: int = 0,
    ):
        """
        Compute the Gas Source Localization error based on the maximum value.
        Args:
            map_c (Map): Map to evaluate.
            feature (int): Feature to evaluate.
            map_mask (Map): Map to extract shape used as mask
            feature_mask (int): Feature to evaluate map_mask.
        Returns:
            (float): index of the maximum value.
        """
        grid_c = map_c.get_map(feature=feature)

        if map_mask is not None:
            grid_mask = map_mask.get_map(feature=feature_mask)
            thr = self._otsu_thr(
                grid = grid_mask,
            )
            mask = grid_mask > thr

            # downsample mask if necessary
            if mask.shape != grid_c.shape:
                m = np.power((mask.shape[0] / grid_c.shape[0]), (1 / map_c.n_dims))
                assert m.is_integer(), "Mask and map have incompatible dimensions."
                m = int(m)

                if map_c.n_dims == 2:
                    mask = mask.reshape((map_c.dims[0], m, map_c.dims[1], m))
                    mask = mask.any(axis=(1, 3))
                elif map_c.n_dims == 3:
                    mask = mask.reshape((map_c.dims[0], m, map_c.dims[1], m, map_c.dims[2], m))
                    mask = mask.any(axis=(1, 3, 5))
                else:
                    raise ValueError("Invalid number of dimensions.")
                mask = mask.flatten()

            grid_c = grid_c * mask

        return np.argmax(grid_c)

    def _otsu_thr(
        self,
        grid,
    ):
        """
        Computes the Otsu threshold for a flattened grayscale image and 
        Parameters:
            grid (np.array): Flattened map values.
        Returns:
            thr (float): Optimal threshold.
        """
        # Compute histogram
        hist, bin_edges = np.histogram(grid, bins=100, range=(np.min(grid), np.max(grid)))
        hist_norm = hist / hist.sum()
        
        # Compute cumulative sums and cumulative means
        cum_sum = np.cumsum(hist_norm)
        cum_mean = np.cumsum(hist_norm * np.arange(len(hist)))
        global_mean = cum_mean[-1]
        
        # Compute between-class variance for each threshold and pick the one that maximizes it
        between_class_variance =  (cum_mean - global_mean*cum_sum)**2 / (cum_sum * (1 - cum_sum) + 1e-9)
        thr = np.argmax(between_class_variance)
        
        return thr
    
    def plot_segmentation(
        self,
        ax: plt.Axes,
        map: Map,
        feature: int = 0,
    ):
        """
        Plot the segmentation.
        Args:
            ax (plt.Axes): Axis to plot on.
            map (Map): Map to evaluate.
            feature (int): Feature to plot.
        Returns:
            ax (plt.Axes): Axis with the plot.
        """
        grid = map.get_map(
            get_flat=False,
            feature=feature,
        )

        # Compute optimal threshold using Otsu's method
        thr_c = self._otsu_thr(
            grid = grid.flatten(),
        )

        # Plot contours
        ax.contour(grid.T, levels=[thr_c], colors='red', linewidths=1)

        return ax
    
    def plot_max(
        self,
        ax: plt.Axes,
        map: Map,
        feature: int = 0,
        map_mask: Map = None,
        feature_mask: int = 0,
    ):
        """
        Plot the maximum.
        Args:
            ax (plt.Axes): Axis to plot on.
            map (Map): Map to evaluate.
            feature (int): Feature to plot.
            map_mask (Map): Map to extract shape used as mask
            feature_mask (int): Feature to evaluate map_mask.
        Returns:
            ax (plt.Axes): Axis with the plot.
        """
        idx_max = self._max_idx(
            map_c=map,
            feature=feature,
            map_mask=map_mask,
            feature_mask=feature_mask,
        )

        pos_max = map.get_cell_poss(
            idxs_flat=[idx_max],
        )[0]

        ax = map.plot2D_poss(
            ax=ax,
            poss=[pos_max],
            color='green',
            label="Max",
        )

        return ax
    
    def plot_squared_error(
        self,
        ax: plt.Axes,
        map: Map,
        feature: int = 0,
        title: str = "Squared Error",
        cmap_name: str = "hot",
        cmap_min_max: Tuple[float] = None,
    ):
        """
        Plot the squared error.
        Args:
            ax (plt.Axes): Axis to plot on.
            map (Map): Map to evaluate.
            feature (int): Feature to plot.
            title (str): Title of the plot.
            cmap_name (str): Name of the colormap.
            cmap_min_max (tuple): Min and max values of the colormap.
        Returns:
            ax (plt.Axes): Axis with the plot.
        """
        grid_c = map.get_map(feature=feature)
        grid_gt = self.map_gt.get_map().flatten()

        map_rmse = Map(
            res=map.res,
            dims_min=map.min,
            dims_max=map.max,
        )
        map_rmse.set_map(
            grid=(grid_c - grid_gt)**2,
        )

        ax = map_rmse.plot2D(
            ax=ax,
            title=title,
            cmap_name=cmap_name,
            cmap_min_max=cmap_min_max,
        )

        return ax
    
    def plot_abs_error(
        self,
        ax: plt.Axes,
        map: Map,
        feature: int = 0,
        title: str = "Squared Error",
        cmap_name: str = "hot",
        cmap_min_max: Tuple[float] = None,
    ):
        """
        Plot the absolute error.
        Args:
            ax (plt.Axes): Axis to plot on.
            map (Map): Map to evaluate.
            feature (int): Feature to plot.
            title (str): Title of the plot.
            cmap_name (str): Name of the colormap.
            cmap_min_max (tuple): Min and max values of the colormap.
        Returns:
            ax (plt.Axes): Axis with the plot.
        """
        grid_c = map.get_map(feature=feature)
        grid_gt = self.map_gt.get_map().flatten()

        map_rmse = Map(
            res=map.res,
            dims_min=map.min,
            dims_max=map.max,
        )
        map_rmse.set_map(
            grid=np.abs(grid_c - grid_gt),
        )

        ax = map_rmse.plot2D(
            ax=ax,
            title=title,
            cmap_name=cmap_name,
            cmap_min_max=cmap_min_max,
        )

        return ax
    