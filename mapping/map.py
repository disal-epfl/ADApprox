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
import matplotlib.colors as mcolors

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from helper.math.gaussian import gaussian_1D

class Map():
    def __init__(
        self, 
        n_dims:int=2,
        n_features:int=1,
        res:float=1.0,
        dims_min:np.ndarray=[0, 0],
        dims_max:np.ndarray=[1, 1],
        init_vals:np.ndarray=np.array([0.0]),
    ):
        """
        Initialize the map.
        Args:
            n_dims (int): Number of dimensions.
            n_features (int): Number of features.
            res (float): Resolution.
            dims_min (np.ndarray): Minimum values of the dimensions (n_dims,).
            dims_max (np.ndarray): Maximum values of the dimensions (n_dims,).
            init_vals (np.ndarray): Initial values of the map (n_features,).
        """
        # Set parameters
        self.n_dims = n_dims
        self.n_features = n_features
        self.res = res
        self.min = np.array(dims_min)
        self.max = np.array(dims_max)

        # Verify dimensions
        if self.n_dims != len(self.min) or self.n_dims != len(self.max):
            raise ValueError('Dimensions do not match.')

        # Verify resolution and determine size of the grid
        dims = ((self.max - self.min) / res)
        if ~np.all(np.abs(dims - np.round(dims)) < 1e-10):
            raise ValueError('Resolution does not match the grid size.') 
        self.dims = dims.astype(int)
        self.num_cells = np.prod(self.dims).astype(int)

        # Create grid
        self._grid = np.ones((self.num_cells, self.n_features)) * init_vals

    def get_map(
        self,
        copy_array:bool=True,
        get_flat:bool=True,
        feature:int=None,
    ):
        """
        Get the map.
        Args:
            copy_array (bool): Copy the array before returning.
            get_flat (bool): Get the flattened map.
            feature (int): Feature to get.
        Returns:
            np.ndarray: Map.
        """
        grid = self._grid # (n_flat_dims, n_features)
        if copy_array:
            grid = np.copy(grid)

        if feature is not None:
            grid = grid[:,feature] # (n_cells,)

        if not get_flat:
            if feature is None:
                grid = grid.reshape(*self.dims, self.n_features)
            else:
                grid = grid.reshape(*self.dims)

        return grid

    def get_cell_vals(
        self,
        poss:np.ndarray=None,
        idxs:np.ndarray=None,
        idxs_flat:np.ndarray=None,
        copy_array:bool=True,
        feature:int=None,
    ):
        """
        Get the values of the cells.
        Args:
            poss (np.ndarray): Positions to get the values, (n_samples, n_dims).
            idxs (np.ndarray): Indices to get the values, (n_samples, n_dims).
            idxs_flat (np.ndarray): Flattened indices to get the values, (n_samples,).
            copy_array (bool): Copy the array before returning.
            feature (int): Feature to get.
        Returns:
            np.ndarray: Cell values.
        """
        if poss is not None:
            idxs_flat = self.pos2flat(
                poss=poss,
                copy_array=False,
            )
        if idxs is not None:
            idxs_flat = self.idx2flat(
                idxs=idxs,
                copy_array=False,
            )

        grid_vals = self._grid[idxs_flat] # (n_samples, n_features)
        if copy_array:
            grid_vals = np.copy(grid_vals)
        
        if feature is None:
            return grid_vals
        return grid_vals[:, feature]

    def get_cell_poss(
        self,
        poss:np.ndarray=None,
        idxs:np.ndarray=None,
        idxs_flat:np.ndarray=None,
        copy_array:bool=True,
        return_flat:bool=True,
    ):
        """
        Get the center of the cells. By default, return all cell positions.
        Args:
            poss (np.ndarray): Positions to get the values, (n_samples, n_dims).
            idxs (np.ndarray): Indices to get the values, (n_samples, n_dims).
            idxs_flat (np.ndarray): Flattened indices to get the values, (n_samples,).
            copy_array (bool): Copy the array before returning.
            return_flat (bool): Flatten the output.
        Returns:
            np.ndarray: Cell centers (n_cells/n_samples, n_dims).
        """
        if poss is not None:
            idxs_flat = self.pos2flat(
                poss=poss,
                copy_array=False,
            )
        if idxs is not None:
            idxs_flat = self.idx2flat(
                idxs=idxs,
                copy_array=False,
            )
        if idxs_flat is None:
            idxs_flat = np.arange(self._grid.shape[0])
        
        poss = self.flat2pos(
            idxs_flat=idxs_flat,
            copy_array=True,
        ) # (n_cells/n_samples, n_dims)

        if return_flat:
            return poss
        return poss.reshape(*self.dims, self.n_dims)
    
    def get_vals_abs_max(
        self,
        feature:int=None,
    ):
        """
        Get the absolute maximal value of the grid.
        Args:
            feature (int): Feature to get. If None, return max of all features.
        Returns:
            (np.ndarray): Maximum values, (n_features).
        """
        max_ = self.get_vals_max(feature=feature)
        min_ = self.get_vals_min(feature=feature)
        return np.max([np.abs(max_), np.abs(min_)])
    
    def get_vals_max(
        self,
        feature:int=None,
    ):
        """
        Get maximal values of grid.
        Args:
            feature (int): Feature to get. If None, return max of all features.
        Returns:
            (np.ndarray): Maximum values, (n_features).
        """
        if feature is None:
            return np.max(self._grid, axis=0)
        return np.max(self._grid[:, feature])
    
    def get_vals_min(
        self,
        feature:int=None,
    ):
        """
        Get minimal values of grid.
        Args:
            feature (int): Feature to get. If None, return min of all features.
        Returns:
            (np.ndarray): Minimum values, (n_features).
        """
        if feature is None:
            return np.min(self._grid, axis=0)
        return np.min(self._grid[:, feature])
    
    def get_neighbours(
        self,
        poss:np.ndarray=None,
        idxs:np.ndarray=None,
        idxs_flat:np.ndarray=None,
        copy_array:bool=True,
    ):
        """
        Get the cell neighbours. For all border cells, the neighbour is the cell itself.
        Args:
            poss (np.ndarray): Positions to get the neighbours, (n_samples, n_dims).
            idxs (np.ndarray): Indices to get the neighbours, (n_samples, n_dims).
            idxs_flat (np.ndarray): Flattened indices to get the neighbours, (n_samples,).
            copy_array (bool): Copy the array before returning.
        Returns:
            idxs_pos (np.ndarray): Neighbours in positive direction (n_samples, n_dims).
            idxs_neg (np.ndarray): Neighbours in negative direction (n_samples, n_dims).
            border_pos (np.ndarray): Boolean array indicating if the neighbour is at the 
                                        border in positive direction (n_samples, n_dims).
            border_neg (np.ndarray): Boolean array indicating if the neighbour is at the
                                        border in negative direction (n_samples, n_dims).
        """
        if poss is not None:
            idxs_flat = self.pos2flat(
                poss=poss,
                copy_array=copy_array,
            )
        if idxs is not None:
            idxs_flat = self.idx2flat(
                idxs=idxs,
                copy_array=copy_array,
            )

        idxs_pos = np.zeros((idxs_flat.shape[0], self.n_dims), dtype=int)
        idxs_neg = np.zeros((idxs_flat.shape[0], self.n_dims), dtype=int)
        border_pos = np.zeros((idxs_flat.shape[0], self.n_dims), dtype=bool)
        border_neg = np.zeros((idxs_flat.shape[0], self.n_dims), dtype=bool)
        for i in range(self.n_dims):
            # Define indices
            idxs_pos[:, i] = idxs_flat + np.prod(self.dims[i+1:])
            idxs_neg[:, i] = idxs_flat - np.prod(self.dims[i+1:])

            # Define borders
            border_pos[:, i] = (idxs_pos[:, i] >= self.num_cells) \
                             | (idxs_pos[:, i] % np.prod(self.dims[i:]) == 0)
            border_neg[:, i] = (idxs_neg[:, i] < 0) \
                             | (idxs_neg[:, i] % np.prod(self.dims[i:]) == np.prod(self.dims[i:]) - 1)

            # Set border indices back to original indices
            idxs_pos[border_pos[:,i], i] = idxs_flat[border_pos[:,i]]
            idxs_neg[border_neg[:,i], i] = idxs_flat[border_neg[:,i]]

        return idxs_pos, idxs_neg, border_pos, border_neg

    def set_map(
        self,
        grid:np.ndarray,
        copy_array:bool=True,
    ):
        """
        Set map.
        Args:
            grid (np.ndarray): Map (n_flat_dims, n_features).
            copy_array (bool): Copy the array before setting the values.
        """ 
        if copy_array:
            grid = np.copy(grid)
        self._grid = grid.reshape(self.num_cells, self.n_features)

    def set_cell_vals(
        self,
        vals:np.ndarray,
        poss:np.ndarray=None,
        idxs:np.ndarray=None,
        idxs_flat:np.ndarray=None,
        copy_array:bool=True,
        feature:int=None,
    ):
        """
        Set the values of the cells.
        Args:
            vals (np.ndarray): Values to set, (n_samples, n_features).
            poss (np.ndarray): Positions to set the values, (n_samples, n_dims).
            idxs (np.ndarray): Indices to set the values, (n_samples, n_dims).
            idxs_flat (np.ndarray): Flattened indices to set the values, (n_samples,).
            copy_array (bool): Copy the value array before setting the values.
            feature (int): Feature to set.
        """
        if poss is not None:
            idxs_flat = self.pos2flat(
                poss=poss,
                copy_array=False,
            )
        elif idxs is not None:
            idxs_flat = self.idx2flat(
                idxs=idxs,
                copy_array=False,
            )
        
        if copy_array:
            vals = np.copy(vals)

        if vals.ndim == 1 and self.n_features == 1:
            vals = vals.reshape(-1, 1)

        if feature is None:
            self._grid[idxs_flat] = vals
        else:
            self._grid[idxs_flat, feature] = vals
        
    def discretize(
        self,
        threshold:np.ndarray,
        return_flat:bool=False,
        copy_array:bool=True,
    ):
        """
        Discretize the grid.
        Args:
            threshold (np.ndarray): Threshold value (n_features,).
            return_flat (bool): Return flattened map.
            copy_array (bool): Copy the array before returning
        Returns:
            np.ndarray: Discretized map (n_flat_dims, n_features).
        """
        grid = (self._grid > threshold).astype(bool)

        if not return_flat:
            grid = grid.reshape(*self.dims, self.n_features)

        if copy_array:
            return np.copy(grid)
        return grid
    
    def clip(
        self,
        min_max:np.ndarray,
    ):
        """
        Clip the values of the grid.
        Args:
            min_max (np.ndarray): Minimum and maximum values (n_features, 2) or (2,).
        """
        min_max = np.array(min_max).reshape(self.n_features, 2)

        self._grid = np.clip(self._grid, min_max[:, 0], min_max[:, 1])

    def normalize(
        self,
        min_max:np.ndarray,
    ):
        """
        Normalize the values of the grid.
        Args:
            min_max (np.ndarray): Minimum and maximum values (n_features, 2) or (2,).
        """
        min_max = np.array(min_max).reshape(self.n_features, 2)
        
        grid_ = (self._grid - np.min(self._grid, axis=0)) / (np.max(self._grid, axis=0) - np.min(self._grid, axis=0))
        self._grid = min_max[:, 0] + grid_ * (min_max[:, 1] - min_max[:, 0])
    
    def verify_pos(
        self,
        poss:np.ndarray,
    ):
        """
        Verify if the positions are within the grid.
        Args:
            poss (np.ndarray): Positions to verify (n_samples, n_dims).
        Returns:
            (np.ndarray): Boolean array indicating if the positions are within the grid (n_samples,).
        """
        return (poss >= self.min) & (poss <= self.max)
        
    def pos2idx(
        self, 
        poss:np.ndarray,
        copy_array:bool=True,
        round:bool=True,
    ):
        """
        Convert values to indices.
        Args:
            poss (np.ndarray): Positions to convert to indices.
            copy_array (bool): Copy the array before returning.
            round (bool): If true, round to integer indices.
        Returns:
            tuple of np.ndarray: Indices n_dims*(n_samples).
        """
        if copy_array:
            poss = np.copy(poss)

        idxs = (poss - self.min) / self.res # (n_samples, n_dims)

        if round:
            idxs = np.floor(idxs).astype(int) # (n_samples, n_dims)
        
        return tuple([idxs[:, i] for i in range(self.n_dims)])
    
    def idx2pos(
        self, 
        idxs:np.ndarray,
        copy_array:bool=True,
        round:bool=True,
    ):
        """
        Convert indices to values.
        Args:
            idxs (tuple of np.ndarray): Indices to convert to values n_dims*(n_samples).
            copy_array (bool): Copy the array before returning.
            round (bool): If true, round the indices first and return the cell centers.
        Returns:
            np.ndarray: Positions of cell centers (n_samples, n_dims).
        """
        if copy_array:
            idxs = tuple([np.copy(idxs[i]) for i in range(self.n_dims)])
        idxs = np.array(idxs).T # (n_samples, n_dims)

        if round:
            idxs = np.floor(idxs).astype(int)
            return self.min + idxs * self.res + self.res / 2
        
        return self.min + idxs * self.res
    
    def idx2flat(
        self,
        idxs:tuple,
        copy_array:bool=True,
    ):
        """
        Convert indices to flattened indices.
        Args:
            idxs (tuple of np.ndarray): Indices to convert to flattened indices n_dims*(n_samples).
            copy_array (bool): Copy the array before returning.
        Returns:
            np.ndarray: Flattened indices (n_samples,).
        """
        if copy_array:
            idxs = tuple([np.copy(idxs[i]) for i in range(self.n_dims)])
        
        return np.ravel_multi_index(
            idxs,
            self.dims,
            mode='raise',
        )

    def flat2idx(
        self,
        idxs_flat:np.ndarray,
        copy_array:bool=True,
    ):
        """
        Convert flattened indices to indices.
        Args:
            idxs_flat (np.ndarray): Flattened indices to convert to indices (n_samples,).
            copy_array (bool): Copy the array before returning.
        Returns:
            tuple of np.ndarray: Indices n_dims*(n_samples).
        """
        if copy_array:
            idxs_flat = np.copy(idxs_flat)
        
        return np.unravel_index(
            idxs_flat,
            self.dims,
            order='C',
        )
    
    def pos2flat(
        self,
        poss:np.ndarray,
        copy_array:bool=True,
    ):
        """
        Convert values to flattened indices.
        Args:
            poss (np.ndarray): Positions to convert to flattened indices (n_samples, n_dims).
            copy_array (bool): Copy the array before returning.
        Returns:
            np.ndarray: Flattened indices (n_samples,).
        """
        idxs = self.pos2idx(
            poss=poss,
            copy_array=copy_array,
        )
        return self.idx2flat(
            idxs=idxs,
            copy_array=False,
        )
    
    def flat2pos(
        self,
        idxs_flat:np.ndarray,
        copy_array:bool=True,
    ):
        """
        Convert flattened indices to values.
        Args:
            idxs_flat (np.ndarray): Flattened indices to convert to values (n_samples,).
            copy_array (bool): Copy the array before returning.
        Returns:
            np.ndarray: Positions (n_samples, n_dims).
        """
        idxs = self.flat2idx(
            idxs_flat=idxs_flat,
            copy_array=copy_array,
        )
        return self.idx2pos(
            idxs=idxs,
            copy_array=False,
        )
    
    def plot2D(
        self,
        ax:plt.Axes,
        title:str="Map",
        grid:np.ndarray=None,
        feature:int=0,
        height:float=None,
        plot_colorbar:bool=True,
        cmap_name:str='viridis',
        cmap_min_max:tuple=None,
        num_ticks:tuple=(7, 5),
        x_label:str='x [m]',
        y_lable:str='y [m]',
    ):
        """
        Plot 2D grid.
        Args:
            ax (matplotlib.axes): Axes to plot on.
            title (str): Title of the plot.
            grid (np.ndarray): Map to plot instead of proper grid (n_dims_x, n_dims_y).
            feature (int): Feature to plot.
            height (float): Height to plot. Only available for 3D maps.
            plot_colorbar (bool): If true, plot colorbar. If false, return imshow object.
            cmap_name (str): Name of the colormap.
            cmap_min_max (tuple): Minimum and maximum values of the colormap.
            num_ticks (tuple): Number of ticks for x and y.
            x_label (str): Label of the x-axis.
            y_label (str): Label of the y-axis.
        Returns:
            matplotlib.axes: Axes.
        """
        # Get 2D grid
        if grid is None:
            grid = self.get_map(
                copy_array=False,
            ) # (n_cells, n_features)
            grid = grid[:, feature] # (n_cells,)
            grid = grid.reshape(*self.dims) # (n_dims_x, n_dims_y) or (n_dims_x, n_dims_y, n_dims_z)
        else:
            assert grid.shape == self.dims, \
                    f"Provided grid shape ({grid.shape}) does not match the map shape ({self.dims})."

        # Get max and min values
        if cmap_min_max is None:
            cmap_min_max = (np.min(grid), np.max(grid))
        else:
            assert cmap_min_max[0] <= np.min(grid), \
                    f"cmap_min is too large: cmap_min={cmap_min_max[0]}, min={np.min(grid)}"
            assert cmap_min_max[1] >= np.max(grid), \
                    f"cmap_max is too small: cmap_max={cmap_min_max[1]}, max={np.max(grid)}"

        # Extract height
        if height is not None:
            assert self.n_dims == 3, "Height is only available for 3D maps."
            height_idx = self.pos2idx(
                poss=np.array([[0, 0, height]]),
                copy_array=False,
            )[2][0] # int
            grid = grid[:,:,height_idx] # (n_dims_x, n_dims_y)  
        
        # Plot
        im_obj = ax.imshow(grid.T, vmin=cmap_min_max[0], vmax=cmap_min_max[1], cmap=cmap_name, origin='lower')
        if plot_colorbar:
            plt.colorbar(im_obj, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_lable)
        x_tick_pos = np.linspace(0, grid.shape[0], num_ticks[0])
        x_tick_labels = [f'{(pos*self.res + self.min[0]):.2f}' for pos in x_tick_pos]  
        ax.set_xticks(x_tick_pos-0.5, x_tick_labels)
        y_tick_pos = np.linspace(0, grid.shape[1], num_ticks[1])
        y_tick_labels = [f'{(pos*self.res + self.min[1]):.2f}' for pos in y_tick_pos]  
        ax.set_yticks(y_tick_pos-0.5, y_tick_labels)

        if not plot_colorbar:
            return ax, im_obj

        return ax
    
    def plot2D_vector_field(
        self,
        ax:plt.Axes,
        features:tuple=(0, 1),
        title:str="Map",
        plot_colorbar:bool=True,
        cmap_name:str='viridis',
        cmap_min_max:float=None,
    ):
        """
        Plot 2D vector field between two features.
        """
        poss = self.get_cell_poss(
            idxs_flat=np.arange(self.num_cells),
        ) # (n_cells, n_dims)
        grid1 = self.get_map(
            feature=features[0],
        ) # (n_cells)
        grid2 = self.get_map(
            feature=features[1],
        ) # (n_cells)
        grid_norm = np.sqrt(grid1**2 + grid2**2) # (n_cells)

        # Get max and min values
        if cmap_min_max is None:
            cmap_min_max = (np.min(grid_norm), np.max(grid_norm))
        else:
            assert cmap_min_max[0] <= np.min(grid_norm), \
                    f"cmap_min is too large: cmap_min={cmap_min_max[0]}, min={np.min(grid_norm)}"
            assert cmap_min_max[1] >= np.max(grid_norm), \
                    f"cmap_max is too small: cmap_max={cmap_min_max[1]}, max={np.max(grid_norm)}"  

        # Plot
        arrow_scale = 50.0
        arrow_width = 0.005
        obj = ax.quiver(poss[:,features[0]], poss[:,features[1]], grid1, grid2, grid_norm, scale=arrow_scale, scale_units='xy', 
                        cmap=cmap_name, norm=mcolors.Normalize(vmin=cmap_min_max[0], vmax=cmap_min_max[1]), width=arrow_width)
        if plot_colorbar:
            plt.colorbar(obj, ax=ax)
        ax.set_aspect('equal')
        ax.set_title(title)
        return ax
    
    def plot2D_poss(
        self,
        ax:plt.Axes,
        poss:np.ndarray,
        color:str='r',
        marker:str='o',
        markersize:int=120,
        label:str='Positions',
    ):
        """
        Plot positions using scatter plot.
        Args:
            ax (matplotlib.axes): Axes to plot on.
            poss (np.ndarray): Positions to plot (n_samples, n_dims).
            color (str): Color of the plot.
            marker (str): Marker of the plot.
            markersize (int): Size of the marker.
            label (str): Label of the plot.
        Returns:
            matplotlib.axes: Axes.
        """
        idxs = self.pos2idx(
            poss=poss,
            round=False,
        ) # n_dims*(n_samples)
        ax.scatter(idxs[0] - 0.5, idxs[1] - 0.5, c=color, label=label, marker=marker, s=markersize)
        return ax
    
    def plot2D_traj(
        self,
        traj:np.ndarray,
        ax:plt.Axes,
        cmap_name:str='Oranges',
        crop_traj:bool=False,
    ):
        """
        Plot the trajectory in 2D.
        Args:
            traj (np.ndarray): Trajectory (n_traj, n_dims).
            ax (matplotlib.axes.Axes): Axes to plot.
            cmap_name (str): Name of the color map.
            crop_traj (bool): Crop the trajectory at the map border.
        Returns:
            ax (matplotlib.axes.Axes): Axes with the plot.
        """
        if crop_traj:
            traj = traj[
                (traj[:, 0] >= self.min[0]) & (traj[:, 0] <= self.max[0]) & \
                (traj[:, 1] >= self.min[1]) & (traj[:, 1] <= self.max[1])
            ]

        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i/traj.shape[0]) for i in range(traj.shape[0])]

        idxs = self.pos2idx(
            poss=traj,
            round=False,
        ) # n_dims*(n_samples)

        for i in range(traj.shape[0]-1):
            ax.plot(
                [idxs[0][i]- 0.5, idxs[0][i+1]- 0.5],
                [idxs[1][i]- 0.5, idxs[1][i+1]- 0.5],
                color=colors[i],
            )
        return ax
            
    
    def plot_cons(
        self,
        ax:plt.Axes,
        cons:np.ndarray,
        cons_min_max:tuple=None,
        bins:int=30,
        fit_gauss:bool=False,
        title:str='Measurements',
        return_bins:bool=False,
    ):
        """
        Plot the calibration data and the fitted gaussian to the calibration data.
        Args:
            ax: Matplotlib axis object.
            cons (np.ndarray): Gas concentration measurements (n_meas,).
            title (str): Plot title.
        """
        if cons_min_max is None:
            cons_min_max = (np.min(cons), np.max(cons))
        else:
            assert np.min(cons) > cons_min_max[0] and np.max(cons) < cons_min_max[1] 

        _, bins, _ = ax.hist(cons, bins=bins, weights=np.ones(cons.shape[0])/cons.shape[0], \
                                alpha=0.8, color='blue', label=title)
        ax.axvline(x=np.mean(cons), color='orange', linestyle=':', label='Mean')
        
        if fit_gauss:
            x = np.linspace(np.min(cons), np.max(cons), 500)
            y = gaussian_1D(
                x=x,
                mu=np.mean(cons),
                sigma=np.std(cons),
            )
            ax.plot(x, y, label='Calibration fit', color='orange')

        ax.set_ylabel('Frequency [%]')
        ax.set_xlabel('Gas concentration [mV]')
        ax.set_xlim(cons_min_max[0], cons_min_max[1])
        ax.set_title(title)
        ax.legend()

        if return_bins:
            return ax, bins
        return ax
    
    # def plot_calibration(
    #     self,
    #     ax:plt.Axes,
    #     cons:np.ndarray,
    #     cons_calib:np.ndarray,
    #     title:str='Calibration',
    # ):
    #     """
    #     Plot the calibration data and the fitted gaussian to the calibration data.
    #     Args:
    #         ax: Matplotlib axis object.
    #         cons (np.ndarray): Gas concentration measurements (n_meas,).
    #         cons_calib (np.ndarray): Calibration data (n_calib,).
    #         title (str): Plot title.
    #     """
    #     # Plot histogram of calibration data
    #     bins = 20
    #     _, bins, _ = ax.hist(cons, bins=bins, weights=np.ones(cons.shape[0])/cons.shape[0], \
    #             alpha=0.8, color='green', label='Measurements')
    #     ax.hist(cons_calib, bins=bins, weights=np.ones(cons_calib.shape[0])/cons_calib.shape[0], \
    #             alpha=0.3, color='blue', label='Calibration data')
        
    #     # fit gaussian to calibration data
    #     x = np.linspace(np.min(cons_calib), np.max(cons_calib), 100)
    #     y = gaussian_1D(
    #         x=x,
    #         mu=np.mean(cons_calib),
    #         sigma=np.std(cons_calib),
    #     )
    #     ax.plot(x, y, 'r--', label='Calibration fit', color='darkblue')

    #     ax.set_ylabel('Frequency [%]')
    #     ax.set_xlabel('Gas concentration [mV]')
    #     ax.axvline(x=np.mean(cons_calib), color='darkblue', linestyle=':', label='Calibration mean')
    #     ax.set_title(title)
    #     ax.legend()

    


