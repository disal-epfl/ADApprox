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

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from mapping.map import Map


class KernelDMV():
    def __init__(
        self, 
        n_dims:int,
        res:float,
        dims_min:np.ndarray,
        dims_max:np.ndarray,
        cons_min:float,
        cons_max:float,
        cons_init_value:float=0.0,
        kernel_std:float=0.3,
        kernel_infl_radius=None,
        kernel_stretching:float=1.0,
        kernel_weight:float=None,
        map_w:Map=None,
    ):
        """
        Args:
            n_dims (int): Number of dimensions.
            res (float): Resolution of the map.
            dims_min (np.ndarray): Minimum values of the map dimensions.
            dims_max (np.ndarray): Maximum values of the map dimensions.
            cons_min (float): Minimum concentration value.
            cons_max (float): Maximum concentration value.
            cons_init_value (float): Initial concentration value.
            kernel_std (float): Kernel sigma not stretched.
            kernel_infl_radius (float): Influence radius of the kernel.
            kernel_stretching (float): Kernel stretching factor.
            kernel_weight (float): Kernel weight between measurement and prior (sigma omega in paper).
            map_w (Map): Wind map. If None, the kernel is isotropic
                                and not stretched in the wind direction.
        """

        # Parameters
        self.cons_min = cons_min # TODO: Add this as calibration
        self.cons_max = cons_max
        self.first_update = True
        self.kernel_weight = kernel_weight
        self.kernel_std = kernel_std
        self.kernel_infl_radius = kernel_infl_radius
        self.kernel_stretching = kernel_stretching
        self.stretch_inf_radius = False

        if self.kernel_infl_radius is None:
            self.kernel_infl_radius = 4*kernel_std

        # Maps
        self.map_c = Map( # gas concentration
            n_dims=n_dims,
            n_features=1,
            res=res,
            dims_min=dims_min,
            dims_max=dims_max,
            init_vals=cons_init_value,
        )
        self.map_w = map_w # wind
        self.map_a = Map( # kernel weights
            n_dims=n_dims,
            n_features=1,
            res=res,
            dims_min=dims_min,
            dims_max=dims_max,
            init_vals=0.0,
        )
        self.map_var = Map( # kernel variances
            n_dims=n_dims,
            n_features=1,
            res=res,
            dims_min=dims_min,
            dims_max=dims_max,
            init_vals=0.0,
        )

    def update(
        self,
        poss:np.ndarray,
        cons:np.ndarray,
    ):
        """
        Update the map with the provided data. Assumes chronological order of measurements.
        The first update is done in batch mode to optimize performance.
        The second and subsequent updates are done in sequential mode.
        Args:
            poss (np.ndarray): Measurement positions (n_meas, n_dims).
            cons (np.ndarray): Gas concentration values (n_meas,).
        """
        # First update
        if self.first_update:
            self.update_first_time(
                poss=poss,
                cons=cons,
            )
            return
        
        raise NotImplementedError
    
    def update_first_time(
        self,
        poss:np.ndarray,
        cons:np.ndarray,
    ):
        """
        Update the map with the provided data. Assumes first time step.
        Args:
            poss (np.ndarray): Measurement positions (n_meas, n_dims).
            cons (np.ndarray): Gas concentration values (n_meas,).
        """
        assert self.first_update==True, "Only for first update."
        self.first_update = False

        # Normalize concentration values
        vals = self.con2val(
            cons=cons,
        ) # (n_meas,)
        # vals = cons # TODO: decide what to do

        # Calculate weights
        weights, meas_idxs, cell_idxs = self.kernel_weights(
            meas_poss=poss,
        ) # (n_infl_cells,), (n_infl_cells,), (n_infl_cells,)
        assert np.all(weights>=0), f"weights should be positive, but min(weights)={np.min(weights)}"

        omegas = np.zeros((self.map_a.num_cells,)) # (n_cells,)
        np.add.at(omegas, cell_idxs, weights) 

        # Calculate readings
        readings = np.zeros((self.map_c.num_cells,)) # (n_cells,)
        np.add.at(readings, cell_idxs, vals[meas_idxs]*weights) # (n_cells,) 

        mask_ = (omegas > 1e-7)
        readings[mask_] = readings[mask_] / omegas[mask_] # (n_cells,)

        # calculate variance
        variances = np.zeros((self.map_c.num_cells,))
        np.add.at(variances, cell_idxs, weights*(vals[meas_idxs] - readings[cell_idxs])**2)

        maks_ = (omegas > 1e-7)
        variances[maks_] = variances[maks_] / omegas[maks_]

        # apply prior
        if self.kernel_weight is not None:
            alphas = 1 - np.exp(- omegas / self.kernel_weight**2)

            readings = alphas*readings + (1-alphas)*np.mean(vals)
            variances = alphas*variances + (1-alphas)*np.mean((vals[meas_idxs] - readings[cell_idxs])**2)

        # Set the map values
        self.map_c.set_map(
            grid=readings,
        )
        self.map_var.set_map(
            grid=variances,
        )
    
    def reverse_scaling(
        self,
    ):
        """
        Reverse the concentration scaling with respect to maximum
        and minimal concentration value.
        """
        grid = self.map_c.get_map()
        grid = self.val2con(
            vals=grid
        )
        self.map_c.set_map(
            grid=grid,
        )

    def con2val(
        self,
        cons:np.ndarray,
        copy_array:bool=True,
    ):
        """
        Convert gas concentration values to filter values.
        Args:
            cons (np.ndarray): Gas concentration values (n_meas,).
            copy_array (bool): Copy the input array.
        Returns:
            np.ndarray: Filter values (n_meas,).
        """               
        if copy_array:
            cons = cons.copy()

        return (cons - self.cons_min) / (self.cons_max - self.cons_min)

    def val2con(
        self,
        vals:np.ndarray,
        copy_array:bool=True,
    ):
        """
        Convert log-odds values to gas concentration values.
        Args:
            vals (np.ndarray): Glog-odd values (n_meas,).
            copy_array (bool): Copy the input array.
        Returns:
            np.ndarray: Gas concentration values (n_meas,).
        """
        if copy_array:
            vals = vals.copy()
        
        return vals*(self.cons_max - self.cons_min) + self.cons_min
    
    def kernel_weights(
        self,
        meas_poss:np.ndarray,
        normalize:bool=False,
    ):
        """
        Calculate the weights for the provided positions.
        Args:
            meas_poss (np.ndarray): Positions of measurements, (n_meas, n_dims).
            normalize (bool): If false, return the normal distribution. If true, normalize the
                                output to [0,1], i.e. do not scale Gaussian by the determinant.
        Returns:
            weights (np.ndarray): Weights, list of (n_infl_cells,).
            meas_idxs (np.ndarray): Indices of measurements, list of (n_infl_cells,).
            cell_idxs (np.ndarray): Indices of influenced cells, list of (n_infl_cells,).
        """        
        cell_poss = self.map_c.get_cell_poss() # (n_cells, n_dims)

        meas_idxs, cell_idxs = self._kernel_infl_cells(
            meas_poss=meas_poss,
            cell_poss=cell_poss,
        ) # (n_infl_cells), (n_infl_cells)

        kernel_inv = self._kernel_stretch(
            meas_poss=meas_poss,
            meas_idxs=meas_idxs,
        ) # (n_dims, n_dims) or (n_meas, n_dims, n_dims)
        
        weights = self._gaussian(
            cell_poss=cell_poss[cell_idxs],
            meas_poss=meas_poss[meas_idxs],
            kernel_inv=kernel_inv,
            normalize=normalize,
        ) # (n_infl_cells,)

        return weights, meas_idxs, cell_idxs
    
    def _kernel_stretch(
        self,
        meas_poss:np.ndarray,
        meas_idxs:np.ndarray,
    ):
        """
        Stretch the kernel in the wind direction if wind map is provided.
        Otherwise, return the isotropic kernel.
        Args:
            meas_poss (np.ndarray): Measurement positions, (n_meas, n_dims).
            meas_idxs (np.ndarray): Measurement indices, (n_infl_cells,).
        Returns:
            kernel_inv (np.ndarray): Inverse of the kernel, (n_infl_cells, n_dims, n_dims).
        """
        if self.map_w is None:
            return self.kernel_std * np.identity(self.map_c.n_dims) # (n_dims, n_dims)
        
        wind = self.map_w.get_cell_vals(
            poss=meas_poss,
        ) # (n_meas, n_dims)
        wind_speed = np.linalg.norm(wind, axis=1) # (n_meas)

        R, R_inv = self._wind_rotation_matrix(
            meas_poss=meas_poss,
        ) # (n_meas, n_dims, n_dims)

        kernel_diag = np.zeros((meas_poss.shape[0], self.map_c.n_dims)) # (n_meas, n_dims)
        kernel_diag[:, 0] = self.kernel_std + self.kernel_stretching * wind_speed
        if meas_poss.shape[1] == 2:
            kernel_diag[:, 1] = self.kernel_std / (1 + self.kernel_stretching * wind_speed / self.kernel_std)
        else:
            kernel_diag[:, 1] = self.kernel_std / np.sqrt(1 + self.kernel_stretching * wind_speed / self.kernel_std)
            kernel_diag[:, 2] = kernel_diag[:, 1]

        kernel_inv = np.zeros((meas_poss.shape[0], self.map_c.n_dims, self.map_c.n_dims)) # (n_meas, n_dims, n_dims)
        kernel_inv[:, 0, 0] = 1 / kernel_diag[:, 0]
        kernel_inv[:, 1, 1] = 1 / kernel_diag[:, 1]
        if meas_poss.shape[1] == 3:
            kernel_inv[:, 2, 2] = 1 / kernel_diag[:, 2]
        
        kernel_inv = np.einsum('nli,nij,njk->nlk', R, kernel_inv, R_inv) # (n_meas, n_dims, n_dims)

        return kernel_inv[meas_idxs] # (n_infl_cells, n_dims, n_dims)

    def _kernel_infl_cells(
        self,
        meas_poss:np.ndarray,
        cell_poss:np.ndarray,
    ):
        """
        Calculate the influenced cells for the provided positions,
        i.e., the cells that are within the influence radius. Return
        all cell indices if the influence radius is None.
        Args:
            meas_poss (np.ndarray): Measurement positions, (n_meas, n_dims).
            cell_poss (np.ndarray): Cell positions, (n_cells x n_dims).
        Returns:
            meas_idxs: Indices of measurements, (n_infl_cells,).
            cell_idxs: Indices of influenced cells, (n_infl_cells,).
        """
        if self.kernel_infl_radius is None:
            cell_idxs = np.arange(self.map_c.num_cells) # (n_cells)
            cell_idxs = np.repeat(cell_idxs[np.newaxis, :], meas_poss.shape[0], axis=0).flatten() # (n_infl_cells)
            meas_idxs = np.arange(meas_poss.shape[0]) # (n_meas)
            meas_idxs = np.repeat(meas_idxs[:, np.newaxis], self.map_c.num_cells, axis=1).flatten() # (n_infl_cells)
            return meas_idxs, cell_idxs
        
        n_meas = meas_poss.shape[0]
        n_cells = cell_poss.shape[0]
        n_dims = meas_poss.shape[1]

        wind = self.map_w.get_cell_vals(
            poss=meas_poss,
        ) # (n_meas, n_dims)
        wind_speed = np.linalg.norm(wind, axis=1) # (n_meas)
        wind_speed = np.repeat(wind_speed[:, np.newaxis], n_cells, axis=1) # (n_meas, n_cells)

        R, R_inv = self._wind_rotation_matrix(
            meas_poss=meas_poss,
        ) # (n_meas, n_dims, n_dims)

        cell_poss = np.repeat(cell_poss[np.newaxis, :, :], n_meas, axis=0) # (n_meas, n_cells, n_dims)
        meas_poss = np.repeat(meas_poss[:, np.newaxis, :], n_cells, axis=1) # (n_meas, n_cells, n_dims)

        # rotate distances
        delta = cell_poss - meas_poss # (n_meas, n_cells, n_dims)
        delta = np.einsum('nij,nkj->nki', R_inv, delta) # (n_meas, n_cells, n_dims)

        # calculate distance metric
        stretch = self.kernel_infl_radius
        if not self.stretch_inf_radius or self.kernel_stretching is None:
            stretch = 0.0

        ellipsoid_ax1 = self.kernel_infl_radius + stretch * wind_speed
        if n_dims == 2:
            ellipsoid_ax2 = self.kernel_infl_radius / (1 + stretch * wind_speed / self.kernel_infl_radius)
            dist = np.sqrt((delta[:, :, 0] / ellipsoid_ax1)**2 \
                           + (delta[:, :, 1] / ellipsoid_ax2)**2) # (n_meas, n_cells)
        elif n_dims == 3:
            ellipsoid_ax2 = self.kernel_infl_radius / np.sqrt(1 + stretch * wind_speed / self.kernel_infl_radius)
            ellipsoid_ax3 = ellipsoid_ax2
            dist = np.sqrt((delta[:, :, 0] / ellipsoid_ax1)**2 \
                           + (delta[:, :, 1] / ellipsoid_ax2)**2 \
                           + (delta[:, :, 2] / ellipsoid_ax3)**2) # (n_meas, n_cells)
        else:
            raise ValueError("Only 2D and 3D are supported.")


        mask = dist < 1.0 # (n_meas, n_cells)
        meas_idxs, cell_idxs = np.where(mask) # (n_infl_cells), (n_infl_cells)

        return meas_idxs, cell_idxs
    
    def _wind_rotation_matrix(
        self,
        meas_poss:np.ndarray,
    ):
        """
        Calculate the rotation matrix based on the wind direction.
        Args:
            meas_poss (np.ndarray): Measurement positions, (n_meas, n_dims).
        Returns:
            R (np.ndarray): Rotation matrix, (n_meas, n_dims, n_dims).
            R_inv (np.ndarray): Inverse of the rotation matrix, (n_meas, n_dims, n_dims).
        """
        wind = self.map_w.get_cell_vals(
            poss=meas_poss,
        ) # (n_meas, n_dims)
        wind_speed = np.linalg.norm(wind, axis=1) # (n_meas)
        wind_dir = np.arctan2(wind[:, 1], wind[:, 0]) # (n_meas)

        if meas_poss.shape[1] == 2:
            R = np.array([
                [np.cos(wind_dir), -np.sin(wind_dir)],
                [np.sin(wind_dir), np.cos(wind_dir)],
            ]) # (2, 2, n_meas)
            R = R.transpose(2,0,1) # (n_meas, 2, 2)
        else:
            # if v1 is not aligned with the y-axis
            v1 = wind / wind_speed[:,None] # (n_meas, n_dims)
            v2 = np.cross(np.array([[0, 1, 0]]), v1) # (n_meas, n_dims)
            v3 = np.cross(v1, v2) # (n_meas, n_dims)
            
            # if v1 is aligned with the y-axis
            mask = np.all(v2 < 1e-6, axis=1) # (n_meas)
            v2b = np.cross(np.array([[0, 0, 1]]), v1) # (n_meas, n_dims)
            v3b = np.cross(v1, v2b) # (n_meas, n_dims)
            v2[mask] = v2b[mask]
            v3[mask] = v3b[mask]

            R = np.stack([v1, v2, v3], axis=2) # (n_meas, n_dims, n_dims)

        R_inv = R.transpose(0, 2, 1)
        return R, R_inv
    

    def _gaussian(
        self,
        cell_poss:np.ndarray,
        meas_poss:np.ndarray,
        kernel_inv:np.ndarray,
        normalize:bool=False,
    ):
        """
        Calculate the Gaussian function.
        Args:
            cell_poss (np.ndarray): Cell positions, (n_infl_cells, n_dims).
            meas_poss (np.ndarray): Measurement positions, (n_infl_cells, n_dims).
            kernel_inv (np.ndarray): Inverse of kernel, (n_infl_cells, n_dims, n_dims).
            normalize (bool): If false, return the normal distribution. If true, normalize the 
                                output to [0,1], i.e. do not scale Gaussian by the determinant.
        Returns:
            np.ndarray: Gaussian function (n_samples,).
        """
        if cell_poss.shape[0] == 0:
            return np.array([])
        
        X = cell_poss - meas_poss

        exp = np.exp(-0.5 * np.einsum('ni,nij,nj->n', X, kernel_inv, X)) 

        if normalize:
            return exp
        
        if cell_poss.shape[1] == 2:
            return exp / (2*np.pi * self.kernel_std)
        return exp / np.sqrt((2*np.pi)**X.shape[1] * self.kernel_std**3)

    
