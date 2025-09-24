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
from matplotlib import pyplot as plt
from typing import Tuple

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from data_preparation.data_loader import load_data
from data_preparation.signal_processing import SignalProcessing


def plot_cf(
    sp: SignalProcessing,
):
    """
    Plotting.
    Args:
        sp (SignalProcessing): Signal processing object.
    """
    # Additional calculations
    sp.derivative(
        signal_keys=['cons', 'cons_ma', 'cons_ma_deconv1'],
        avg_window=100,
    )

    # Plotting
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(40, 20), tight_layout=True)
    axes = axes.flatten()

    poss = sp.get_data(
        keys=['poss'],
    )['poss']
    center_mask = (poss[:,1] > 1.95) & (poss[:,1] < 2.05)

    ax = axes[0]
    ax = sp.plot_data(
        ax=ax,
        y_axis_keys=['cons'],
        title='Cons',
        fill_area_mask=center_mask,
    )

    ax = axes[1]
    ax = sp.plot_data(
        ax=ax,
        y_axis_keys=['cons_grad'],
        title='Cons derivative',
        fill_area_mask=center_mask,
    )

    ax = axes[2]
    ax = sp.plot_data(
        ax=ax,
        y_axis_keys=['cons_ma'],
        title='Cons MA',
        fill_area_mask=center_mask,
    )

    ax = axes[3]
    ax = sp.plot_data(
        ax=ax,
        y_axis_keys=['cons_ma_grad'],
        title='Cons MA derivative',
        fill_area_mask=center_mask,
    )

    ax = axes[4]
    ax = sp.plot_data(
        ax=ax,
        y_axis_keys=['cons_ma_deconv1'],
        title='Cons MA Deconv 1',
        fill_area_mask=center_mask,
    )

    ax = axes[5]
    ax = sp.plot_data(
        ax=ax,
        y_axis_keys=['cons_ma_deconv1_grad'],
        title='Cons EMA Deconv 1 derivative',
        fill_area_mask=center_mask,
    )

    plt.show()

def cf_pipeline(
    file_folder: str,
    file_name: str,
    poss_min_max: Tuple[np.array, np.array],
    cons_min_max: tuple,
    param_time_delay: float = 2.0,
    param_ma_window_length: int = 100,
    param_gain: float = 1.0,
    param_time_constant: float = 0.6,
    plot_results: bool = False,
    print_stats: bool = False,
):
    """
    Signal processing.
    Args:
        file_folder (str): Folder with the file.
        file_name (str): File name.
        poss_min_max (tuple): Min and max position values.
        cons_min_max (tuple): Min and max consumption values.
        param_time_delay (float): Time delay.
        param_ma_window_length (int): Moving average window length.
        param_gain (float): Filter gain.
        param_time_constant (float): Time constant.
        plot_results (bool): Plot results
        print_stats (bool): Print stats
    Returns:
        poss (np.array): Trajectory (N, 2)
        cons (np.array): Consumption (N,)
        time (np.array): Time (N,)
    """

    # Load data
    data, column_names = load_data(
        csv_file=file_folder + file_name,
    )
    assert column_names[0] == 'cf_odor_time' and column_names[1] == 'cf_x' and column_names[2] == 'cf_y' \
            and column_names[3] == 'cf_z' and column_names[4] == 'cf_odor'
    
    # data_blank, column_names_blank = load_data(
    #     csv_file=file_folder + file_blank_name,
    # )
    # assert column_names_blank[0] == 'cf_odor_time' and column_names_blank[1] == 'cf_x' and column_names_blank[2] == 'cf_y' \
    #         and column_names_blank[3] == 'cf_z' and column_names_blank[4] == 'cf_odor'
    
    # Signal processing
    sp = SignalProcessing(
        data={
            'poss': data[:, 1:4],
            'cons': data[:, 4],
            'time': data[:, 0],
        },
        print_stats=print_stats,
    )

    # Remove unwanted data
    sp.remove_outliers(
        data_lims={
            'cons': cons_min_max,
            'poss': poss_min_max,
        },
    )
    sp.remove_duplicates(
        keys=('time',),
    )
    
    # Remove offset
    """
    TODO: Replace this with a proper calibration blank measurement
    """
    data = sp.get_data(
        keys=['cons', 'poss'],
    )
    mask_ = data['poss'][:, 0] > 8.75
    offset = - np.mean(data['cons'][mask_])
    if print_stats:
        print(f"Offset: {offset}")
    """
    TODO: END
    """
    sp.apply_offsets(
        offsets_dict={
            'cons': offset,
        },
    )

    # Filter data
    sp.apply_interpolation(
        interpolation_key='time',
        frequency=100,
    )
    sp.time_shift(
        signal_keys=['cons'],
        delay=param_time_delay,
    )
    sp.filter_ma(
        signal_keys=['cons'],
        window_length=param_ma_window_length,
    )
    sp.deconv_1_order(
        signal_keys=['cons_ma'],
        filter_gain=param_gain,
        time_cosntant=param_time_constant,
    )
    sp.remove_outliers(
        data_lims={
            'length': (100, -100),
        },
    )

    # Plotting
    if plot_results:
        plot_cf(
            sp=sp,
        ) 

    # Return processed data
    data = sp.get_data(
        keys=['poss', 'cons_ma_deconv1', 'time'],
    )
    poss = data['poss'][:, :2] # remove z
    cons = data['cons_ma_deconv1']
    time = data['time']

    down_sample = 10
    cons = cons[::down_sample]
    poss = poss[::down_sample]
    time = time[::down_sample]
    if print_stats:
        print(f"sample frequency: {cons.shape[0] / (time[-1] - time[0])}")

    return poss, cons, time



def gt_pipeline(
    file_folder: str,
    file_name: str,
    poss_min_max: Tuple[np.array, np.array],
    cons_min_max: Tuple[float, float],
):
    """
    Data loading and processing pipeline of the ground truth data.
    Args:
        file_folder (str): Folder with the file.
        file_name (str): File name.
        poss_min_max (tuple): Min and max position values.
        cons_min_max (tuple): Min and max consumption values.
    Returns:
        poss (np.array): Trajectory (N, 2)
        cons (np.array): Consumption (N,)
        cons_std (np.array): Consumption standard deviation (N,)
    """

    # Load data
    data, cols = load_data(
        csv_file = file_folder + file_name,
    )
    assert cols[0] == 'x' and cols[1] == 'y' and cols[2] == 'z' \
            and cols[3] == 'odor_mean' and cols[4] == 'odor_std'

    # Signal processing
    sp = SignalProcessing(
        data={
            'poss': data[:, :3],
            'cons': data[:, 3],
            'cons_std': data[:, 4],
        },
        print_stats=True,
    )

    # Remove unwanted data
    sp.remove_outliers(
        data_lims={
            'cons': cons_min_max,
            'poss': poss_min_max,
        },
    )

    # Remove offset
    """
    TODO: Replace this with a proper calibration blank measurement
    """
    data = sp.get_data(
        keys=['cons', 'poss'],
    )
    mask_ = data['poss'][:, 0] > 7.5
    offset = - np.mean(data['cons'][mask_])
    print(f"Offset: {offset}")
    """
    TODO: END
    """
    sp.apply_offsets(
        offsets_dict={
            'cons': offset,
        },
    )

    # Return processed data
    data = sp.get_data(
        keys=['poss', 'cons', 'cons_std'],
    )
    poss = data['poss'][:, :2] # remove z
    cons = data['cons']
    cons_std = data['cons_std']
    
    return poss, cons, cons_std


def simu_cf_pipeline(
    file_folder: str,
    file_name: str,
    keep_3D: bool = False,
    print_stats: bool = False,
):
    """
    Signal processing.
    Args:
        file_folder (str): Folder with the file.
        file_name (str): File name.
        keep_3D (bool): Keep 3D data. If False, only x and y are kept.
        print_stats (bool): Print stats
    Returns:
        poss (np.array): Trajectory (N, 2)
        cons (np.array): Consumption (N,)
        time (np.array): Time (N,)
    """

    # Load data
    data, column_names = load_data(
        csv_file=file_folder + file_name,
    )
    assert column_names[0] == 'timestamp' and column_names[1] == 'x' and column_names[2] == 'y' \
            and column_names[3] == 'z' and column_names[4] == 'odor'
    
    if keep_3D:
        poss = data[:, 1:4]
    else:
        poss = data[:, 1:3]
    cons = data[:, 4]
    time = data[:, 0]

    down_sample = 3
    cons = cons[::down_sample]
    poss = poss[::down_sample]
    time = time[::down_sample]
    if print_stats:
        print(f"sample frequency: {cons.shape[0] / (time[-1] - time[0])}")
        
    return poss, cons, time

def simu_gt_pipeline(
    file_folder: str,
    file_name: str,
    keep_3D: bool = False,
):
    """
    Signal processing.
    Args:
        file_folder (str): Folder with the file.
        file_name (str): File name.
        keep_3D (bool): Keep 3D data. If False, only x and y are kept.
    Returns:
        poss (np.array): Trajectory (N, 2)
        cons (np.array): Consumption (N,)
    """

    # Load data
    data, column_names = load_data(
        csv_file=file_folder + file_name,
    )
    assert column_names[0] == 'x' and column_names[1] == 'y' \
            and column_names[2] == 'z' and column_names[3] == 'odor'
    
    if keep_3D:
        poss = data[:, :3]
    else:
        poss = data[:, :2]
    cons = data[:, 3]
    return poss, cons