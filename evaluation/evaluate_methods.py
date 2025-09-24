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
import time
import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from args.args import Args
from mapping.kernelDMV import KernelDMV
from mapping.mean_meas import MeanMeas
from mapping.least_square_variance import LeastSquareVariance
from evaluation.metrics.metrics import Metrics
from mapping.map import Map
from data_preparation.data_pipline import simu_cf_pipeline, simu_gt_pipeline, cf_pipeline, gt_pipeline



def convert_dims(
    args:Args,
    min_not_max:bool = False,
):
    """
    Convert the dimensions of the map to the correct format
    """
    if args(["envi", "n_dims"]) == 2:
        if min_not_max:
            return [args(["envi", "poss_min_max"])[0][0], args(["envi", "poss_min_max"])[0][1]]
        else:
            return [args(["envi", "poss_min_max"])[1][0], args(["envi", "poss_min_max"])[1][1]]
        
    if args(["envi", "n_dims"]) == 3:
        if min_not_max:
            return args(["envi", "poss_min_max"])[0]
        else:
            return args(["envi", "poss_min_max"])[1]
        
    raise ValueError("The number of dimensions is not supported")

def gt(
    args:Args,
    print_results:bool = False,
):
    if print_results:
        print("Ground truth pipeline")

    # Load ground truth data
    if args(["data", "real_not_simu"]):
        gt_poss, gt_cons, gt_cons_std = gt_pipeline(
            file_folder = args(["data", "data_folder"]),
            file_name = args(["data", "file_gt"]),
            poss_min_max = np.array(args(["envi", "poss_min_max"])),
            cons_min_max = (args(["envi", "cons_min_max"])[0], args(["envi", "cons_min_max"])[1]),
        )
    else:
        gt_poss, gt_cons = simu_gt_pipeline(
            file_folder = args(["data", "data_folder"]),
            file_name = args(["data", "file_gt"]),
            keep_3D = (args(["envi", "n_dims"]) == 3),
        )
        gt_cons_std = None

    # Ground truth filter
    filter_gt = MeanMeas(
        n_dims = args(["envi", "n_dims"]),
        res = args(["envi", "res"]),
        dims_min = convert_dims(args, min_not_max=True),
        dims_max = convert_dims(args, min_not_max=False),
        cons_init_value = args(["envi", "cons_init_value"]),
    )
    filter_gt.update(
        poss=gt_poss,
        cons=gt_cons,
    )
    filter_gt.map_c.clip(
        min_max=[0.0, np.inf],
    )

    # Evaluation
    metrics = Metrics(
        map_gt=filter_gt.map_c,
        source_pos = args(["eval", "source_pos"]),
    )

    # Variance map
    filter_gt_var = None
    if gt_cons_std is not None:
        filter_gt_var = MeanMeas(
            n_dims = args(["envi", "n_dims"]),
            res = args(["envi", "res"]),
            dims_min = convert_dims(args, min_not_max=True),
            dims_max = convert_dims(args, min_not_max=False),
            cons_init_value = args(["envi", "cons_init_value"]),
        )
        filter_gt_var.update(
            poss=gt_poss,
            cons=gt_cons_std,
        )

    return filter_gt, filter_gt_var, metrics

def load_cf_data(
    args:Args,
    exp: str,
    print_results:bool = False,
    plot_results:bool = False,
):
    if print_results:
        print("\nLoad CF data")

    if args(["data", "real_not_simu"]):
        cf_poss, cf_cons, cf_time = cf_pipeline(
            file_folder = args(["data", "data_folder"]),
            file_name = args(["data", "folder_cf"]) + exp + "scan_1.csv",
            poss_min_max = np.array(args(["envi", "poss_min_max"])),
            cons_min_max = (args(["envi", "cons_min_max"])[0], args(["envi", "cons_min_max"])[1]),
            plot_results=plot_results,
            print_stats=print_results,
        )
    else:
        cf_poss, cf_cons, cf_time = simu_cf_pipeline(
            file_folder = args(["data", "data_folder"]),
            file_name = args(["data", "folder_cf"]) + exp + "scan_1.csv",
            keep_3D = (args(["envi", "n_dims"]) == 3),
            print_stats=print_results,
        )

    return cf_poss, cf_cons, cf_time

def meas(
    args:Args,
    cf_poss:np.ndarray,
    cf_cons:np.ndarray,
    print_results:bool = False,
):
    
    if print_results:
        print("\nMeasurments")

    filter_mm = MeanMeas(
        n_dims = args(["envi", "n_dims"]),
        res = args(["envi", "res"]),
        dims_min = convert_dims(args, min_not_max=True),
        dims_max = convert_dims(args, min_not_max=False),
        cons_init_value = args(["envi", "cons_init_value"]),
    )
    filter_mm.update(
        poss=cf_poss,
        cons=cf_cons,
    )
    filter_mm.map_c.clip(
        min_max=[0.0, np.inf],
    )

    return filter_mm

def kernel(
    args:Args,
    cf_poss:np.ndarray,
    cf_cons:np.ndarray,
    metrics: Metrics,
    filter_mm: MeanMeas,
    print_results:bool = False,
):
    if print_results:
        print("\nFilter: Kernel DMV")

    # Wind map
    if args(["envi", "n_dims"]) == 2:
        init_vals = np.array([args(["envi", "wind_speed"]), 0.0])
    else:
        init_vals = np.array([args(["envi", "wind_speed"]), 0.0, 0.0])
    map_w = Map(
        n_dims = args(["envi", "n_dims"]),
        res = args(["envi", "res"]),
        dims_min = convert_dims(args, min_not_max=True),
        dims_max = convert_dims(args, min_not_max=False),
        init_vals = init_vals,
    )

    # Kernel DMV filter
    time_start = time.time()
    filter_km = KernelDMV(
        n_dims = args(["envi", "n_dims"]),
        res = args(["envi", "res"]),
        dims_min = convert_dims(args, min_not_max=True),
        dims_max = convert_dims(args, min_not_max=False),
        cons_min = np.min(cf_cons),
        cons_max = np.max(cf_cons), # TODO: remove
        cons_init_value = args(["envi", "cons_init_value"]),
        kernel_std = args(["kernel", "std"]),
        kernel_infl_radius = args(["kernel", "infl_radius"]),
        kernel_stretching = args(["kernel", "stretching"]),
        kernel_weight= 1 / np.sqrt(2*np.pi*args(["kernel", "std"])**2),
        map_w=map_w,
    )
    filter_km.update(
        poss=cf_poss,
        cons=cf_cons,
    )
    filter_km.map_c.clip(
        min_max=[0.0, np.inf],
    )
    filter_km.map_c.normalize(
        min_max=[0.0, filter_mm.map_c.get_vals_max().item()],
    )
    filter_km.map_var.clip(
        min_max=[0.0, np.inf],
    )
    filter_km.map_var.normalize(
        min_max=[0.0, filter_mm.map_c.get_vals_max().item()**2],
    )
    time_end = time.time()

    # Evaluation
    results_km = metrics(
        map_c=filter_km.map_c,
        metric_names = args(["eval", "metric_names"]),
        map_mask=filter_km.map_c,
        feature_mask=0,
    )
    results_km_var = metrics( # TODO: integrate into results_km
        map_c=filter_km.map_var,
        metric_names=['gsl_max'],
        map_mask=filter_km.map_c,
        feature_mask=0,
    )
    results_km['time'] = time_end - time_start
    results_km['gsl_var'] = results_km_var['gsl_max']

    return filter_km, results_km

def ls(
    args:Args,
    cf_poss:np.ndarray,
    cf_cons:np.ndarray,
    metrics: Metrics,
    print_results:bool = False,
):
    if print_results:
        print("\nLeast Square")

    # Wind map
    if args(["envi", "n_dims"]) == 2:
        init_vals = np.array([args(["envi", "wind_speed"]), 0.0])
    else:
        init_vals = np.array([args(["envi", "wind_speed"]), 0.0, 0.0])
    map_w = Map(
        n_dims = args(["envi", "n_dims"]),
        res = args(["ls", "res"]),
        dims_min = convert_dims(args, min_not_max=True),
        dims_max = convert_dims(args, min_not_max=False),
        init_vals = init_vals,
    )

    # Least square simulation
    time_start = time.time()
    simu = LeastSquareVariance(
        map_w=map_w,
        kernel_precision = args(["ls", "precision"]),
        print_info=True,
    )
    simu(
        poss_meas=cf_poss,
        cons_meas=cf_cons,
    )

    # Prediction map
    map_ls = Map(
        n_dims = args(["envi", "n_dims"]),
        res = args(["envi", "res"]),
        dims_min = convert_dims(args, min_not_max=True),
        dims_max = convert_dims(args, min_not_max=False),
    )
    poss_pred = map_ls.get_cell_poss(
        idxs_flat=np.arange(map_ls.num_cells),
    )
    cons_pred, cons_var = simu.predict(
        poss_meas=poss_pred,
    )
    map_ls.set_cell_vals(
        vals=cons_pred,
        poss=poss_pred,
    )
    map_ls.clip(
        min_max=[0.0, np.inf],
    )

    map_ls_var = Map(
        n_dims = args(["envi", "n_dims"]),
        res = args(["envi", "res"]),
        dims_min = convert_dims(args, min_not_max=True),
        dims_max = convert_dims(args, min_not_max=False),
    )
    map_ls_var.set_cell_vals(
        vals=cons_var,
        poss=poss_pred,
    )
    
    time_end = time.time()

    # Evaluation
    results_ls = metrics(
        map_c=map_ls,
        metric_names = args(["eval", "metric_names"]),
        map_mask=map_ls,
        feature_mask=0,
    )
    results_ls_wind = metrics( # TODO: integrate into results_ls
        map_c=simu.map_params,
        feature=2,
        map_mask=map_ls,
        feature_mask=0,
        metric_names=['gsl_max'],
    )
    results_ls_var = metrics(
        map_c=map_ls_var,
        metric_names=['gsl_max'],
        map_mask=map_ls,
    )
    results_ls['time'] = time_end - time_start
    results_ls['gsl_wind'] = results_ls_wind['gsl_max']
    results_ls['gsl_var'] = results_ls_var['gsl_max']

    return simu, map_ls, map_ls_var, results_ls

