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

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from args.args import Args
from evaluation.evaluate_methods import gt, load_cf_data, meas, kernel, ls
from evaluation.optimization.grid_search import GridSearch



def opt_kernel(
    args_file: str,
    std_min_max_steps: tuple,
    inf_rad_min_max_steps: tuple,
    stretch_min_max_steps: tuple,
    metric_name: str,
    print_results: bool,
    plot_results: bool,
):
    """
    Optimize the kernel filter.
    Args:
        args_file (str): Path to the args file.
        std_min_max_steps (tuple): Min, max, and step for the std.
        inf_rad_min_max_steps (tuple): Min, max, and step for the inf_rad.
        stretch_min_max_steps (tuple): Min, max, and step for the stretch.
        metric_name (str): Name of the metric.
        print_results (bool): Print the results.
        plot_results (bool): Plot the results.
    """
    # Load args
    args = Args(
        file_path=args_file
    )

    # Load GT
    filter_gt, filter_gt_var, metrics = gt(
        args=args,
        print_results=True,
    )

    # Load CF data
    cf_poss, cf_cons, cf_time = load_cf_data(
        args=args,
        exp=args(["data", "default_exp"]),
        print_results=True,
    )

    # Mean measurements
    filter_mm = meas(
        args=args,
        cf_poss=cf_poss,
        cf_cons=cf_cons,
        print_results=True,
    )

    # Optimization
    grid_search = GridSearch(
        mapping_fct=kernel,
        map_gt=filter_gt.map_c,
        metric_name=metric_name,
        larger_is_better=True,
    )

    args_kernel = {
        'args': args,
        'cf_poss': cf_poss,
        'cf_cons': cf_cons,
        'metrics': metrics,
        'filter_mm': filter_mm,
        'print_results': print_results,
    }
    grid_search(
        params_min=[std_min_max_steps[0], inf_rad_min_max_steps[0], stretch_min_max_steps[0]],
        params_max=[std_min_max_steps[1], inf_rad_min_max_steps[1], stretch_min_max_steps[1]],
        params_step=[std_min_max_steps[2], inf_rad_min_max_steps[2], stretch_min_max_steps[2]],
        params_name=[['kernel', 'std'], ['kernel', 'inf_rad'], ['kernel', 'stretch']],
        args_mapping_fcts=args_kernel,
        print_results=print_results,
        plot_results=plot_results,
    )

def opt_ls(
    args_file: str,
    precision_min_max_steps: tuple,
    metric_name: str,
    print_results: bool,
    plot_results: bool,
):
    """
    Optimize the least square filter.
    Args:
        args_file (str): Path to the args file.
        precision_min_max_steps (tuple): Min, max, and step for the precision.
        metric_name (str): Name of the metric.
        print_results (bool): Print the results.
        plot_results (bool): Plot the results.
    """
    # Load args
    args = Args(
        file_path=args_file
    )

    # Load GT
    filter_gt, filter_gt_var, metrics = gt(
        args=args,
        print_results=True,
    )

    # Load CF data
    cf_poss, cf_cons, cf_time = load_cf_data(
        args=args,
        exp=args(["data", "default_exp"]),
        print_results=True,
    )

    # Optimization
    grid_search = GridSearch(
        mapping_fct=ls,
        map_gt=filter_gt.map_c,
        metric_name=metric_name,
        larger_is_better=True,
    )

    args_ls = {
        'args': args,
        'cf_poss': cf_poss,
        'cf_cons': cf_cons,
        'metrics': metrics,
        'print_results': print_results,
    }
    grid_search(
        params_min=[precision_min_max_steps[0]],
        params_max=[precision_min_max_steps[1]],
        params_step=[precision_min_max_steps[2]],
        params_name=[['ls', 'precision']],
        args_mapping_fcts=args_ls,
        print_results=print_results,
        plot_results=plot_results,
    )

def run_opt_simu_kernel():
    """
    Run the kernel optimization for the simulation.
    """
    opt_kernel(
        args_file="args/simulation_gdm.json",
        std_min_max_steps=(0.004, 0.006, 3),
        inf_rad_min_max_steps=(0.9, 1.1, 3),
        stretch_min_max_steps=(0.6, 0.8, 3),
        metric_name="shape",
        print_results=True,
        plot_results=True,
    )

def run_opt_simu_ls():
    """
    Run the least square optimization for the simulation.
    """
    opt_ls(
        args_file="args/simulation_gdm.json",
        precision_min_max_steps=(20, 40, 3),
        metric_name="shape",
        print_results=True,
        plot_results=True,
   )
    
def run_opt_real_kernel():
    """
    Run the kernel optimization for the real-world data.
    """
    opt_kernel(
        args_file="args/realworld_gdm.json",
        std_min_max_steps=(0.003, 0.005, 3),
        inf_rad_min_max_steps=(0.7, 0.9, 3),
        stretch_min_max_steps=(0.7, 0.9, 3),
        metric_name="shape",
        print_results=True,
        plot_results=True,
    )

def run_opt_real_ls():
    """
    Run the least square optimization for the real-world data.
    """
    opt_ls(
        args_file="args/realworld_gdm.json",
        precision_min_max_steps=(20, 40, 3),
        metric_name="shape",
        print_results=True,
        plot_results=True,
   )
    

if __name__ == "__main__":
    run_opt_simu_kernel()
    run_opt_simu_ls()
    run_opt_real_kernel()
    run_opt_real_ls()