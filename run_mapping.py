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
from evaluation.plotting_maps import plot


def run_mapping(
    args_file: str,
):
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

    # Kernel map
    filter_km, results_km = kernel(
        args=args,
        cf_poss=cf_poss,
        cf_cons=cf_cons,
        metrics=metrics,
        filter_mm=filter_mm,
        print_results=True,
    )

    # Least square
    simu, map_ls, map_ls_var, results_ls = ls(
        args=args,
        cf_poss=cf_poss,
        cf_cons=cf_cons,
        metrics=metrics,
        print_results=True,
    )

    # Plot
    plot(
        args=args,
        filter_gt=filter_gt,
        filter_gt_var=filter_gt_var,
        filter_mm=filter_mm,
        filter_km=filter_km,
        simu=simu,
        map_ls=map_ls,
        map_ls_var=map_ls_var,
        metrics=metrics,
        cf_poss=cf_poss,
        exp=args(["data", "default_exp"]),
        print_results=True,
    )


def run_mapping_realworld():
    """
    Run the mapping pipeline on real-world data.
    """
    args_file = "args/realworld_gdm.json"
    run_mapping(
        args_file=args_file
    )



if __name__ == "__main__":
    run_mapping_realworld()
    