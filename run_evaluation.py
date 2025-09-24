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
from evaluation.plotting_boxplots import plot_gdm, plot_simu_wind, plot_simu_spacing, plot_gsl
from evaluation.metrics.metrics import Metrics



def save_results(
    results: dict,
    results_file: str,
    exp: str,
    write_header: bool = False,
    print_results: bool = False,
):
    """
    Save results to a CSV file.
    Args:
        results (dict): The results to be saved.
        results_file (str): The path to the CSV file.
        exp (str): The experiment name.
        write_header (bool): If True, the header will be written to the file.
        print_results (bool): If True, the results will be printed.
    """
    if write_header:
        with open(results_file, 'w') as f:
            first_line = "# experiment"
            for m in results.keys():
                first_line += f", {m}"
            f.write(first_line + '\n')
    
    with open(results_file, 'a') as f:
        line = exp
        for m in results.keys():
            line += f", {results[m]}"
        f.write(line + '\n')

    if print_results:
        print(f"\nResults: {exp}, {results}")


def eval_kernel(
    args: Args,
    gdm_not_gsl: bool = True,
    print_results: bool = False,
):
    """
    Evaluate the kernel mapping.
    Args:
        args (Args): The arguments.
        gdm_not_gsl (bool): If True, the evaluation is for GDM, otherwise for GSL.
        print_results (bool): If True, the results will be printed.
    """
    # Results file
    results_file = args(["data", "data_folder"]) + args(["data", "folder_cf"]) + "results_kernel.csv"
    
    # Load GT
    if gdm_not_gsl:
        filter_gt, filter_gt_var, metrics = gt(
            args=args,
            print_results=print_results,
        )
    else:
        metrics = Metrics()

    for i, exp in enumerate(args(["data", "all_exp"])):
        if print_results:
            print(f"Experiment: {exp}")

        # Set source position
        if not gdm_not_gsl:
            metrics.source_pos = args(["eval", "source_pos"])[i]

        # Load CF data
        cf_poss, cf_cons, cf_time = load_cf_data(
            args=args,
            exp=exp,
            print_results=print_results,
        )

        # Mean measurements
        filter_mm = meas(
            args=args,
            cf_poss=cf_poss,
            cf_cons=cf_cons,
            print_results=print_results,
        )

        # Kernel map
        filter_km, results_km = kernel(
            args=args,
            cf_poss=cf_poss,
            cf_cons=cf_cons,
            metrics=metrics,
            filter_mm=filter_mm,
            print_results=print_results,
        )

        save_results(
            results=results_km,
            results_file=results_file,
            exp=exp,
            write_header=(i == 0),
            print_results=True, #print_results,
        )

def eval_ls(
    args: Args,
    gdm_not_gsl: bool = True,
    print_results: bool = False,
):
    """
    Evaluate the least square mapping.
    Args:
        args (Args): The arguments.
        gdm_not_gsl (bool): If True, the evaluation is for GDM, otherwise for GSL.
        print_results (bool): If True, the results will be printed.
    """
    # Results file
    results_file = args(["data", "data_folder"]) + args(["data", "folder_cf"]) \
                    + "results_ls_" + str(args(["ls", "res"])).replace('.', '_') + ".csv"
    
    # Load GT
    if gdm_not_gsl:
        filter_gt, filter_gt_var, metrics = gt(
            args=args,
            print_results=print_results,
        )
    else:
        metrics = Metrics()

    for i, exp in enumerate(args(["data", "all_exp"])):
        if print_results:
            print(f"Experiment: {exp}")

        # Set source position
        if not gdm_not_gsl:
            metrics.source_pos = args(["eval", "source_pos"])[i]

        # Load CF data
        cf_poss, cf_cons, cf_time = load_cf_data(
            args=args,
            exp=exp,
            print_results=print_results,
        )

        # Mean measurements
        filter_mm = meas(
            args=args,
            cf_poss=cf_poss,
            cf_cons=cf_cons,
            print_results=print_results,
        )

        # Kernel map
        simu, map_ls, map_ls_var, results_ls = ls(
            args=args,
            cf_poss=cf_poss,
            cf_cons=cf_cons,
            metrics=metrics,
            print_results=print_results,
        )

        save_results(
            results=results_ls,
            results_file=results_file,
            exp=exp,
            write_header=(i == 0),
            print_results=print_results,
        )

def run_eval_realworld_gdm():
    """
    Run the GDM evaluation for the real-world data.
    """
    file_path = "args/realworld_gdm.json"
    args = Args(
        file_path=file_path,
    )

    eval_kernel(
        args=args,
        print_results=True,
    )

    for res in [0.1, 0.2, 0.4]:
        args.set(
            name=["ls", "res"],
            value=res,
        )
        eval_ls(
            args=args,
            print_results=True,
        )

    plot_gdm(
        experiment_folder = 'data/realworld/2025_02_05/',
    )

def run_eval_simu_gdm_res():
    """
    Run the GDM evaluation for the simulation data.
    """
    file_path = "args/simulation_gdm.json"
    args = Args(
        file_path=file_path,
    )

    eval_kernel(
        args=args,
        print_results=True,
    )

    for res in [0.1, 0.2, 0.4]:
        args.set(
            name=["ls", "res"],
            value=res,
        )
        eval_ls(
            args=args,
            print_results=True,
        )

    experiment_folder = 'data/simulation/wind_0_7/rect_1/'
    plot_gdm(
        experiment_folder=experiment_folder,
    )

def run_eval_simu_gdm_wind():
    """
    Run the GDM evaluation for the simulation data.
    """
    file_path = "args/simulation_gdm.json"
    args = Args(
        file_path=file_path,
    )

    windspeeds = [-0.3, -0.7, -1.3]
    windspeed_names = ["wind_0_3", "wind_0_7", "wind_1_3"]
    for w, w_name in zip(windspeeds, windspeed_names):
        args.set(
            name=["envi", "wind_speed"],
            value=w,
        )
        args.set(
            name=["data", "folder_cf"],
            value="simulation/" + w_name + "/rect_1/",
        )
        args.set(
            name=["data", "file_gt"],
            value="simulation/" + w_name + "/gt/gt.csv",
        )

        eval_kernel(
            args=args,
            print_results=True,
        )
        eval_ls(
            args=args,
            print_results=True,
        )

    plot_simu_wind(
        experiment_folder=args(["data", "data_folder"]) + "simulation/",
    )

def run_eval_simu_gdm_spacing():
    """
    Run the GDM evaluation for the simulation data with different spacings.
    """
    file_path = "args/simulation_gdm.json"
    args = Args(
        file_path=file_path,
    )

    spacing_names = ["rect_1", "rect_1_5", "rect_2"]  
    for s_name in spacing_names:
        args.set(
            name=["data", "folder_cf"],
            value="simulation/wind_0_7/" + s_name + "/",
        )

        eval_kernel(
            args=args,
            print_results=True,
        )
        eval_ls(
            args=args,
            print_results=True,
        )

    plot_simu_spacing(
        experiment_folder=args(["data", "data_folder"]) + "simulation/",
    )

def run_eval_real_gsl():
    """
    Run the GSL evaluation for the real-world and simulated data.
    """
    args = Args(
        file_path="args/realworld_gsl.json",
    )

    eval_kernel(
        args=args,
        gdm_not_gsl=False,
        print_results=True,
    )

    for res in [0.1, 0.2, 0.4]:
        args.set(
            name=["ls", "res"],
            value=res,
        )
        eval_ls(
            args=args,
            gdm_not_gsl=False,
            print_results=True,
        )

    plot_gsl(
        experiment_folder=args(["data", "data_folder"]) + args(["data", "folder_cf"]),
    )

def run_eval_simu_gsl():
    """
    Run the GSL evaluation for the real-world and simulated data.
    """
    file_path = "args/simulation_gsl.json"
    args = Args(
        file_path=file_path,
    )

    eval_kernel(
        args=args,
        gdm_not_gsl=False,
        print_results=True,
    )

    for res in [0.1, 0.2, 0.4]:
        args.set(
            name=["ls", "res"],
            value=res,
        )
        eval_ls(
            args=args,
            gdm_not_gsl=False,
            print_results=True,
        )

    plot_gsl(
        experiment_folder=args(["data", "data_folder"]) + args(["data", "folder_cf"]),
    )



if __name__ == "__main__":
    run_eval_realworld_gdm()
    run_eval_simu_gdm_res()
    run_eval_simu_gdm_wind()
    run_eval_simu_gdm_spacing()
    run_eval_real_gsl()
    run_eval_simu_gsl()