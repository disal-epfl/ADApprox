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
from itertools import product

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from mapping.map import Map
from evaluation.metrics.metrics import Metrics


class GridSearch():
    def __init__(
        self,
        mapping_fct: callable,
        map_gt: Map,
        metric_name:str,
        larger_is_better: bool = False,
    ):

        # Evaluation
        self.metrics = Metrics(
            map_gt=map_gt,
        )
        self.metric_name = metric_name
        self.mapping_fct = mapping_fct
        self.larger_is_better = larger_is_better

    def __call__(
        self,
        params_min: np.ndarray,
        params_max: np.ndarray,
        params_step: np.ndarray,
        params_name: list,
        args_mapping_fcts: dict={},
        print_results:bool=False,
        plot_results:bool=False,
    ):
        """
        Search for the optimal parameter.
        Args:
            params_min (np.ndarray): Minimum parameter value.
            params_max (np.ndarray): Maximum parameter value.
            params_step (np.ndarray): Number of steps per parameter.
            params_name (list): Name of the parameters to access them in args.
            args_mapping_fcts (dict): Additional arguments for the mapping function.
            print_results (bool): Print the results.
            plot_results (bool): Plot the results.
        Returns:
            (np.ndarray): Optimal parameter.
        """
        params = self.generate_param_permutations(
            params_min=params_min,
            params_max=params_max,
            params_step=params_step,
        )

        results = []
        for i, param in enumerate(params):
            if print_results:
                print(f"{i+1}/{len(params)}: Parameter: {param}")

            for j, p_name in enumerate(params_name):
                args_mapping_fcts['args'].set(
                    name=p_name,
                    value=param[j],
                )

            r = self.mapping_fct(
                **args_mapping_fcts,
            )[-1]
            results.append(r[self.metric_name])

        if print_results:
            for param, result in zip(params, results):
                print(f"Parameter: {param}, \t\t{self.metric_name} = {result:.4f}")
            
            if self.larger_is_better:
                print(f"Best parameter value: {params[np.argmax(results)]}")
            else:
                print(f"Best parameter value: {params[np.argmin(results)]}")

        if plot_results:
            self.plot_results(
                results=np.array(results),
                params=params,
            )

        return params[np.argmin(results)]
    
    def generate_param_permutations(
        self,
        params_min: np.ndarray,
        params_max: np.ndarray,
        params_step: np.ndarray,
    ):
        """
        Generate all permutations of parameter values.
        Args:
            params_min (np.ndarray): Minimum parameter value.
            params_max (np.ndarray): Maximum parameter value.
            params_step (np.ndarray): Number of steps per parameter.
        Returns:
            (np.ndarray): Permutations of parameter values.
        """
        if not (len(params_min) == len(params_max) == len(params_step)):
            raise ValueError("All input arrays must have the same length.")
        
        param_values = [
            np.linspace(start, stop, num=int(steps)) 
            for start, stop, steps in zip(params_min, params_max, params_step)
        ]
        
        return np.array(list(product(*param_values)))
    
    def plot_results(
        self,
        results: np.ndarray,
        params: np.ndarray,
    ):
        """
        Plot the results.
        Args:
            results (np.ndarray): Results.
            params (np.ndarray): Parameters.
        """
        fig, axes = plt.subplots(nrows=params.shape[1], ncols=1, figsize=(20, 20))
        axes = np.array([axes]).flatten()
        for i, ax in enumerate(axes):
            
            mean_results = []
            params_unique = np.unique(params[:,i])
            for param in params_unique:
                mean_results.append(np.mean(results[params[:,i]==param]))

            if self.larger_is_better:
                best_value = params[np.argmax(results), i]
            else:
                best_value = params[np.argmin(results), i]

            ax.bar(params_unique, mean_results, width=(np.max(params_unique)-np.min(params_unique))/(2*len(params_unique)), color='b')
            ax.set_xlabel(f"Parameter {i}")
            ax.set_ylabel(self.metric_name)
            ax.set_title(f"Best value: {best_value}")
            ax.set_ylim([np.min(results), np.max(results)])

        plt.show()


        

        
