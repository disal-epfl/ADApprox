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

import json
import os
from typing import List, Union


class Args:
    def __init__(
        self, 
        file_path: str,
    ):
        """
        Initializes the Args instance by loading arguments from the given JSON file.
        Args:
            file_path: The path to the JSON file containing the arguments.
        """
        self.file_path = file_path
        self.args = {}
        self._load_arguments()

    def __call__(
        self,
        name: Union[str, List[str]],
    ):
        """
        Gets the value of an argument.
        Args:
            name (str | List[str]): The argument name. If a list of strings is provided,
                                    the argument will be retrieved from self.args[name[0]][name[1]]...
        Returns:
            The value of the argument.
        """
        if isinstance(name, str):
            return self.args.get(name, None)
        
        if isinstance(name, list):
            current = self.args
            for n in name:
                if n not in current:
                    return None
                current = current[n]
            return current
        
        raise ValueError("Name must be a string or a list of strings.")
    
    def set(
        self, 
        name: Union[str, List[str]],
        value,
    ):
        """
        Sets an argument to a specific domain.
        Args:
            name (str | List[str]): The argument name. If a list of strings is provided,
                                    the argument will be added to self.args[name[0]][name[1]]...
            value: The argument value
        """
        current = self.args
        for n in name[:-1]:
            current = current[n]
        current[name[-1]] = value

    def add(
        self, 
        name: Union[str, List[str]],
        value,
    ):
        """
        Adds an argument to a specific domain. If the domain doesn't exist, it will be created.
        Args:
            name (str | List[str]): The argument name. If a list of strings is provided,
                                    the argument will be added to self.args[name[0]][name[1]]...
            value: The argument value
        """
        current = self.args
        for n in name[:-1]:
            if n not in current:
                current[n] = {}
            current = current[n]
        current[name[-1]] = value

    def save(
        self, 
        folder: str,
        file_name: str = None,
    ):
        """
        Saves the current arguments to a JSON file in the specified folder.
        Args:
            folder (str): The folder where the JSON file will be saved.
            file_name (str): The name of the JSON file. If None, the name 
                            of the original file will be used.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        if file_name is None:
            file_name = os.path.basename(self.file_path)

        output_path = os.path.join(folder, file_name)
        try:
            with open(output_path, "w") as f:
                json.dump(self.args, f, indent=4)
            print(f"Args:save: Arguments saved to {output_path}")
        except IOError as e:
            print(f"Args:save: Failed to save file: {e}")

    def _load_arguments(
        self,
    ):
        """
        Loads the arguments from the JSON file.
        """
        try:
            with open(self.file_path, "r") as f:
                self.args = json.load(f)
        except FileNotFoundError:
            print(f"Args:save: File {self.file_path} not found. Starting with empty arguments.")
        except json.JSONDecodeError as e:
            print(f"Args:save: Error decoding JSON: {e}. Starting with empty arguments.")