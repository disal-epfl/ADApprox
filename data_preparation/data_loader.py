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

import numpy as np


def load_data(
    csv_file,
    max_rows=None,
):
    """
    Load data from a CSV file. Assume the first line contains 
    the column names starting with a '# '.
    Args:
        csv_file (str): Path to the CSV file.
        max_rows (int): Maximum number of rows to load.
    Returns:
        data (np.ndarray): Data from the CSV file.
        column_names (list): List of column names
    """
    # Read the first line to get the column names
    with open(csv_file, 'r') as f:
        first_line = f.readline().strip()
        column_names = first_line.lstrip('# ').split(',')
    column_names = [name.strip() for name in column_names]

    # Load data
    data = np.genfromtxt(
        fname=csv_file,
        delimiter=',',
        skip_header=1,
        dtype=None,
        max_rows=max_rows,
    )

    return data, column_names

