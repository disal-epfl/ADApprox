
# ADApprox: Physics-Based Gas Mapping with Nano Aerial Vehicles

This repository contains the source code for **ADApprox**, a novel Gas Distribution Mapping (GDM) algorithm grounded in the advection-diffusion equation. It provides robust interpolation capabilities using sparse gas sensor data collected by Nano Aerial Vehicles (NAVs), as described in the publication:

**“Physics-Based Gas Mapping with Nano Aerial Vehicles: The ADApprox Algorithm”**  
*Nicolaj Bösel-Schmid, Wanting Jin, Alcherio Martinoli*  
[EPFL, 2025]

## Repository Overview

The code base is structured around three main scripts:

- `run_mapping.py`: Executes the mapping pipeline to generate gas concentration maps (Fig. 2 in publication).
- `run_evaluation.py`: Evaluates mapping performance (GDM and GSL) over various experiments (Figs. 3-6 in publication).
- `run_optimization.py`: Performs hyperparameter optimization for the ADApprox and benchmark algorithms.

Each script is designed to be modular and flexible via JSON-based configuration defined in `args/`.

---

## Installation

1. **Download the dataset** from https://zenodo.org/records/17189796 and add the dataset as folder called "data" in the main directory.

2. **Open a terminal** and navigate to that directory:

   ```bash
   cd /path/to/project/repository
   ```

3. **Create the environment**: The prerequisites are defined in `environment.yaml`.

   ```bash
   # using conda
   conda env create -f environment.yaml

   # using micromamba
   micromamba create -n gdm --file environment.yaml
   ```

4. **Activate the environment**:

   ```bash
   # using conda
   conda activate gdm

   # using micromamba
   micromamba activate gdm
   ```

---

## 1. `run_mapping.py`

### Description

Runs the complete gas mapping pipeline on a **single experiment** using both:

- The ADApprox (least-square) method
- The Kernel-based mapping method

Results are visualized and saved.

### Usage

```bash
python run_mapping.py
```

This defaults to using the `args/realworld_gdm.json` configuration.

### Outputs

- Mean measurements and robot trajectory
- Gas maps (ground truth, ADApprox, Kernel DM+V/W)
- Variance maps (ground truth, ADApprox, Kernel DM+V/W)
- Learned parameters of ADApprox

---

## 2. `run_evaluation.py`

### Description

Evaluates the ADApprox and baseline Kernel DM+V/W methods over **multiple experiments** (real-world and simulated). The script includes several benchmark scenarios:

- Real-world GDM
- Simulated GDM with different wind speeds and sensor spacings
- Real-world and simulated GSL evaluation

### Usage

```bash
python run_evaluation.py
```

This will sequentially run all evaluation modes:
- `run_eval_realworld_gdm()` (Fig. 5 in publication)
- `run_eval_simu_gdm_res()` (Fig. 3 in publication)
- `run_eval_simu_gdm_wind()` (Table 1 in publication)
- `run_eval_simu_gdm_spacing()` (Table 2 in publication)
- `run_eval_real_gsl()` (Fig. 6 in publication)
- `run_eval_simu_gsl()` (Fig. 4 in publication)

### Outputs

- CSV files (e.g., `results_kernel.csv`, `results_ls_0_1.csv`) summarizing evaluation metrics
- Box plots and comparison figures for each scenario

---

## 3. `run_optimization.py`

### Description

Performs grid search to optimize hyperparameters for:

- The kernel-based baseline (Kernel DM+V/W)
- The ADApprox method (least squares precision)

Optimization is performed over both **simulation** and **real-world** datasets using shape metric as the objective.

### Usage

```bash
python run_optimization.py
```

This runs the following by default:
- `run_opt_simu_kernel()`
- `run_opt_simu_ls()`
- `run_opt_real_kernel()`
- `run_opt_real_ls()`

You can customize parameter search ranges and metrics inside the respective function calls.

### Outputs

- Grid search histogram for performance visualization
- Best-performing hyperparameter combinations printed to console


---

## Citation

If you use this code or dataset, please cite:

> Bösel-Schmid, N., Jin, W., & Martinoli, A. (2025). *Physics-Based Gas Mapping with Nano Aerial Vehicles: The ADApprox Algorithm*. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025.
