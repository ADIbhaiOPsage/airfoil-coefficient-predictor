# Airfoil Aerodynamic Coefficient Predictor & Generation Pipeline

An end-to-end framework for generating aerodynamic data via XFOIL, preprocessing features, and training a PyTorch-based neural network surrogate model to rapidly predict airfoil lift, drag, and moment coefficients as a fast alternative to iterative CFD simulations.

---

## Project Structure
* `XFOIL.py` & `XFOIL_AUTOGEN.py`: Automated scripts for interfacing with XFOIL to generate aerodynamic datasets.
* `Xfoil699src.zip`: Source code archive for XFOIL v6.99.
* `NN_pipeline_func.py`: Main execution pipeline handling data ingestion, preprocessing, scaling, training, and evaluation.
* `NN_pytorch.py`: Defines the PyTorch neural network architecture, forward pass logic, and training loop.
* `Pandas_101.py` & `numpy_101.py`: Helper scripts for data manipulation and feature engineering.
* `OOPs_PY.py`: Object-oriented implementations for modularizing the data pipeline.
* `requirements.txt`: Python package dependencies.

---

## Prerequisites & IDE Setup
* **Python 3.10+**
* **Fortran Compiler (`gfortran`) and `make`**: Required for compiling XFOIL from source.
* **Recommended VS Code Extensions for Fortran:**
  * **Modern Fortran** (`fortran-lang.linter-gfortran`): Provides syntax highlighting, code snippets, and linting.
  * **Fortran Language Server (`fortls`)**: Adds advanced IDE features like *Go-to-Definition* and autocomplete (Install via `pip install fortls`).
  * **Code Runner** (`formulahendry.code-runner`): For quickly executing scripts.

---

## Step-by-Step Setup & Installation Guide

### Step 1: Clone the Repository
```bash
git clone [https://github.com/ADIbhaiOPsage/airfoil-coefficient-predictor.git](https://github.com/ADIbhaiOPsage/airfoil-coefficient-predictor.git)
cd airfoil-coefficient-predictor
