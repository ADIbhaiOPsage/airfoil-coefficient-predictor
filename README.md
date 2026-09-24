# Airfoil Aerodynamic Coefficient Predictor

A PyTorch-based neural network surrogate model designed to predict airfoil aerodynamic coefficients (lift, drag, and moment) directly from geometric parameters and angle of attack. This model serves as a rapid alternative to iterative fluid dynamics simulations.

## Project Structure
* `NN_pipeline_func.py`: Main execution pipeline handling data ingestion, preprocessing, scaling, and parity plot generation.
* `NN_pytorch.py`: PyTorch neural network architecture definition, forward pass logic, and training loop.
* `Pandas_101.py` & `numpy_101.py`: Data manipulation scripts for feature extraction and structural formatting.
* `OOPs_PY.py`: Object-oriented implementations for modularizing the data pipeline.

## Setup & Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/ADIbhaiOPsage/airfoil-coefficient-predictor.git](https://github.com/ADIbhaiOPsage/airfoil-coefficient-predictor.git)
cd airfoil-coefficient-predictor
pip install -r requirements.txt
