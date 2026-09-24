# predictor.py
import torch
import torch.nn as nn
import numpy as np
import joblib
import json

class FlexibleSurrogate(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
        layers     = []
        current_in = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_in, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            current_in = hidden_size
        layers.append(nn.Linear(current_in, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_model(model_path='airfoil_surrogate.pt',
               scaler_x_path='scaler_X.pkl',
               scaler_y_path='scaler_y.pkl'):
    model = FlexibleSurrogate(4, [64, 128, 64], 3)
    model.load_state_dict(
        torch.load(model_path, map_location='cpu')
    )
    model.eval()
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_X, scaler_y


def predict(camber, position, thickness, aoa,
            model, scaler_X, scaler_y):
    # Validate inputs
    assert -4  <= aoa       <= 16, "AoA out of range"
    assert  0  <= camber    <=  9, "Camber out of range"
    assert  1  <= position  <=  9, "Position out of range"
    assert  6  <= thickness <= 24, "Thickness out of range"

    raw   = np.array([[camber, position, thickness, aoa]],
                      dtype=np.float32)
    scaled = scaler_X.transform(raw)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(tensor)

    pred_real = scaler_y.inverse_transform(
                pred_scaled.numpy())

    return {
        'Cl': round(float(pred_real[0, 0]), 4),
        'Cd': round(float(pred_real[0, 1]), 5),
        'Cm': round(float(pred_real[0, 2]), 4),
        'LD': round(float(pred_real[0, 0]) /
                    float(pred_real[0, 1]), 2)
    }