import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import torch

def load_and_prepare_dataset(filepath,
                             target_cols=['Cl','Cd','Cm'],
                             save_scalar=True):
     """
    Complete data preparation pipeline for airfoil surrogate ML training.
    
    Takes raw Fluent CSV output and returns scaled train/test arrays
    ready to feed directly into PyTorch.
    
    Parameters:
        filepath    : path to the raw Fluent CSV
        target_cols : list of output columns to predict
        save_scaler : whether to save the scaler for use in Streamlit app
    
    Returns:
        X_train, X_test, y_train, y_test : numpy arrays, scaled
        feature_names, target_names      : column name lists
    """
     
    #  ----<step 1:load>--------
     df=pd.read_csv(filepath)
     print(f"loaded: {df.shape[0]} rows, {df.shape[1]} columns ")
    # -----<step-2: clean>------
     df=df.dropna(subset=target_cols)
     df=df[(df['Cl']>-0.5)&(df['Cd']<2.0)]
     df=df[(df['Cd']>0.003)&(df['Cd']<0.15)]
     print(f"after cleaning: {df.shape[0]} rows")

    #  ----<step 3:define features and targets>-----
     feature_cols=['camber_pct', 'camber_pos', 'thickness_pct', 'AoA_deg']
     X = df[feature_cols].values   # numpy array, shape (n, 4)
     Y = df[target_cols].values    # numpy array, shape (n, len(targets))

    #  -----<step 4:train/test split>--------
     indices=np.random.permutation(len(X)) # shuffle row numbers
     split_idx = int(0.8 * len(X))              # 80% cutoff point

     train_idx = indices[:split_idx]   # first 80%
     test_idx  = indices[split_idx:]   # last 20%

     X_train, X_test = X[train_idx], X[test_idx]
     y_train, y_test = Y[train_idx], Y[test_idx]

     print(f"train samples: {len(X_train)} |test samples: {len(X_test)}")

    #  -----<Scale features>-------
     scalar_X=StandardScaler()
     scalar_Y=StandardScaler()

     X_train=scalar_X.fit_transform(X_train)
     X_test=scalar_X.transform(X_test)

     y_train = scalar_Y.fit_transform(y_train)
     y_test  = scalar_Y.transform(y_test)

     if save_scalar:
          joblib.dump(scalar_X, 'scalar_X.pkl')
          joblib.dump(scalar_Y,'scaler_Y.pkl')
          print("scalers saved.")
     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
     X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
     y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
     y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)    

     return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, feature_cols,target_cols


# ------>test run<------
X_train, X_test, y_train, y_test, feat, targ = load_and_prepare_dataset(
    'fluent_dataset_clean.csv',
    target_cols=['Cl', 'Cd', 'Cm']
)

print(f"\nX_train shape: {X_train.shape}")  # (n_train, 4)
print(f"y_train shape: {y_train.shape}")  # (n_train, 3)
print(f"\nFeatures : {feat}")
print(f"Targets  : {targ}")

# Run this — confirm output clean
# X_train, X_test, y_train, y_test, feat, targ = load_and_prepare_dataset(
#     'fluent_dataset_clean.csv',
#     target_cols=['Cl', 'Cd', 'Cm']
# )
# print(X_train.shape)  # should print (n, 4)
# print(y_train.shape)  # should print (n, 3)