import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

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
     df=df[(df['Cl']>-0.5)&(df['Cl']<2.0)]
     df=df[(df['Cd']>0.003)&(df['Cd']<0.15)]
     print(f"after cleaning: {df.shape[0]} rows")

     df['Cd']=np.log10(df['Cd']) # Forcing the network to learn exponential drag physics

    #  ----<step 3:define features and targets>-----
     feature_cols=['camber_pct', 'camber_pos', 'thickness_pct', 'AoA_deg']
     X = df[feature_cols].values   # numpy array, shape (n, 4)
     Y = df[target_cols].values    # numpy array, shape (n, len(targets))

    #  -----<step 4:train/test split>--------
     np.random.seed(42)
     indices=np.random.permutation(len(X)) # shuffle row numbers
     split_idx = int(0.8 * len(X))              # 80% cutoff point

     train_idx = indices[:split_idx]   # first 80%
     test_idx  = indices[split_idx:]   # last 20%

     X_train, X_test = X[train_idx], X[test_idx]
     y_train, y_test = Y[train_idx], Y[test_idx]

     print(f"train samples: {len(X_train)} |test samples: {len(X_test)}")

    #  -----<Scale features>-------
     scaler_X=StandardScaler()
     scaler_Y=StandardScaler()

     X_train=scaler_X.fit_transform(X_train)
     X_test=scaler_X.transform(X_test)

     y_train = scaler_Y.fit_transform(y_train)
     y_test  = scaler_Y.transform(y_test)

     if save_scalar:
          joblib.dump(scaler_X, 'scalar_X.pkl')
          joblib.dump(scaler_Y,'scalar_Y.pkl')
          print("scalers saved.")

       

     return X_train, X_test, y_train, y_test, feature_cols,target_cols



#-----flexible surrogate model------//
class FlexibleSurrogate(nn.Module):
       """
       input_size : int---number of input features
       hidden_layers :list --neurons per hidden layer e.g:-[64,128,64]
       output_size :int--number of outptuts
       """
       def __init__(self,input_size,hidden_layers,output_sizes):
              super().__init__()
              self.input_size=input_size
              self.hidden_layers=hidden_layers
              self.output_sizes=output_sizes
              
              layers=[]
              current_in=input_size
              for i in hidden_layers:
                     layers.append(nn.Linear(current_in,i))
                     layers.append(nn.ReLU())
                     layers.append(nn.Dropout(p=0.2))

                     current_in=i
                     
        
              layers.append(nn.Linear(current_in,output_sizes))
              self.network=nn.Sequential(*layers)
       def forward(self,x):
              return self.network(x)
       

# m1 = FlexibleSurrogate(4, [64, 128, 64], 3)     # same as model1
# m2 = FlexibleSurrogate(4, [32, 64, 32], 3)      # smaller
# m3 = FlexibleSurrogate(4, [128, 256, 128], 3)   # larger

x_train_t,x_test_t,y_train_t,y_test_t,feat,targ=load_and_prepare_dataset(
       'Xfoil_dataset_5k.csv',
       target_cols=['Cl','Cd','Cm']
)
X_train_tensor = torch.tensor(x_train_t, dtype=torch.float32)
X_test_tensor  = torch.tensor(x_test_t,  dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_t, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test_t,  dtype=torch.float32)    

model=FlexibleSurrogate(
       input_size=4,
       hidden_layers=[64,128,64],
       output_sizes=3

)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")


criterion=nn.MSELoss()       #loss function
optimizer=optim.Adam(model.parameters(),lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
)
n_epochs=1000
train_losses=[]
val_losses=[]


#Preventing Overfitting-----//
best_val_loss=float('inf')
threshold=100
threshold_count=0
best_weights=None

for epoch in range(n_epochs):
       model.train()
       #forward pass---//
       y_pred=model(X_train_tensor)
       loss=criterion(y_pred,y_train_tensor)

       #backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       train_losses.append(loss.item())

       model.eval()

       with torch.no_grad():
              y_val_pred=model(X_test_tensor)
              val_loss=criterion(y_val_pred,y_test_tensor)

       val_losses.append(val_loss.item())

       scheduler.step(val_loss.item())

       if epoch% 100==0:
              print(f"Epoch {epoch:4d} | "
              f"Train: {loss.item():.6f} | "
              f"Val: {val_loss.item():.6f}")



       if val_loss.item()<best_val_loss:
                best_val_loss=val_loss.item()
                threshold_count=0

                best_weights={k: v.clone() for k, v in model.state_dict().items()}
       else:
              threshold_count+=1

       if threshold_count>=threshold:
              print(f"\nEarly stopping at epoch {epoch}")
              print(f"Best val loss: {best_val_loss:.6f}")
              break

if best_weights:
       model.load_state_dict(best_weights)

# Saving model-------//
torch.save(model.state_dict(),'airfoil_surrogate.pt')
print("\nSaved")

torch.save(model,'airfoil_surrogate_full.pt')
print("\nSaved")

np.savez('training_history.npz',
         train_losses=train_losses,
         val_losses=val_losses)
print("\nSaved")

plt.figure(figsize=(10,5))
plt.plot(train_losses,label='train loss',alpha=0.8)
plt.plot(val_losses,   label='Val loss',   alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('training_curve.png',dpi=180)
plt.show()
print("saved")


#-----VALIDATION-------//
model.eval()
with torch.no_grad():
       y_pred_scaled=model(X_test_tensor)

#loading scalar---inverse transform to physical units
scaler_y=joblib.load('scalar_Y.pkl')
y_pred_real=scaler_y.inverse_transform(y_pred_scaled.numpy())
y_test_real=scaler_y.inverse_transform(y_test_tensor.numpy())

y_pred_real[:, 1] = 10 ** y_pred_real[:, 1]
y_test_real[:, 1] = 10 ** y_test_real[:, 1]


#R**2===for each output----//
target_names=['Cl','Cd','Cm']
for i,name in enumerate(target_names):
       r2=r2_score(y_test_real[:,i],y_pred_real[:,i])
       rmse=np.sqrt(mean_squared_error(
              y_test_real[:,i],y_pred_real[:,i]
       ))
       print(f"{name}: R2={r2:.4f}  RMSE={rmse:.5f}")


#----PARITY PLOTS-------///
fig,axes=plt.subplots(1,3,figsize=(15,6))
colors=['#2196F3','#4CAF50','#FF5722']
for i,(name, color) in enumerate(zip(target_names,colors)):
       ax=axes[i]
       true=y_test_real[:,i]
       pred=y_pred_real[:,i]
       r2=r2_score(true,pred)

       ax.scatter(true,pred,color=color,alpha=0.7,s=40)
       lims=[min(true.min(),pred.min())*0.95,
             max(true.max(),pred.max())*1.05]
       ax.plot(lims,lims,'k--',linewidth=1.5)
       ax.set_xlabel(f'actual {name}')
       ax.set_ylabel(f'predicted {name}')
       ax.set_title(f'{name} R2={r2:.4f}')
       ax.grid(True,alpha=0.5)

plt.suptitle('surrogate validation--predicted v/s actual',
             fontsize=13,fontweight='bold')
plt.tight_layout(pad=2)
plt.savefig('parity_plots.png',dpi=180)
plt.show()

df=pd.read_csv('Xfoil_dataset_5k.csv')
print(df['Cm'].describe())