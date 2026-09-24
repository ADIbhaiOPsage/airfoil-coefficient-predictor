import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

#CREATING TENSORS-- 4 ways
#way-1:--from python list(most common for small data)
airfoil_inputs=torch.tensor([2.0,3.0,12.0,8.0])
print(f"\nWay 1--from list    :{airfoil_inputs}")
print(f"shape      :{airfoil_inputs.shape}")
print(f"Data type   :{airfoil_inputs.dtype}")

#way-2--from numpy array(how our pipeline connects to PyTorch)
X_numpy=np.array([[2,4,12,8],
                 [4,4,12,4],
                 [0,0,12,0]])

X_tensor=torch.tensor(X_numpy,dtype=torch.float32)
print(f"\nway 2--from numpy  :\n{X_tensor}")
print(f"shape     :{X_tensor.shape}")

#way--3-built-in constructors (like np.zeros,np.ones)
zeros=torch.zeros(3,4)
ones=torch.ones(3,4)
rand=torch.rand(3,4)
randn=torch.randn(3,4)

print(f"\nway--3 TORCH randn:\n{randn}")

#way 4--linspace(like np.linspace)
aoa_range=torch.linspace(-4,16,50)
print(f"\nway 4--linspace(first 5): {aoa_range[:5]}")

#TENSOR PROPERTIES
t=torch.randn(5,4)
print(f"\nTensor:\n{t}")
print(f"shape :{t.shape}")
print(f"Dtype  :{t.dtype}")
print(f"Device  :{t.device}")
print(f"Dims  :{t.ndim}")


print(f"\nRows (dim 0)  :{t.shape[0]}")
print(f"\n (dim 1)  :{t.shape[1]}")

#TENSOR OPERATIONS->
X=torch.tensor([
    [2,4,12,4],
    [4,4,12,8],
    [0,0,12,0],
    [6,4,15,12]
],dtype=torch.float32)

#Arthemetic--operates on entire tensor at once
print(f"\nMean of each column (feature means):")
print(torch.mean(X,dim=0))


print(f"\nMax AoA in batch: {X[:, 3].max()}")
print(f"Min AoA in batch: {X[:, 3].min()}")

print(f"\nALL AoA values (column 3): {X[:,3]}")
print(f"first case (row 0)   :{X[0,:]}")
print(f"cases 1 and 2 :\n{X[1:3,:]}")


# requires_grad=True tells PyTorch:
# "track every operation on this tensor — I will need its gradient"

# Model WEIGHTS have requires_grad=True automatically
# Your INPUT DATA does not need gradients (usually)
# EXCEPTION: inverse design — you DO want gradient of output
#            w.r.t. input geometry → set requires_grad=True on input

#EXAMPLE--manual gradient computation
x=torch.tensor(3.0,requires_grad=True)
y=x**2 + 2*x +1
y.backward()  #compute dy/dx
print(f"\nGradient example.")
print(f"x      ={x.data}")
print(f"y=x2+2x+1 ={y.data}")
print(f"dy/dx   ={x.grad}")

#converting NUMPY TO PYTORCH
numpy_array=np.array([[1,2],
                      [3,4]])
torch_tensor=torch.tensor(numpy_array,dtype=torch.float32)
print(f"\nNumpy-->tensor:\n{torch_tensor}")

back_to_numpy=torch_tensor.detach().numpy()
print(f"tensor-->numpy:\n{back_to_numpy}")
# .detach() = stop gradient tracking before converting
# Without .detach() on gradient-tracked tensor → error
# Always use .detach().numpy() when converting back

# ------------------------------------------------------------------------>>>

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

     return X_train, X_test, y_train, y_test, feature_cols,target_cols


# ------>test run<------
X_train, X_test, y_train, y_test, feat, targ = load_and_prepare_dataset(
    'fluent_dataset_clean.csv',
    target_cols=['Cl', 'Cd', 'Cm']
)
x_train_t=torch.tensor(X_train,dtype=torch.float32)
x_test_t=torch.tensor(X_test,dtype=torch.float32)
y_train_t=torch.tensor(y_train,dtype=torch.float32)
y_test_t=torch.tensor(y_test,dtype=torch.float32)
print(f"\nfeatures in training set:\n{x_train_t.shape[1]}")
print(f"\ntargets :\n{y_train_t.shape[1]}")

# print(f"\nX_train shape: {X_train.shape}")  # (n_train, 4)
# print(f"y_train shape: {y_train.shape}")  # (n_train, 3)
# print(f"\nFeatures : {feat}")
# print(f"Targets  : {targ}")

print(f"\nmean  :\n{torch.mean(x_train_t,dim=0)}")
print(f"\nstandard deviation :\n{torch.std(x_train_t,dim=0)}")

print("first 5 rows of training data:")
print(x_train_t[:5,:])

print("\nAoA column only(first 5 values):")
print(x_train_t[:5,3])

print("\nrows where AoA is positive (first 3 matches):")
mask=x_train_t[:,3]>0
positive_aoa_cases=x_train_t[mask]
print(positive_aoa_cases[:3,:])


#Matirx Multipication--critical for understanding linear layers.
# in nn.linear(4,64): output-->input x weights.T + bias
weights=torch.randn(3,4)
bias=torch.zeros(3)
case=x_train_t[0]
output=weights@case+bias
print(f"input case shape: {case.shape}")
print(f"layer weights shape: {weights.shape}")
print(f"output shape: {output.shape}")
print(f"output values:{output}")

a=torch.tensor(5,dtype=torch.float32,requires_grad=True)
b=a**3
n_numpy=b.detach().numpy()
print(f"successfully converted to numpy: {b.numpy}")


#Mapping OOPs --->PyTorch---------->>
import torch.nn as nn
from sklearn.metrics import r2_score

#understanding nn.linear in isolation.
layer=nn.Linear(in_features=4,out_features=64)
print(f"\nlayer weight shape: {layer.weight.shape}")
print(f"layer bias shape :{layer.bias.shape}")
print(f"total parameters :{layer.weight.numel()+layer.bias.numel()}")

# .numel() = number of elements in tensor

# One forward pass through this single layer
single_case=torch.tensor([2,4,12,8],dtype=torch.float32)
batch_cases=torch.tensor([
     [2,4,12,4],
     [4,4,12,8],
     [0,0,12,0],
     [6,4,15,12]
],dtype=torch.float32)

output_single=layer(single_case.float())
output_batch=layer(batch_cases)

print(f"\n single case input shape:{single_case.shape}")
print(f"single case output shape: {output_single.shape}")
print(f"\nBatch input  shape: {batch_cases.shape}")         # [4, 4]
print(f"Batch output shape: {output_batch.shape}")  

manual=torch.sum(layer.weight[0]*single_case)+layer.bias[0]
print(f"manual: {manual.item():.6f}")
print(f"layer :{layer(single_case.float())[0].item():.6f}")

# Understanding ReLU in isolation

relu=nn.ReLU()
test_values=torch.tensor([-2,-0.5,0,0.3,1,2.5],dtype=torch.float32)
relu_output=relu(test_values)

print(f"\nInputs :{test_values.tolist()}")
print(f"output :{relu_output.tolist()}")

#why relu matters--visualise this
x_plot=torch.linspace(-3,3,200)
y_relu=relu(x_plot)
y_linear=x_plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x_plot.numpy(),y_linear.numpy(),'b-',label='No activation ')
plt.plot(x_plot.numpy(),y_relu.numpy(),'r-',label='relu activation')
plt.title('ReLU vs No activation')
plt.xlabel('input value')
plt.ylabel('output value')
plt.legend()
plt.grid(True,alpha=0.4)
plt.axhline(y=0,color='k',linewidth=0.5)
plt.axvline(x=0,color='k',linewidth=0.5)

#showing what relu to a layer's output distribution
layer_out=layer(batch_cases)  #raw layer output 
relu_out=relu(layer_out)      #after relu

plt.subplot(1,2,2)
plt.hist(layer_out.detach().numpy().flatten(),
         bins=30,alpha=0.5,label='before ReLU',color='blue')
plt.hist(relu_out.detach().numpy().flatten(),
         bins=30,alpha=0.6,label='after reLU',color='red')
plt.title('neuron output distribution')
plt.xlabel('activation value')
plt.ylabel('count')
plt.legend()
plt.grid(True,alpha=0.5)
plt.tight_layout(pad=0.5)
plt.show()

fraction=(relu_out==0).float().mean()
print(f"the fraction of activated neurons:- {fraction*100}")

#--------------------------------------------------------------------------------------->>
#building model 1__airfoilsurrogate(cl,Cd,Cm)

class airfoilsurrogate(nn.Module):
     """surrogate model predicting Cl,Cd,Cm from airfoil geometry+AoA
        Input (4):- camber%, camber_poisiton , thickness% , AoA_degrees
        output (3):- Cl, Cd,Cm
        architecture: 4-->[64-->128->64]-->3
        activation : RelU on hidden layers , none on output
     """

     def __init__(self):
          super().__init__()

          self.network=nn.Sequential(
               #layer 1: expands inputs to 64 neurons
               nn.Linear(4,64),
               nn.ReLU(),

               #layer 2:find the combinations of patterns
               nn.Linear(64,128),
               nn.ReLU(),

               #layer 3:compress to essential
               nn.Linear(128,64),
               nn.ReLU(),

               #layer 4: output values--Cl,Cd,Cm
               nn.Linear(64,3)
          )
     def forward(self,x):
               """
               define forward pass. called when you do model(input).
               PyTorch calls this automatically--never call forward() directly."""
               return self.network(x)
          
#--creating and inspecting model---------------------#
model1=airfoilsurrogate()
print(f"\nmodel architure:")
print(model1)

total_params=sum(p.numel() for p in model1.parameters())
trainable=sum(p.numel() for p in model1.parameters() if p.requires_grad)
print(f"\nTotal parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable:,}")

#---single pass-------#\
single=torch.tensor([2,4,12,8],dtype=torch.float32)
pred=model1(single)
print(f"\nSingle case forward pass:")
print(f"input shape: {single.shape}")
print(f"output shape: {pred.shape}")
print(f"predictions : Cl={pred[0].item():.4f}  "
      f"Cd={pred[1].item():.4f} "
      f"Cm={pred[2].item():.4f}")

#----batch---shape[4,4]----#
batch_pred=model1(batch_cases)
print(f"\nBatch forward pass:")
print(f"Input  shape: {batch_cases.shape}")   # [4, 4]
print(f"Output shape: {batch_pred.shape}")    # [4, 3]
print(f"\nPredictions (4 cases):")
df=pd.DataFrame(batch_pred.detach().numpy(),
                columns=['Cl','Cd','Cm']
)
df.index+=1
print(df)

#-----inspecting layer-by-layer weights--------#
layers=[]
layers_name=['linear(4->64)','linear(64->128)','linear(128->64)','linear(64->3)']
layer_idx=0
for m in model1.modules():
      if isinstance(m,nn.Linear):
            w=tuple(m.weight.shape)
            b=tuple(m.bias.shape)
            params=m.weight.numel() + m.bias.numel()

            layers.append({
                  'layer': layers_name[layer_idx],
                  'weight shape':str(w),
                  'bias shape':str(b),
                  'total params':params

            })
            
            layer_idx+=1
layer_df=pd.DataFrame(layers)
layer_df.index+=1

print(layer_df)







class Cpsurrogate(nn.Module):
      """
      predict full Cp distribution along chord.
      inputs (4): camber%, position, thickness% , AoA_deg
      outputs(100): Cp values at 100 chord loactions
      architecture: 4->[129->256->256->128]->100
      wider+deeper than model 1--100 ouptuts need more capacity
      """
      def __init__(self,n_cp_points=100):
            super().__init__()

            self.n_cp_points=n_cp_points

            self.network=nn.Sequential(
                   nn.Linear(4,   128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_cp_points) 
            )

      def forward(self,x):
                  return self.network(x)
            
model2=Cpsurrogate(n_cp_points=100)
total_cp=sum(p.numel() for p in model2.parameters())
print(f"\nCp model total parameters: {total_cp:,}")
print(f"Scalar model parameters   : {total_params:,}")
print(f"Cp model is {total_cp/total_params:.1f}× larger")

cp_pred=model2(single)
print(f"\nCp prediction shape: {cp_pred.shape}")  # [100]
print(f"First 5 Cp values  : {cp_pred[:5].detach().numpy()}")
print("(Random — not trained yet)")

x=np.linspace(0,1,100)
y=cp_pred.detach().numpy()
plt.figure(figsize=(10,5))
plt.plot(x,y,'b-',linewidth=2,label='predicted Cp')
plt.gca().invert_yaxis()
plt.title("Cp Prediction (untrained — random weights)")
plt.xlabel("Chord Location (x/c)")
plt.ylabel("Pressure Coefficient (Cp)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()


#MODEL COMPARISON UTILITY-----------//
def compare_model(*models_and_names):
       """
       take pairs of (model,name) and print comparison table.
       show architecture + parameter count side by side.
       """
       table=[]
       for model,name in models_and_names:
              params=sum(p.numel() for p in model.parameters())
              layers=sum(1 for m in model.modules() if isinstance(m,nn.Linear))
              table.append({
                     'model':name,
                     'params':params,
                     'layers':layers
                     
              })

       df_table=pd.DataFrame(table)
       df_table.index+=1
       print(df_table)
       return df_table


df_comparison=compare_model(
       (model1,"airfoilsurrogate (Cl,Cd,Cm)"),
       (model2,"CpSurrogate (100 Cp points)")
)


#train vs Eval mode
class airfoilsurrogateWithDropout(nn.Module):
       def __init__(self):
              super().__init__()
              self.network=nn.Sequential(
                     nn.Linear(4,64),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),

                     nn.Linear(64,128),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),

                     nn.Linear(128,64),
                     nn.ReLU(),
                     nn.Dropout(p=0.2),
            
                     nn.Linear(64,3)

              )

       def forward(self,x):
              return self.network(x)

model_drop=airfoilsurrogateWithDropout()
test_case=torch.tensor([2,4,12,8],dtype=torch.float32) 

print("Train Mode (dropout is actively randomly zeroing neurons):")
preds=[]
# model_drop.train()
# for i in range(5):
#        pred=model_drop(test_case).detach().numpy()
#        preds.append(pred)

# preds_df=pd.DataFrame(preds,
#                    columns=['Cl','Cd','Cm'])
# print(preds_df)

#---Evaluation Mode-----///
model_drop.eval()
for i in range(5):
       pred=model_drop(test_case).detach().numpy()
       preds.append(pred)

preds_df=pd.DataFrame(
       preds,columns=['Cl','Cd','Cm']
)
print(preds_df)











                     



       




       
                  


            
            
      



