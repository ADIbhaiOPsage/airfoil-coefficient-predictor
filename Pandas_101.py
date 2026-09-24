import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n=100

AoA=np.random.uniform(-4,16,n)
camber=np.random.uniform(0,9,n)
pos=np.random.randint(1,10,n).astype(float)
thick=np.random.uniform(6,24,n)

# Cl increases with AoA and camber (simplified physics)
Cl = 0.11 * AoA + 0.08 * camber + np.random.normal(0, 0.05, n)

# Cd increases with AoA squared and decreases with thickness (simplified)
Cd = 0.008 + 0.0006 * AoA**2 + np.random.normal(0, 0.002, n)
Cd = np.clip(Cd, 0.005, 0.15)  # physical bounds

# Cm slightly negative (nose-down pitching moment)
Cm = -0.05 - 0.003 * camber + np.random.normal(0, 0.01, n)

# Deliberately inject 5 bad rows (simulating non-converged cases)
bad_idx = np.random.choice(n, 5, replace=False)
Cl[bad_idx] = np.nan
Cd[bad_idx] = np.nan

# Build the dataframe
df = pd.DataFrame({
    'camber_pct'   : camber,
    'camber_pos'   : pos,
    'thickness_pct': thick,
    'AoA_deg'      : AoA,
    'Cl'           : Cl,
    'Cd'           : Cd,
    'Cm'           : Cm
})

df.to_csv('fluent_dataset.csv', index=False)
print("Dataset saved.\n")

df_loaded=pd.read_csv('fluent_dataset.csv')
print(f"loaded dataset shape:{df_loaded.shape}")
print(df_loaded.head(6))
print(df_loaded.describe())
print(df_loaded.dtypes)

#--------------> cleaning and filtering(data quality pipeline)<--------------
df=pd.read_csv('fluent_dataset.csv')
# TASK-1:removing non-converged cases
df_garb=df.dropna(subset=['Cl','Cd','Cm'])
print(f"Original rows : {len(df)}")
print(f"After cleaning: {len(df_garb)}")
print(f"Cases removed : {len(df) - len(df_garb)}\n")
# removing physical bounds filtering
#  Even converged cases can be unphysical if the solver diverged slowly
# Apply engineering judgement bounds:
# Cl must be between -0.5 and 2.0  (subsonic airfoil physics)
# Cd must be between 0.003 and 0.15 (realistic drag range)
# Cm must be between -0.3 and 0.1  (typical pitching moment range)

df_clean=df_garb[(df_garb['Cl']>=-0.5)&(df_garb['Cl']<=2.0)&
                (df_garb['Cd']>=0.003)&(df_garb['Cd']<=0.15)&
                (df_garb['Cm']>=-0.3)&(df_garb['Cm']<=0.1)].copy()
print(f"Rows remaining after physical filtering: {len(df_clean)}\n")

df_clean['LD_ratio']=df_clean['Cl']/df_garb['Cd']


df_clean['AoA_regime']=pd.cut(df_garb['AoA_deg'],
                             bins=[-np.inf,0,6,12,np.inf],
                             labels=['neg','low','mod','high'],
                             right=False)


# highest L/D
best_ld_row=df_clean.loc[df_clean['LD_ratio'].idxmax()]
print(f"highest L/D Ratio ({best_ld_row['LD_ratio']:.2f}):")
print(f"  --> NACA {int(best_ld_row['camber_pct'])}{int(best_ld_row['camber_pos'])}{int(best_ld_row['thickness_pct']):02d} at {best_ld_row['AoA_deg']:.1f}° AoA")

max_cl_row = df_clean.loc[df_clean['Cl'].idxmax()]
print(f"Highest Lift Coefficient (Cl = {max_cl_row['Cl']:.3f}):")
print(f"  --> NACA {int(max_cl_row['camber_pct'])}{int(max_cl_row['camber_pos'])}{int(max_cl_row['thickness_pct']):02d} at {max_cl_row['AoA_deg']:.1f}° AoA")

# Lowest Cd
min_cd_row = df_clean.loc[df_clean['Cd'].idxmin()]
print(f"Lowest Drag Coefficient (Cd = {min_cd_row['Cd']:.4f}):")
print(f"  --> NACA {int(min_cd_row['camber_pct'])}{int(min_cd_row['camber_pos'])}{int(min_cd_row['thickness_pct']):02d} at {min_cd_row['AoA_deg']:.1f}° AoA\n")

df_clean.to_csv('fluent_dataset_clean.csv',index=False)
print("column list in clean dataset:")
for i,col_name in enumerate(df_clean.columns,1):
    print(f"{i}.{col_name}")


df=pd.read_csv('fluent_dataset_clean.csv')
group_stats=df.groupby('AoA_regime')[['Cl','Cd','Cm']].mean()
print(group_stats.round(3))

# MEthod 1:- using select_dtypes to get only numeric columns and then corr()
numeric_df=df.select_dtypes(include=[np.number])
correlation_matrix=numeric_df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))
# method 2:- using a predefined list of numeric columns (safer if we know the column names)
numeric_cols=['camber_pct','camber_pos','thickness_pct','AoA_deg','Cl','Cd','Cm']
corr_matrix=df[numeric_cols].corr()
print(corr_matrix['Cl'].sort_values(ascending=False))
print("\n")  # Correlation of all features with Cl
print(corr_matrix['Cd'].sort_values(ascending=False)) 


cl_correlations = corr_matrix['Cl'].drop('Cl')
max_corr=cl_correlations.max()
max_corr_feature=cl_correlations.idxmax()
print(f"Maximum correlation with Cl: {max_corr} (Feature: {max_corr_feature})")
cd_correlations = corr_matrix['Cd'].drop('Cd')
max_corr=cd_correlations.max()
max_corr_feature=cd_correlations.idxmax()
print(f"Maximum correlation with Cd: {max_corr} (Feature: {max_corr_feature})")


# correlation heatmap
fig,ax=plt.subplots(figsize=(9,7))

# HEATMAP USING SEABORN LIB--------->
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='RdBu',
    vmin=-1,vmax=1,
    cbar=True
)
plt.title('input-output correlation matrix-airfoil Dataset',fontsize=12)
plt.tight_layout()
plt.show()

# HISTOGRAM USING SEABORN----->
fig,axes=plt.subplots(2,3,figsize=(15,8))
axes=axes.flatten()
cols_to_plot=['camber_pct','thickness_pct', 'AoA_deg', 'Cl', 'Cd', 'LD_ratio']
for i,cols in enumerate(cols_to_plot):
    sns.histplot(data=df,x=cols,bins=15,ax=axes[i],kde=True,color='teal')
    axes[i].set_title(f"distribution of{cols}")

plt.tight_layout()
plt.show()


