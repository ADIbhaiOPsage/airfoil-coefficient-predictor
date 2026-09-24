import numpy as np
import matplotlib.pyplot as plt
# 1D array
# wind_speed= np.array([10,20,30,40,60])
# print("wind speeds in m/s: ")
# print(wind_speed)
# # 2D matrix
# press_matrix=np.array([[101,105,107],
#                       [111,104,110]])
# print("\n pressure matrix (Pa): ")
# print(press_matrix)
# # shaping matrix
# print("\nshape of the matrix (rows,columns): ")
# print(press_matrix.shape)
pressure_matrix = np.array([
    [101, 105, 102],
    [103, 108, 104],
    [106, 110, 105],
    [108, 115, 104],
    [109, 120, 103]])
# Slicing 
# sensor_b_data=pressure_matrix[:,1]
# print("sensor B data",sensor_b_data)
# # plotting and labeling
# import matplotlib.pyplot as plt;
# plt.plot(sensor_b_data,marker='o',color='blue')
# plt.title("wind tunnel: sensor B pressure spike")
# plt.xlabel("time(seconds)")
# plt.ylabel("pressure (Pa)")
# plt.grid(True)
# plt.show()

# vectorization
# EXAMPLE:-
# NACA 4-digit thickness distribution formula
# This is the actual equation used to generate airfoil shapes
# t = max thickness as fraction of chord (e.g. 0.12 for 12%)
# x = array of points from 0 to 1 along the chord

x=np.linspace(0,1,2000)  #100 points from leading to trailing edge
t=0.24 # 12% thickness - NACA xx12

#thickness formula (NACA equation)
yt=(t / 0.2) * (0.2969*np.sqrt(x) 
                 - 0.1260*x 
                 - 0.3516*x**2 
                 + 0.2843*x**3 
                 - 0.1015*x**4)
upper=yt
lower=-yt

plt.figure(figsize=(10,4))
plt.plot(x,upper,'b-',label='upper surface')
plt.plot(x,lower,'r-',label='lower surface')
plt.fill_between(x, upper, lower, alpha=0.2, color='gray')
plt.axis('equal')
plt.grid(True)
plt.title(f'NACA 00{t*100:02f} airfoil profile')
plt.xlabel('x/c (chord fraction)')
plt.ylabel('y/c')
plt.legend()
plt.show()


# Simulated wind tunnel results for 5 different airfoils
# Columns: [AoA (deg), Cl, Cd, Cl/Cd ratio]
# Each row is one test condition
tunnel_data=np.array([
    [-2,  0.10, 0.012, 8.3],
    [ 0,  0.25, 0.011, 22.7],
    [ 2,  0.45, 0.012, 37.5],
    [ 4,  0.65, 0.013, 50.0],
    [ 6,  0.85, 0.015, 56.7],
    [ 8,  1.02, 0.018, 56.7],
    [10,  1.15, 0.024, 47.9],
    [12,  1.20, 0.035, 34.3],
    [14,  1.10, 0.060, 18.3],  # approaching stall
    [16,  0.80, 0.110,  7.3],  # post stall
])
AoA= tunnel_data[:, 0]
cl  = tunnel_data[:, 1]
cd  = tunnel_data[:, 2]
ld  = tunnel_data[:, 3]  # lift-to-drag ratio


stall_idx=np.argmax(cl)
max_ld_idx=np.argmax(ld)

fig,axes=plt.subplots(1,3,figsize=(15,5))
fig.subplots_adjust(wspace=0.2)

#plot-1:lift curve (cl vs AoA)
axes[0].plot(AoA,cl,marker='o',linestyle='-',label='lift Curve')
axes[0].plot(AoA[stall_idx],cl[stall_idx],'ro',markersize=10,label='stall point')
axes[0].set_title('Lift Curve ($C_l$ vs AoA)')
axes[0].set_xlabel('Angle of Attack (deg)')
axes[0].set_ylabel('Coefficient of Lift ($C_l$)')
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()

# Plot 2: Drag Polar (Cd vs AoA)
axes[1].plot(AoA, cd, marker='o', linestyle='-', color='orange',label='drag curve')
axes[1].set_title('Drag Polar ($C_d$ vs AoA)')
axes[1].set_xlabel('Angle of Attack (deg)')
axes[1].set_ylabel('Coefficient of Drag ($C_d$)')
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend()

# Plot 3: Efficiency (L/D ratio vs AoA)
axes[2].plot(AoA, ld, marker='o', linestyle='-', color='green')
axes[2].plot(AoA[max_ld_idx], ld[max_ld_idx], 'g*', markersize=15, label='Max L/D')
axes[2].set_title('Efficiency (L/D Ratio vs AoA)')
axes[2].set_xlabel('Angle of Attack (deg)')
axes[2].set_ylabel('L/D Ratio')
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].legend()

plt.tight_layout(pad=2)
plt.show()

def generate_naca4(m_percent,p_tenth,t_percent,n_points=100):
    '''  Generate NACA 4-digit airfoil coordinates.
    
    Parameters:
        m_percent : max camber as percentage (0-9), e.g. 2 for NACA 2xxx
        p_tenth   : camber position in tenths (1-9), e.g. 4 for NACA x4xx  
        t_percent : max thickness as percentage (6-24), e.g. 12 for NACA xx12
        n_points  : number of points along chord
    
    Returns:
        x_upper, y_upper : upper surface coordinates
        x_lower, y_lower : lower surface coordinates
        '''
    m=m_percent/100
    p=p_tenth/10
    t=t_percent/100
    x=np.linspace(0,1,n_points)

# thickness distribution
    yt = (t/0.2) * (0.2969*np.sqrt(x) - 0.1260*x 
                   - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    
# Camber line and gradient
    # The camber line yc has two equations - one for x < p, one for x >= p
    # The gradient dyc_dx also has two equations
    # Look up "NACA 4 digit camber line equation" and implement using np.where()
    # np.where(condition, value_if_true, value_if_false) - handles both regions at once

    yc=np.where(x<p,
                
                # forward section(x<p)
                (m/p**2)*(2*p*x-x**2),
                # rear section(x>=p)
                (m/(1-p)**2)*(1-2*p+2*p*x-x**2))
    
    dyc_dx = np.where(x < p,
                      (2*m/p**2) * (p - x),
                      (2*m/(1-p)**2) * (p - x))
    
    theta = np.arctan(dyc_dx)

    # Upper and lower surface coordinates
    x_upper = x  - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x  + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)

    return x_upper, y_upper, x_lower, y_lower

# subplots of three aerofoils(2214,4412,6412)
fig,axes=plt.subplots(1,3,figsize=(15,5))
fig.subplots_adjust(wspace=0.3)
xu, yu, xl, yl = generate_naca4(2, 4, 12)

axes[0].plot(xu,yu,'b-',label='upper')
axes[0].plot(xl,yl,'r-',label='lower')
axes[0].fill_between(xu,yu,yl,alpha=0.2,color='steelblue')
axes[0].axis('equal')
axes[0].grid(True,alpha=0.4)
axes[0].set_title('NACA 2412')
axes[0].legend()

xu, yu, xl, yl = generate_naca4(4, 4, 12)

axes[1].plot(xu,yu,'b-',label='upper')
axes[1].plot(xl,yl,'r-',label='lower')
axes[1].fill_between(xu,yu,yl,alpha=0.2,color='steelblue')
axes[1].axis('equal')
axes[1].grid(True,alpha=0.4)
axes[1].set_title('NACA 4412')
axes[1].legend()

xu, yu, xl, yl = generate_naca4(6, 4, 12)

axes[2].plot(xu,yu,'b-',label='upper')
axes[2].plot(xl,yl,'r-',label='lower')
axes[2].fill_between(xu,yu,yl,alpha=0.2,color='steelblue')
axes[2].axis('equal')
axes[2].grid(True,alpha=0.4)
axes[2].set_title('NACA 6412')
axes[2].legend()

plt.tight_layout(pad=2)
plt.show()


np.random.seed(42)
n_cases=50
dataset= np.column_stack([
    np.random.uniform(0, 9,   n_cases),   # camber
    np.random.randint(1, 10,  n_cases),   # camber position
    np.random.uniform(6, 24,  n_cases),   # thickness
    np.random.uniform(-4, 16, n_cases),   # AoA
    np.random.uniform(0.1, 1.5, n_cases), # Cl (fake)
    np.random.uniform(0.01, 0.1, n_cases),# Cd (fake)
    np.random.uniform(-0.1, 0.05, n_cases)# Cm (fake)
])

print(f"Dataset shape: {dataset.shape}")  # should be (50, 7)

AoA=dataset[:,3]
Cl=dataset[:,4]
Cd=dataset[:,5]

# Mean and Std of Cl
mean_cl=np.mean(Cl)
std_cl=np.std(Cl)
print(f"MEAN CL: {mean_cl:.4f}, Standard deviation= {std_cl:.4f}")

# Cl > 1.0 creates a boolean array [True, False, True...]. 
# np.sum() treats True as 1 and False as 0.
cases_over_1 = np.sum(Cl > 1.0)
print(f"Q2: There are {cases_over_1} cases with a Cl > 1.0")

# Highest L/D ratio (Cl/Cd) and its row index ---
ld_ratio = Cl / Cd
max_ld = np.max(ld_ratio)
max_idx = np.argmax(ld_ratio) # argmax returns the INDEX of the maximum value
print(f"Q3: Highest L/D ratio is {max_ld:.2f}, located at row index {max_idx}")

# We use the bitwise '&' operator for element-wise boolean 'AND'
# Parentheses around the conditions are strictly required!
mask = (AoA >= 4) & (AoA <= 10)
filtered_data = dataset[mask]
print(f"Q4: Found {filtered_data.shape[0]} cases where AoA is between 4° and 10°")

# Normalise Cl to range [0,1] ---
cl_min = np.min(Cl)
cl_max = np.max(Cl)
cl_normalized = (Cl - cl_min) / (cl_max - cl_min)
# Printing just the first 5 so it doesn't flood your console
print(f"Q5: First 5 normalised Cl values: {cl_normalized[:5].round(4)}")