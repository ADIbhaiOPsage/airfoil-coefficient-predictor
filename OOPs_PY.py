# EXERCISE 1: Class basics
# Goal: understand class, __init__, self, methods, objects
class FluidProperties:
     """
    Stores properties of a fluid used in CFD simulation.
    Like how Fluent needs density, viscosity before running.
    """
     def __init__(self,name,density,viscosity,temperature):
           """
        __init__ = constructor. Runs automatically on object creation.
        self     = this specific object. Stores values ON it.
        
        Think: self.density = "MY density" (this object's density)
        """
           self.name        = name
           self.density     = density      # kg/m³
           self.viscosity   = viscosity    # Pa·s
           self.temperature = temperature  # Kelvin

     def display_properties(self):
          """Print all fluid properties cleanly."""
          print(f"\nFluid     : {self.name}")
          print(f"Density   : {self.density} kg/m³")
          print(f"Viscosity : {self.viscosity} Pa·s")
          print(f"Temp      : {self.temperature} K")

     def kinematic_viscosity(self):
            """
        Kinematic viscosity = dynamic viscosity / density
        Used in Reynolds number calculation.
        """
            return self.viscosity / self.density
    
     def is_compressible(self,velocity):
            """
        Mach > 0.3 = compressible flow. Different equations needed.
        Speed of sound in air ≈ 343 m/s at 293K
        Scales with sqrt(temperature/293)
        """
            speed_of_sound = 343 * (self.temperature / 293) ** 0.5
            mach           = velocity / speed_of_sound
            return mach > 0.3, round(mach, 3)
     def reynolds_number(self,velocity,length):
           return(self.density*velocity*length/self.viscosity)
     
 
 #Create objects---->
air_sea_level=FluidProperties(
      name="Air at sea level",
      density=1.225,
      viscosity=1.81e-5,
      temperature=293
)

air_cruise=FluidProperties(
      name="air at 10,000m cruise",
       density     = 0.414,
    viscosity   = 1.46e-5,
    temperature = 223
)

water = FluidProperties(
    name        = "Water at 20°C",
    density     = 998,
    viscosity   = 1.002e-3,
    temperature = 293
)

fluids=[air_sea_level,air_cruise,water]
# displaying properties--->
for i in fluids:
      i.display_properties()

# Kinematic Viscosity--->
for i in fluids:
      kv=i.kinematic_viscosity()
      print(f"{i.name}: {kv:.3e} m2/s")

# Reynolds Number Comparison--->
v_uav = 60
chord = 0.5

re_sea = air_sea_level.reynolds_number(v_uav, chord)
re_alt = air_cruise.reynolds_number(v_uav, chord)

print(f"Re at Sea Level : {re_sea:.2e}")
print(f"Re at 10km Alt  : {re_alt:.2e}")

if(re_sea>re_alt):
      print("Conclusion: Sea level has a higher Reynolds number.")
else:
     print("Conclusion: 10km altitude has a higher Reynolds number.")
print("Why it matters: Higher Re means the flow is more likely to transition to turbulence sooner,"
      "requiring different turbulence models (like k-omega SST) in Fluent.")



# Compressibility Check--->
test_velocities=[50,80,100,120,200]

for v in test_velocities:
      is_comp,mach=air_sea_level.is_compressible(v)
      print(f"velocity {v:3} m/s--> Mach{mach:3f}-->compressible:{is_comp}")

# EXERCISE 2: Methods that modify object state
# Goal: understand how methods read AND write self attributes
#       __str__ method for clean printing




class simulationCase:
      """
    Represents one Fluent simulation case.
    Tracks: inputs, status, results.
    Like your PyFluent script managing one case in the batch.
    """
      def __init__(self,case_id, camber, position, thickness, aoa):
              # Inputs
        self.case_id   = case_id
        self.camber    = camber
        self.position  = position
        self.thickness = thickness
        self.aoa       = aoa

        # status tracking-starts empty
        self.status='pending'
        self.iteration=0
        self.converged=False

        # results-empty until simulation runs
        self.Cl=None
        self.Cd=None
        self.Cm=None
        self.residual=None

      def __str__(self):
             """
        Special method. Called automatically when you print(object).
        Like a custom label for your object.
        """
             return(f"Case {self.case_id:03d} | "
                f"NACA {self.camber}{self.position}{self.thickness:02d} | "
                f"AoA={self.aoa:+.1f}° | "
                f"Status={self.status}")
      def start_simulation(self):
             """Change status to running."""
             self.status='running'
             print(f"case {self.case_id:03d}: simulation started. ")

      def update_iteration(self,iteration,residual):
            """
        Called each convergence check.
        Modifies self.iteration and self.residual.
        """
            self.iteration=iteration
            self.residual=residual

            # auto-converge if residual below threshold
            if residual<1e-04:
                  self.converged=True

      def store_results(self,Cl,Cd,Cm):
             """Store aerodynamic results. Mark as done."""
             if not self.converged:
                   print(f"WARNING!: Case {self.case_id} storing results before convergence.")
             self.Cl=Cl
             self.Cd=Cd
             self.Cm=Cm
             self.status="done"

      def mark_failed(self,reason):
             """Mark case as failed. Used for bad Fluent runs."""
             self.status=f"failed: {reason}"
             print(f"Case {self.case_id:03d} FAILED- {reason}")

      def lift_to_drag(self):
            """Return L/D ratio. None if results not available."""
            if self.Cl is None or self.Cd is None:
                  return None
            if self.Cd==0:
                  return None
            return round(self.Cl/self.Cd,2)
      
      def to_dict(self):
            """
        Convert case to dictionary — for saving to pandas DataFrame.
        You'll use this to build your training dataset.
        """
            return{
            'case_id'       : self.case_id,
            'camber'        : self.camber,
            'position'      : self.position,
            'thickness'     : self.thickness,
            'aoa'           : self.aoa,
            'Cl'            : self.Cl,
            'Cd'            : self.Cd,
            'Cm'            : self.Cm,
            'converged'     : self.converged,
            'iterations'    : self.iteration,
            'status'        : self.status
                  
            } 
      

# Simulating a batch of cases------>
import pandas as pd
import numpy as np

# creating 5 simulation cases
cases=[
      simulationCase(1,2,4,12,4.0),
      simulationCase(2,4,4,12,8.0),
      simulationCase(3,0,0,12,0.0),
      simulationCase(4,2,4,12,20.0)
]

for i in cases:
      print(i)

for i in cases:
      i.start_simulation()

cases[0].update_iteration(150,8e-5)
print(cases[0])

cases[1].update_iteration(200, 6e-5)
print(cases[1])

cases[2].update_iteration(100, 3e-5)
print(cases[2])

cases[3].mark_failed("diverged at high AoA")
print(cases[3])


cases[0].store_results(Cl=0.65, Cd=0.013, Cm=-0.04)
cases[1].store_results(Cl=1.02, Cd=0.018, Cm=-0.06)
cases[2].store_results(Cl=0.25, Cd=0.011, Cm=-0.01)

for case in cases:
    # We only want to print L/D if the case actually finished
    if case.status == 'done':
        print(f"Case {case.case_id} L/D: {case.lift_to_drag()}")


records=[case.to_dict() for case in cases]
 #Using a list comprehension to quickly grab dictionaries from all 5 objects
df=pd.DataFrame(records)
print(df)

df_clean=df[df['status']=='done']

mean_cl = df_clean['Cl'].mean()
mean_cd = df_clean['Cd'].mean()

print(f"Cleaned dataset has {len(df_clean)} valid cases.")
print(f"Mean Cl: {mean_cl:.3f}")
print(f"Mean Cd: {mean_cd:.3f}")
            


# EXERCISE 3: Inheritance + super()
# Goal: child class extends parent. super() calls parent code.
#       IDENTICAL pattern to class MyModel(nn.Module)


# Parent class---->
class baseairfoil:
       """
    Parent class. Defines core airfoil behaviour.
    All airfoil types inherit from this.
    Like nn.Module — defines core behaviour all models inherit.
    """
       def __init__(self,camber,position,thickness):
              self.camber    = camber
              self.position  = position
              self.thickness = thickness
              self.name      = f"NACA {camber}{position}{thickness:02d}"

       def geometry_summary(self):
             print(f"{self.name}: camber={self.camber}%,"
                   f"pos={self.position},thick={self.thickness}%")
       def is_symmetric(self):
             return self.camber==0 and self.position==0
       def max_thickness_location(self):
             '''NACA 4-digit: max thickness always at 30% chord.'''
             return 0.30
       

# child class---->
class windtunnelairfoil(baseairfoil):
        """
    Extends BaseAirfoil with experimental measurement data.
    Inherits ALL BaseAirfoil methods automatically.
    Adds its own methods on top.
    
    IDENTICAL pattern to:
    class AirfoilSurrogate(nn.Module):
        def __init__(self):
            super().__init__()   ← calls nn.Module's setup
            self.network = ...   ← adds your own stuff
    """
        def __init__(self,camber,position,thickness,reynolds_number,tunnel_speed):
               # MUST call parent __init__ first
        # Sets up self.camber, self.position, self.thickness, self.name
        # Without this → those attributes don't exist → AttributeError
            super().__init__(camber,position,thickness)

            # adding child-specific attributes
            self.reynolds_number=reynolds_number
            self.tunnel_speed=tunnel_speed
            self.measurements=[] #list of (aoa,cl,cd,cm)

        def add_measurement(self,aoa,Cl,Cd,Cm):
              '''add one experimental data point'''
              self.measurements.append(
                    {
                      'aoa':aoa,'Cl':Cl,'Cd':Cd,'Cm':Cm
                    }
              )

        def stall_angle(self):
              """Find AoA at which maximum Cl across all measurements."""
              if not self.measurements:
                    return None
              best=max(self.measurements,key=lambda m:m['Cl'])
              return best['aoa']
        def max_Cl(self):
              if not self.measurements:
                    return None
              return max(m['Cl'] for m in self.measurements)
        
        def to_dataframe(self):
              '''coverts measurements to dataframe for ML training'''
              df_1=pd.DataFrame(self.measurements)
              df_1['camber']=self.camber
              df_1['position']  = self.position
              df_1['thickness'] = self.thickness
              return df_1
        

# child class-2 ----->
class CFDairfoil(baseairfoil):
      """
    Extends BaseAirfoil with CFD simulation data.
    Different child, same parent.
    Like having two different model architectures
    both inheriting from nn.Module.
    """
      def __init__(self, camber, position, thickness,
                 mesh_cells, turbulence_model):
           super().__init__(camber, position, thickness)
           self.mesh_cells       = mesh_cells
           self.turbulence_model = turbulence_model
           self.cfd_results      = {} 

      def add_cfd_result(self, aoa, Cl, Cd, Cm, iterations):
          """Store one CFD case result."""
          self.cfd_results[aoa] = {
            'Cl': Cl, 'Cd': Cd, 'Cm': Cm,
            'iterations': iterations
        } 
          
      def mesh_quality_check(self):
               """Minimum cells for reliable SST k-w results."""
               if self.mesh_cells < 80000:
                 return "WARNING: mesh too coarse. Minimum 80,000 cells."
               elif self.mesh_cells < 120000:
                return "ACCEPTABLE: results reliable but not optimal."
               else:
                return "GOOD: mesh quality sufficient."
    
        
      def compare_with_experiment(self, wind_tunnel_airfoil,aoa):
          """
         Compare CFD result vs experimental at same AoA.
        Takes a WindTunnelAirfoil object as input.
          """
          if aoa not in self.cfd_results:
            return f"No CFD result at AoA={aoa}"
        
          cfd = self.cfd_results[aoa]
          exp = next((m for m in wind_tunnel_airfoil.measurements
                   if m['aoa'] == aoa), None)
        
          if exp is None:
            return f"No experimental data at AoA={aoa}"
        
          cl_error = abs(cfd['Cl'] - exp['Cl']) / abs(exp['Cl']) * 100
          cd_error = abs(cfd['Cd'] - exp['Cd']) / abs(exp['Cd']) * 100
          
          print(f"\nValidation at AoA = {aoa}°:")
          print(f"  Cl: CFD={cfd['Cl']:.3f}  Exp={exp['Cl']:.3f}  "
              f"Error={cl_error:.1f}%")
          print(f"  Cd: CFD={cfd['Cd']:.4f}  Exp={exp['Cd']:.4f}  "
              f"Error={cd_error:.1f}%")
        
         
          return cl_error < 5.0 
      

    #   tasl-1---->
wt_airfoil=windtunnelairfoil(camber=2,position=4,thickness=12,
                            reynolds_number=3e6,tunnel_speed=50 )

# Format: (aoa, Cl, Cd, Cm)
raw_data = [
    (-2, 0.05, 0.011, -0.02),
    ( 0, 0.25, 0.011, -0.03),
    ( 4, 0.65, 0.013, -0.04),
    ( 8, 1.02, 0.018, -0.06),
    (12, 1.20, 0.028, -0.08),
    (14, 1.15, 0.055, -0.10)
]

# The asterisk (*) unpacks the tuple directly into your function's arguments!
for row in raw_data:
    wt_airfoil.add_measurement(*row)

wt_airfoil.geometry_summary()

print(f"Maximum Lift Coefficient (Cl) : {wt_airfoil.max_Cl()}")
print(f"Stall Angle (AoA)             : {wt_airfoil.stall_angle()}°")

# creating the CFD object(example:- NACA2412)
cfd_airfoil=CFDairfoil(camber=2,position=4,thickness=12,
                       mesh_cells=100000,turbulence_model="SST K-w")

cfd_airfoil.add_cfd_result(aoa=4,Cl=0.648,Cd=0.0128,Cm=0.041,iterations=250)
cfd_airfoil.add_cfd_result(aoa=8,Cl=1.008,Cd=0.0176,Cm=0.059,iterations=310)

print(f"mesh status:{cfd_airfoil.mesh_quality_check()}")

cfd_airfoil.geometry_summary()

is_valid_4=cfd_airfoil.compare_with_experiment(wind_tunnel_airfoil=wt_airfoil,aoa=4)
if is_valid_4:
     print("verdict: PASS! CFD is within 5% wind tunnel data")
else:
     print("verdict: FAIL! CFD requires mesh refinement or turbulence model tuning")



wt_dataset=wt_airfoil.to_dataframe()
print(wt_dataset)

# EXERCISE 4: Class methods, static methods, __repr__
# Goal: understand difference between:
#       regular method  → uses self (per object)
#       class method    → uses cls  (per class, shared)
#       static method   → uses neither (utility function)

class MLDataset:
     """
    Manages the airfoil surrogate training dataset.
    Tracks how many datasets created (class-level).
    Provides utilities for data validation (static methods).
    """
     dataset_count=0

     def __init__(self,name,n_Samples,feature_cols,target_cols):
          self.name=name
          self.n_samples=n_Samples
          self.feature_cols=feature_cols
          self.target_cols=target_cols
          self.is_scaled=False
          self.is_split=False

        #increment shared counter every time new dataset created
          MLDataset.dataset_count+=1
          self.dataset_id=MLDataset.dataset_count
     def __repr__(self):
          """
        __repr__ = unambiguous string representation.
        Shows when you type object name in console without print().
        More technical than __str__.
        """
          return(f"MLDataset(id={self.dataset_id},"
                 f"name={self.name}, "
                 f"n={self.n_samples}, "
                 f"features={self.feature_cols},"
                 f"targets={self.target_cols}")
     def __str__(self):
          '''human readable version for print().'''
          status=[]
          if self.is_scaled:status.append("scaled")
          if self.is_split: status.append("split")
          status_str=", ".join(status) if status else "raw"  
          return f"Dataset ' {self.name} ' [ {self.n_samples} samples | {status_str}]"
     

     def mark_Scaled(self):
          '''regular method-modifies THIS object's state.'''
          self.is_scaled=True
          print(f"Dataset '{self.name}' marked as scaled. ")

     def mark_split(self,train_size,test_size):
          '''Mark dataset as split. store split sizes. '''
          self.is_split=True
          self.train_size=train_size
          self.test_size=test_size


#Class Method----->
     @classmethod
     def how_many_datasets(cls):
      """
        cls = the CLASS itself (not one object).
        Access class-level variables here.
        Called on class, not instance:
        MLDataset.how_many_datasets"""
      print(f"total datasets created: {cls.dataset_count}")

     @classmethod
     def create_airfoil_dataset(cls,n_Samples):
      """
        Alternative constructor — creates dataset with
        airfoil-specific columns already set.
        Shortcut so you don't repeat column names every time.
        """
      return cls(
          name =f"airfoil_fluent_{n_Samples}cases",
          n_Samples=n_Samples,
          feature_cols=['camber','position','thickness','AoA'],
          target_cols=['Cl','Cd','Cm']
     )


#STATIC METHOD------->
     @staticmethod
     def minimum_samples_needed(n_features,n_outputs):
      """
        Utility: estimate minimum training samples needed.
        Rule of thumb: 50× (features + outputs) for neural network.
        No self. No cls. Pure calculation utility.
        """
      return 50*(n_features + n_outputs)

     @staticmethod
     def train_test_recommendation(n_Samples):
      """
        Recommend train/test split based on dataset size.
        No self or cls needed — pure logic.
        """
      if n_Samples<100:
          return "70/30 split or k-fold cross validation"
      elif n_Samples<500:
          return "80/20 split"
      elif n_Samples<5000:
          return "90/10 split"
      else:
          return "95/5 split"
     
     @staticmethod
     def validate_columns(df,required_cols):
      """
        Check if dataframe has all required columns.
        Returns missing columns list.
        """
      missing=[c for c in required_cols if c not in df.columns]
      return missing


ds1=MLDataset.create_airfoil_dataset(80)
ds2=MLDataset.create_airfoil_dataset(200)
print(ds1)
print(ds2)
print(repr(ds1))
MLDataset.how_many_datasets()

ds1.mark_Scaled()
ds1.mark_split(64,16)
print(ds1)

print(f"recommendation: {MLDataset.minimum_samples_needed(4,3)}")
print(f"recommendation: {MLDataset.train_test_recommendation(200)}")

fake_df=pd.DataFrame(
     np.random.randn(10,5),
     columns=['camber','pos','thick','AoA','Cl']
)
required = ['camber','pos','thick','AoA','Cl','Cd','Cm']
print(f"missing:{ MLDataset.validate_columns(fake_df,required)}")

# ══════════════════════════════════════════════════════════
# T5: Explain in comments — when to use each:
# ══════════════════════════════════════════════════════════

# Regular method  → 
#   WHEN TO USE: Use this 90% of the time. Use it when the function needs 
#   to read or modify the unique data of a specific object (instance).
#   CLUE: It always takes `self` as the first argument.
#   THE PRONOUN RULE: It says "MY". ("I need to change MY dataset status.")
#   EXAMPLE: mark_scaled(self) needs to change the `is_scaled` flag for ONE specific dataset.

# Class method    →
#   WHEN TO USE: Use this when the function needs to interact with the blueprint
#   itself, rather than a specific object. Perfect for "Alternative Constructors" 
#   (shortcuts to build objects with preset data) or for tracking global/shared variables.
#   CLUE: It uses the @classmethod decorator and takes `cls` as the first argument.
#   THE PRONOUN RULE: It says "OUR". ("How many datasets has OUR factory built?")
#   EXAMPLE: create_airfoil_dataset(cls) uses the blueprint (cls) to build a new object.

# Static method   →
#   WHEN TO USE: Use this for pure utility or math functions. If the function 
#   doesn't care about the object's unique data AND doesn't care about the class 
#   blueprint, but logically makes sense to group with this class, make it static.
#   CLUE: It uses the @staticmethod decorator and takes NEITHER `self` nor `cls`.
#   THE PRONOUN RULE: It says "THE". ("Just give me THE numbers and I'll do the math.")
#   EXAMPLE: minimum_samples_needed(features, outputs) just calculates a number.

#------------------------------------------------------------------------------------------------>>

# EXERCISE 5: Composition — objects inside objects
# Goal: build complex systems from simple classes
#       PyTorch uses this: model contains layers,
#       layers contain weights — all objects inside objects


class Layer:
     """
     represents one neural network layer.
     mirrors what nn.linear does internally.
     """
     def __init__(self,in_features,out_features,activation='ReLu'):
          self.in_features=in_features
          self.out_features=out_features
          self.activation=activation
      # weights: matrix of shape(out,in)
      # intialized small random-same as PyTorch default
          self.weights=np.random.randn(out_features,in_features)
          self.biases=np.zeros(out_features)

      # Track parameter count
          self.n_weights=in_features * out_features
          self.n_biases=out_features

     def parameter_Count(self):
          return self.n_weights + self.n_biases

     def __str__(self):
          return (f"layer({self.in_features}-->{self.out_features},"
                  f"activation={self.activation},"
                  f"params={self.parameter_Count()}"
                  )   


class NeuralNetworkManual:
     """
     Manual neural network built from layer objects.
     composition: neuralnetwork HAS layers(objects inside objects)
     mirrors pytorch nn.sequential:
     self.network=nn.sequential(
     nn.linear(4,64),<---layer object inside network object
     nn.linear(64,128),<---another layer object
     )
     """

     def __init__(self,name):
          self.name=name
          self.layers=[]   #list of layer objects - composition
     def add_layer(self,in_Features,out_Features,activation='ReLu'):
          """create layer object and store inside this network"""
          layer=Layer(in_Features,out_Features,activation)
          self.layers.append(layer)
          return self  #return self allows method chaining
     def total_parameters(Self):
          """sum parameters across all layer objects. """
          return sum(layer.parameter_Count() for layer in Self.layers)
     def architercture_summary(self):
          """print full architecture-mirror Pytorch's print(model). """
          print(f"\nModel: {self.name}")
          print("--" *45)
          for i,layer in enumerate(self.layers):
               print(f" layer {i+1}: {layer}")
          print("--"*45)
          print(f" total parameters: {self.total_parameters():,}")

     def layer_shapes(self):
          """return list of (in,out) shapes per layer."""
          return [{L.in_features,L.out_features} for L in self.layers]
     def validate_architecture(self):
          """
          check layers connect properly.
          output of layer N must match input of layer N+1
          """
          errors=[]
          for i in range(len(self.layers)-1):
               out=self.layers[i].out_features
               inp=self.layers[i+1].in_features
               if out!=inp:
                    errors.append(
                         f"layers {i+1}-->{i+2}: "
                         f"output {out} not equal to input {inp}"
                    )
               if errors:
                    print("ARCHITECTURE ERRORS:")
                    for e in errors: print(f" x{e}")
                    return False
               print("Architecture valid ")
               return True
          
# TASKS--->
model=NeuralNetworkManual("airfoilsurrogate_Cl_Cm_Cm")
model.add_layer(6,64,activation='ReLu')\
      .add_layer(64,128,activation='ReLu')\
      .add_layer(128,64,activation='ReLu')\
      .add_layer(64,3,activation='ReLu')
model.architercture_summary()
print('/n')

model_1=NeuralNetworkManual("Cp distribution")
model.add_layer(4,128,activation='ReLu')\
      .add_layer(128,256,activation='ReLu')\
      .add_layer(256,256,activation='ReLu')\
      .add_layer(256,128,activation='ReLu')\
      .add_layer(128,100,activation='None')
model.architercture_summary()

broken=NeuralNetworkManual('brokenModel')
broken.add_layer(4,64,activation='ReLu')\
      .add_layer(32,128,activation='ReLu')
broken.validate_architecture()

print(f"layers for both models: {model.layer_shapes()} || {model.layer_shapes()}")

# --------------------------------------------------------------------------------------------->
# EXERCISE 6: Complete system — all OOP concepts together
# Goal: build mini training manager class
#       mirrors exactly how PyTorch training works
#       After this — PyTorch syntax is just shorter version

class trainingHistory:
     """Tracks Loss valuse during training. One  job only."""

     def __init__(self):
          self.train_losses=[]
          self.val_losses=[]
          self.epochs=[]

     def record(self,epoch,train_loss,val_loss):
          self.epochs.append(epoch)
          self.train_losses.append(train_loss)
          self.val_losses.append(val_loss)

     def best_epoch(self):
          """Epoch with lowest validation loss."""
          if not self.val_losses:
               return None
          best_idx=np.argmin(self.val_losses)
          return self.epochs[best_idx], self.val_losses[best_idx]
     
     def is_overfitting(self, window=10):
          """overfitting detected if val loss increasing 
          while train loss still decreasing.
          check last N epochs.
          """
          if len(self.val_losses)<window:
               return False
          recent_Val=self.val_losses[-window:]
          recent_train=self.train_losses[-window:]
          val_increasing=recent_Val[-1] >recent_Val[0]
          train_dcreasing=recent_train[-1] < recent_train[0]
          return val_increasing and train_dcreasing

     def plot(self):
          import matplotlib.pyplot as plt
          plt.figure(figsize=(10,4))
          plt.plot(self.epochs,self.train_losses,label='train loss')
          plt.plot(self.epochs,self.val_losses,label='val loss')
          plt.xlabel('Epoch')
          plt.ylabel('MSE loss')
          plt.title('training History')
          plt.legend()
          plt.yscale('log')  #log scale shows convergence clearly
          plt.grid(True,alpha=0.4)
          plt.tight_layout()
          plt.show()

class ModelTrainer:
     """
     manages full training loop for one model.
     composition:HAS a trainingHistory object inside.

     this is the manual version of what PyTorch does.
     after understanding this--Pytorch training loop
     is just 4 lines doing the same thing.
     """
     # class variable -- track all trainers created
     experiments=[]

     def __init__(self,model_name,learning_rate=0.001):
          self.model_name=model_name
          self.learning_rate=learning_rate
          self.history=trainingHistory() #composition
          self.is_trained=False
          self.final_r2=None

          ModelTrainer.experiments.append(model_name)

     def __str__(self):
          status="trained" if self.is_trained else "untrained"
          return(f"Trainer('{self.model_name}' | "
                 f"lr={self.learning_rate} | {status})")
     
     def simulate_training(self,n_epochs=500,
                           initial_loss=1.0,
                           target_loss=0.01,
                           add_overfit=False):
          """
          simulates training loss curve.
          Real Pytorch training replaces this with actual gradient steps.
          but the STRUCTURE -- epochs loop,record history --identical.
          """
          print(f"\ntraining '{self.model_name}'...")
          print(f"Epochs: {n_epochs} | LR: {self.learning_rate}")
          print("--"*40)

          for epoch in range(n_epochs):
               #simulate exponential loss decay (what good training looks like)
               decay=epoch/n_epochs
               train_loss=initial_loss*np.exp(-5*decay) + target_loss
               train_loss=np.random.normal(0,0.002)  #noise

               if add_overfit and epoch >n_epochs*0.6:
                    #Val loss starts rising after 60% of training
                    overfit_factor=1+2*(decay-0.6)
                    val_loss=train_loss*overfit_factor

               else:
                    val_loss=train_loss*1.05+np.random.normal(0,0.003)

                 # Record every epoch — same as logging in real training
               self.history.record(epoch,max(train_loss,0),max(val_loss,0))
               if epoch % 100==0:
                    print(f"Epoch {epoch:4d} |"
                          f"Train: {target_loss:.4f} | "
                          f"Val: {val_loss:.4f}")
                    
               self.is_trained=True
               print("--"*40)
               print(f"training complete.")


               best_e,best_loss=self.history.best_epoch()
               print(f"best epoch: {best_e}  | val loss: {best_loss} ")

               if self.history.is_overfitting():
                    print("Warning: overfitting detected. reduce model size")
               else:
                    print("No overfitting detected.")

     @classmethod
     def list_experiment(cls):
          """show all training runs created."""
          print(f"\nALL experiments ({len(cls.experiments)}): ")
          for i,name in enumerate(cls.experiments):
               print(f" {i+1}, {name}")

     @staticmethod
     def learning_rate_guide(model_size):
           """recommend learning rae based on model size. """
           if model_size<1000:
               return 0.01
           elif model_size<50000:
                return 0.001
           else:
               return 0.0001
               

#TASK-1
trainer_1=ModelTrainer("airfoilsurrogate_Cl_Cd_Cm",learning_rate=0.001)
trainer_2=ModelTrainer("airfoilsurrogate_Cp",learning_rate=0.0005)
print(trainer_1)
print(trainer_2)
ModelTrainer.list_experiment()


#Task-2
trainer_1.simulate_training(n_epochs=500)
trainer_2.simulate_training(n_epochs=500, add_overfit=True)
trainer_1.history.plot()
trainer_2.history.plot()






               

      
               
     



     
          





     
     
     
     
     
          
            
      




      


    
           
          


       
        

       

          
          