import subprocess
import numpy as np
import os

XFOIL_PATH = r'D:\Surrogate_Models\xfoil.exe'

def run_xfoil(naca_code, aoa, reynolds=3e6, max_iter=100):
    """Run Xfoil for one naca airfoil at one AoA"""

    # Force both programs to use the exact absolute path
    polar_filename = f'polar_temp_{naca_code}_{aoa:.1f}.txt'
    project_folder = r"D:\Surrogate_Models"
    polar_files_path = os.path.join(project_folder, polar_filename)

    # Build XFOIL command sequence
    commands = (
        f"NACA {naca_code}\n"     # load airfoil
        f"OPER\n"                 # enter operating menu
        f"VISC {reynolds:.0f}\n"  # set Reynolds number 
        f"ITER {max_iter}\n"      # set max iterations
        f"PACC\n"                 # start polar accumulation
        f"{polar_filename}\n"     # output file name
        f"\n"                     # no reject file 
        f"ALFA {aoa:.2f}\n"       # run at this AoA
        f"PACC\n"                 # stop accumulation
        f"\n"                     
        f"QUIT\n"                 # exit XFOIL 
    )
    
    try:
        proc = subprocess.run(
            [XFOIL_PATH],
            input=commands,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_folder
        )

        # Parse polar output file
        result = parse_polar_file(polar_files_path, aoa)

        if os.path.exists(polar_files_path):
            os.remove(polar_files_path)

        return result
    
    except subprocess.TimeoutExpired:
        if os.path.exists(polar_files_path):
            os.remove(polar_files_path)
        return None
        
    except Exception as e:
        if os.path.exists(polar_files_path): # Fixed typo here
            os.remove(polar_files_path)
        return None
    
def parse_polar_file(filepath, target_aoa):
    if not os.path.exists(filepath):
        return None
    
    try:
        results = []
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # finding data lines -- skip header
        data_started = False
        for line in lines:
            line = line.strip()

            if line.startswith('------'):
                data_started = True
                continue

            if data_started and line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        alpha = float(parts[0])
                        cl = float(parts[1])
                        cd = float(parts[2])
                        cm = float(parts[4])
                        # Fixed the order here to match the unpacking below
                        results.append((alpha, cl, cd, cm)) 
                    except ValueError:
                        continue
                        
        if not results:
            return None
            
        # finding closest AoA to target
        best = min(results, key=lambda m: abs(m[0] - target_aoa))
        alpha, cl, cd, cm = best

        # validating physical bounds
        if (-0.5 < cl < 2.0 and 0.003 < cd < 0.15):
            return {
                'Cl': round(cl, 5),
                'Cd': round(cd, 6),
                'Cm': round(cm, 5)
            }
        return None
        
    except Exception:
        return None
    
# --- TEST -- single case first --- //

result = run_xfoil('2412', 4.0, reynolds=3e6)

if result:
    print(f"Cl = {result['Cl']:.4f} ")
    print(f"Cd = {result['Cd']:.5f} ")
    print(f"Cm = {result['Cm']:.4f}  ")
    print("\nXFOIL working correctly ✓")
else:
    print("XFOIL failed — check:")
    print("1. xfoil.exe in project folder?")
    print("2. XFOIL_PATH correct?")
    print("3. Run xfoil.exe manually in terminal — does it open?")
    



