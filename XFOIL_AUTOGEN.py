from scipy.stats import qmc
import numpy as np
import pandas as pd
from XFOIL import run_xfoil


def generate_dataset(
        n_cases=5000,
        reynolds=3e6,
        output_file='Xfoil_dataset_5k.csv'
):
    print(f"Generating {n_cases} LHS cases---")

    #--LHS sampling------>>
    sampler = qmc.LatinHypercube(d=4, optimization='random-cd', seed=42)
    lhs_sampling = sampler.random(n=n_cases)
    #scaling to phyiscal ranges---//
    camber_arr=lhs_sampling[:,0]*9
    position_arr=lhs_sampling[:,1]*8+1
    thickness_arr = lhs_sampling[:, 2] * 18 + 6     # 6–24
    aoa_arr= lhs_sampling[:, 3] * 20 - 4 

    results=[]
    failed=0


    for idx in range(n_cases):


        #round to valid NACA integers
        m=int(round(camber_arr[idx]))
        p=int(round(position_arr[idx]))
        t = int(round(thickness_arr[idx]))  
        a = round(aoa_arr[idx], 2)  

        #--skipping invalid combinations
        if m==0 and p!=0:
            p=0      #naca 00xx must have p=0
        if p==0 and m!=0:
            p=4    #default pos for cambered 

        naca_code=f"{m}{p}{t:02d}"


        result=run_xfoil(naca_code=naca_code,aoa=a,
                         reynolds=reynolds )
        
        if result:
            results.append({
                'camber_pct':m,
                'camber_pos':p,
                'thickness_pct': t,
                'AoA_deg'      : a,
                'Cl'           : result['Cl'],
                'Cd'           : result['Cd'],
                'Cm'           : result['Cm'],
                'naca_code'    : naca_code
            })

        else:
            failed+=1

        #progress every 50 cases
        if (idx+1)%50==0:
            success_rate=len(results)/(idx+1)*100
            print(
                f"Case {idx+1:4d}/{n_cases} | "
                  f"Collected: {len(results):4d} | "
                  f"Failed: {failed:3d} | "
                  f"Rate: {success_rate:.0f}%"
            )

    df=pd.DataFrame(results)
    df.to_csv(output_file,index=False)

    print(f"Total cases  : {n_cases}")
    print(f"Successful   : {len(results)}")
    print(f"Failed       : {failed}")
    print(f"Success rate : {len(results)/n_cases*100:.1f}%")
    print(f"Saved to     : {output_file}")
    print(f"\nDataset summary:")
    print(df[['Cl','Cd','Cm']].describe().round(4))
    
    return df



df=generate_dataset(
    n_cases=5000,
    reynolds=3e6,
    output_file='Xfoil_dataset_5k.csv'
 )
    