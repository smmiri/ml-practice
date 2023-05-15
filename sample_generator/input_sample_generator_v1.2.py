import os
import shutil
from scipy.stats.qmc import LatinHypercube as LH

# Create a Latin Hypercube sampler
engine = LH(1)

import pandas as pd

"""
COPPER Input vars: 1) carbon tax 2) capital cost 3) demand growth (1~2) 4) technology toggles (nuclear on/off, 
hydro on/off, ccs on/off, transmission on/off)
two combinations of technology toggles: first 1) all on and 2) all off except transmission
"""

CCS = False

provinces_full = {
            "British Columbia": 4497392,
            "Alberta": 3584417,
            "Manitoba": 1234225,
            "New Brunswick": 717014,
            "Newfoundland and Labrador": 532691,
            "Nova Scotia": 944301,
            "Ontario": 12557108,
            "Quebec": 8190949,
            "Saskatchewan": 1078184,
            "Prince Edward Island": 140204
            }

pds=['2025','2030','2035','2040','2045','2050']



total_pop = sum(provinces_full.values())
pop_growth = []

path = os.getcwd()
os.makedirs(path+'/annual_growths',exist_ok=True)
os.makedirs(path+'/capital_costs',exist_ok=True)
os.makedirs(path+'/shells',exist_ok=True)
os.makedirs(path + '/scripts', exist_ok=True)

gendata = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0)
gendata_fix = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0, index_col=0)
if CCS:
    gendata = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0)
    gendata_fix = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0, index_col=0)

else:
    gendata = pd.read_excel (r'Generation_type_data.xlsx',header=0)
    gendata_fix = pd.read_excel (r'Generation_type_data.xlsx',header=0, index_col=0)
    
capitalcost=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['capitalcost'])))
annualgrowth = pd.read_csv(r'annual_growth.csv', index_col=0, dtype={'2025': float, '2030': float, '2035': float, '2040': float, '2045': float, '2050': float})

gendata = gendata.drop(index=8)

ranges = pd.read_excel('Capital cost ranges.xlsx', sheet_name='Estimates and range suggestions', index_col=0, header=6)
ranges = ranges.loc[:, 'Min': 'Max']

max_th = 1.1
min_th = 0.9
max_vre = 1
min_vre = 0.6
#max_fp = 2
#min_fp = 0.9

sample_size = 2000

for type in gendata.iloc[:]['Type']:
    locals()["r_cap_" + type] = pd.DataFrame(engine.random(sample_size))

#for PDS in pds:
#    locals()["r_ctax_"+ PDS]= uniform(0.5, 2, size=sample_size)

r_ctax = pd.DataFrame(100*engine.random(sample_size))
r_growth = pd.DataFrame(1+1.5*(engine.random(sample_size)))

for i in range(0, sample_size):
    
    for type in gendata.iloc[:]['Type']:
        gendata_fix.loc[type, 'capitalcost'] = ranges.at[type, 'Min'] + locals()["r_cap_" + type].at[i,0] * (ranges.at[type, 'Max'] - ranges.at[type, 'Min'])

    test = r_growth.at[i,0]
    annualgrowth_new = annualgrowth*r_growth.at[i,0]
    provinces_new_pop = {}

    for province in provinces_full.keys():
        provinces_new_pop[province]= ((1 + annualgrowth_new.loc[province, '2030']) ** 12) * (
                (1 + annualgrowth_new.loc[province, '2040']) ** 10) * ((1 + annualgrowth_new.loc[province, '2050']
                                                                        ) ** 10)* provinces_full[province]

    pop_growth.append(sum(provinces_new_pop.values()) / total_pop)

    os.chdir(path+'/annual_growths')
    annualgrowth_new.to_csv('annual_growth_{}.csv'.format(i))
    os.chdir(path)
    os.chdir(path + '/capital_costs')
    if CCS:
        gendata_fix.to_excel('Generation_type_data_SMR_CCS_{}.xlsx'.format(i),index=True)
    else:
        gendata_fix.to_excel('Generation_type_data_{}.xlsx'.format(i),index=True)
    
    os.chdir(path)

    shutil.copy2('COPPER7_3.py', 'scripts/COPPER7.3.py')
    os.chdir(path + '/scripts')
    file_name = f'COPPER7.3_{i}.py'
    # print(file_name)
    shutil.copy2('COPPER7.3.py', file_name)

    with open(file_name, 'r') as python_file:
        python_file_lines = python_file.readlines()

    ctax = {
        "2025": 95,
        "2030": round(95 + 1.5 * r_ctax.at[i, 0],3),
        "2035": round(95 + 2.5 * r_ctax.at[i, 0],3),
        "2040": round(95 + 3.5 * r_ctax.at[i, 0],3),
        "2045": round(95 + 4.5 * r_ctax.at[i, 0],3),
        "2050": round(95 + 5.5 * r_ctax.at[i, 0],3)
    }
    python_file_lines[33] = 'ctax={' + f"'2025':{ctax['2025']},'2030':{ctax['2030']},'2035':{ctax['2035']},'2040':{ctax['2040']},'2045':{ctax['2045']},'2050':{ctax['2050']}" + '}\n'
    #python_file_lines[216] = f"    gendata = pd.read_excel(r'Generation_type_data_SMR_CCS_{i}.xlsx',header=0) \n"
    python_file_lines[218] = f"    gendata = pd.read_excel(r'Generation_type_data_{i}.xlsx',header=0) \n"
    python_file_lines[395] = f"demand_growth = pd.read_csv(r'annual_growth_{i}.csv',header=0,index_col=0) \n"
    python_file_lines[1383] = f"folder_name=str({i}) + '_outputs'+'_ct'+str(ctax)+'_rd'+str(len(rundays)) + '_pds'+str(len(pds)) \n"

    with open(file_name, 'w') as python_file:
        python_file.writelines(python_file_lines)
    
    os.chdir(path)

    shutil.copy2('COPPER7.3.sh', 'shells/COPPER7.3.sh')
    os.chdir(path + '/shells')
    file_name = 'COPPER7.3_' + str(i) + '.sh'
    shutil.copy2('COPPER7.3.sh', file_name)

    with open(file_name, 'rt') as shell_file:
        shell_file_lines = shell_file.readlines()
    

    shell_file_lines[2] = '#SBATCH --output=COPPER7.3_' + str(i) + '.out \n'
    shell_file_lines[8] = 'python /scratch/smmiri/COPPER7.3_' + str(i) + '.py \n'
    with open(file_name, 'w') as shell_file:
        shell_file.writelines(shell_file_lines)
    os.chdir(path)
    
pd.DataFrame(pop_growth).to_csv('pop_growth.csv', index=True, header=None)