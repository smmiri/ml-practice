import shutil
import os

path = os.getcwd()
os.makedirs(path+'/COPPER6',exist_ok=True)
os.chdir(path+'/COPPER6')


for i in range(0,100):
    file_name = 'COPPER5_' +str(i)+'.py'
    #print(file_name)
    shutil.copy2('COPPER5.1.py', file_name)
    
    with open(file_name, 'r') as python_file:
        python_file_lines = python_file.readlines()

    #print (python_file_lines)

    #python_file_lines[83] = "configuration = pd.read_excel (r'COPPER_configuration_"+str(i)+".xlsx',header=0)"+'\n'
    python_file_lines[199] = "    gendata = pd.read_excel (r'Generation_type_data_"+str(i)+".xlsx',header=0 )"+'\n'
    python_file_lines[366] = "demand_growth = pd.read_csv(r'annual_growth_{}.csv',header=0,index_col=0)".format(i)+'\n'
    python_file_lines[1249] = "folder_name= str({})+'_outputs'+'_ct'+str(ctax['2030'])+'_'+str(ctax['2040'])+'_'+str(ctax['2050'])+'_rd'+str(len(rundays))+'_pds'+str(len(pds))".format(i) +'\n'

    with open(file_name, 'w') as python_file:
        python_file.writelines( python_file_lines )




