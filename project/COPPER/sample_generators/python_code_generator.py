import shutil
import os

#os.chdir('C://Users/smoha/documents/git/ml_progress/project/copper/sample_generators/samples_models')
os.chdir('/mnt/c/Users/smoha/documents/git/ml_progress/project/copper/sample_generators/samples_models')

for i in range(0,3000):
    file_name = 'COPPER5_' +str(i)+'.py'
    #print(file_name)
    shutil.copy2('COPPER5.py', file_name)
    
    with open(file_name, 'r') as python_file:
        python_file_lines = python_file.readlines()

    #print (python_file_lines)

    python_file_lines[38] = "configuration = pd.read_excel (r'COPPER_configuration_"+str(i)+".xlsx',header=0)"+'\n'
    python_file_lines[122] = "    gendata = pd.read_excel (r'Generation_type_data_SMR_CCS_"+str(i)+".xlsx',header=0 )"+'\n'
    python_file_lines[1127] = "folder_name= str("+str(i) +") + '_outputs'+'_ct'+str(ctax)+'_rd'+str(len(rundays))+'_pds'+str(len(pds))" +'\n'

    with open(file_name, 'w') as python_file:
        python_file.writelines( python_file_lines )




