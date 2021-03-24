import shutil
import os

#os.chdir('C://Users/smoha/documents/git/ml_progress/project/copper/sample_generators/samples_shell')
os.chdir('/mnt/c/Users/smoha/documents/git/ml_progress/project/copper/sample_generators/samples_shell')

for i in range(0,3000):
    file_name = 'COPPER5_' +str(i)+'.sh'
    #print(file_name)
    shutil.copy2('COPPER5.sh', file_name)
    
    with open(file_name, 'rt') as shell_file:
        shell_file_lines = shell_file.readlines()

    #print (python_file_lines)

    shell_file_lines[2] = '#SBATCH --output=COPPER5_' +str(i) +'.out \n'
    shell_file_lines[8] = 'python3 /scratch/smmiri/COPPER5_' +str(i) +'.py \n'
    with open(file_name, 'w') as shell_file:
        shell_file.writelines( shell_file_lines )
        