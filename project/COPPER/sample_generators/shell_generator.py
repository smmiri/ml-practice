import shutil
for i in range(8,201):
    file_name = 'crest-fy2030-5-ct' +str(i)+'.sh'
    #print(file_name)
    shutil.copy2('crest-fy2030-5-ct7.sh', file_name)
    
    with open(file_name, 'rt') as shell_file:
        shell_file_lines = shell_file.readlines()

    #print (python_file_lines)

    shell_file_lines[5] = '#SBATCH --output=Solver-sr5-fy2030-ct'+str(i)+'.out\n'
    shell_file_lines[11] = 'python3 /project/6049269/smmiri/CRESTv4/CRESTver4optimized5fy2030ct'+str(i)+'.py\n'
    with open(file_name, 'w') as shell_file:
        shell_file.writelines( python_file_lines )
        