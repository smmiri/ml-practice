import shutil
for i in range(101,201):
    file_name = 'CRESTver4optimized5fy2030ct' +str(i)+'.py'
    #print(file_name)
    shutil.copy2('CRESTver4optimized5fy2030ct100.py', file_name)
    
    with open(file_name, 'r') as python_file:
        python_file_lines = python_file.readlines()

    #print (python_file_lines)

    python_file_lines[27] = 'ctax='+str(i)+'\n'

    with open(file_name, 'w') as python_file:
        python_file.writelines( python_file_lines )
        



