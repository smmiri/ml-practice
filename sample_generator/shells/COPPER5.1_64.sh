#!/bin/bash

#SBATCH --output=COPPER5_64.out 

export PATH=$PATH:/home/smmiri/CPLEX/cplex/bin/x86-64_linux/   
export PATH=$PATH:/home/smmiri/CPLEX/cplex/python/3.7/x86-64_linux/  

source /home/smmiri/copper-env/bin/activate
python /scratch/smmiri/COPPER5.1_64.py 
