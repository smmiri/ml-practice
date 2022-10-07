#!/bin/bash

#SBATCH --output=COPPER5_84.out 

export PATH=$PATH:/home/smmiri/CPLEX/cplex/bin/x86-64_linux/   
export PATH=$PATH:/home/smmiri/CPLEX/cplex/python/3.7/x86-64_linux/  

source /home/smmiri/copper-env/bin/activate
python /scratch/smmiri/COPPER5.1_84.py 
