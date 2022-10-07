#!/bin/bash

export PATH=$PATH:/home/smmiri/CPLEX/cplex/bin/x86-64_linux/
export PATH=$PATH:/home/smmiri/CPLEX/cplex/python/3.7/x86-64_linux/  

source /home/smmiri/copper-env/bin/activate
python /scratch/smmiri/COPPER6.2.py
