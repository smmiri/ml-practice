#!/bin/bash

export PATH=$PATH:/home/smmiri/CPLEX/cplex/bin/x86-64_linux/   
export PATH=$PATH:/home/smmiri/CPLEX/cplex/python/3.7/x86-64_linux/  

source /home/smmiri/ENV_SILVER/bin/activate
python3 /scratch/smmiri/COPPER5.1.py
