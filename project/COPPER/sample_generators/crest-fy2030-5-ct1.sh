#!/bin/bash
#SBATCH --time=11:59:59
#SBATCH --cpus-per-task=9
#SBATCH --mem=40000
#SBATCH --account=rrg-mcpher16
#SBATCH --output=Solver-sr5-fy2030-ct1.out

export PATH=$PATH:/home/smmiri/CPLEX/cplex/bin/x86-64_linux/   
export PATH=$PATH:/home/smmiri/CPLEX/cplex/python/3.7/x86-64_linux/  

source /home/smmiri/ENV_SILVER/bin/activate
python3 /project/6049269/smmiri/CRESTv4/CRESTver4optimized5fy2030ct1.py
