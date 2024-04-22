#!/bin/bash
#
#SBATCH --job-name=performance
#SBATCH --error=%x-%j.e
#SBATCH --output=%x-%j.o
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --account=ap_calcua_staff

. ./env.sh

# srun python ../../tests/dpvssp/test_dpvssp.py
srun python ../../scripts/time_heat_equation.py