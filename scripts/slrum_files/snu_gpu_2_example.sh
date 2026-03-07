#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2      # Cores per node
#SBATCH --partition=snu-gpu2     # Partition Name
##
#SBATCH --job-name=snu_gpu_2_example
#SBATCH --time=24-00:00          # Runtime: Day-HH:MM
#SBATCH -o test.%N.%j.out         # STDOUT
#SBATCH -e test.%N.%j.err         # STDERR
#
#SBATCH --gres=gpu:l40s:1
##

hostname
date
module purge

StartTime=$(date +%s)
cd $SLURM_SUBMIT_DIR

python test.py
EndTime=$(date +%s)

echo Run time
echo $StartTime $EndTime | awk '{print $2-$1}' | xargs -I{} echo {} sec
echo $StartTime $EndTime | awk '{print ($2-$1)/60}' | xargs -I{} echo {} min
echo $StartTime $EndTime | awk '{print ($2-$1)/3600}' | xargs -I{} echo {} hour
