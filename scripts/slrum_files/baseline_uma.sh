#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=snu-gpu2
##
#SBATCH --job-name="baseline_uma"
#SBATCH --time=07-00:00
#SBATCH -o stdout_baseline_uma.%N.%j.out
#SBATCH -e stderr_baseline_uma.%N.%j.err
#
#SBATCH --gres=gpu:l40s:1
##
hostname
date
module purge
StartTime=$(date +%s)
cd $SLURM_SUBMIT_DIR

conda activate fairchem
python /DATA/user_scratch/pn50212/2024/12_AtomAttention/scripts/experiments/mlip_baseline_uma.py

EndTime=$(date +%s)
echo "Run time:"
echo $StartTime $EndTime | awk '{print $2-$1}' | xargs -I{} echo {} sec
echo $StartTime $EndTime | awk '{print ($2-$1)/60}' | xargs -I{} echo {} min
echo $StartTime $EndTime | awk '{print ($2-$1)/3600}' | xargs -I{} echo {} hour
