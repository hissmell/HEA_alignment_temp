#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2      # Cores per node
#SBATCH --partition=snu-gpu1     # Partition Name
##
#SBATCH --job-name=debug_equiformer
#SBATCH --time=1-00:00          # Runtime: Day-HH:MM (1 hour for debug)
#SBATCH -o debug_equiformer.%N.%j.out         # STDOUT
#SBATCH -e debug_equiformer.%N.%j.err         # STDERR
#
#SBATCH --gres=gpu:a6000:1
##

hostname
date
module purge

StartTime=$(date +%s)
cd $SLURM_SUBMIT_DIR

# Activate fairchem1 environment (has proper fairchem modules)
source /home/pn50212/anaconda3/etc/profile.d/conda.sh
conda activate fairchem1

# Move to project directory
cd /DATA/user_scratch/pn50212/2024/12_AtomAttention

# Run the debug script
python scripts/analysis/debug_equiformer_extraction.py

EndTime=$(date +%s)

echo Run time
echo $StartTime $EndTime | awk '{print $2-$1}' | xargs -I{} echo {} sec
echo $StartTime $EndTime | awk '{print ($2-$1)/60}' | xargs -I{} echo {} min
echo $StartTime $EndTime | awk '{print ($2-$1)/3600}' | xargs -I{} echo {} hour