#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2      # Cores per node
#SBATCH --partition=snu-gpu2     # Partition Name
##
#SBATCH --job-name=eq_31M_O
#SBATCH --time=12-00:00          # Runtime: Day-HH:MM
#SBATCH -o eq_31M_O.%N.%j.out         # STDOUT
#SBATCH -e eq_31M_O.%N.%j.err         # STDERR
#
#SBATCH --gres=gpu:l40s:1
##

hostname
date
module purge

StartTime=$(date +%s)
cd $SLURM_SUBMIT_DIR

# Activate fairchem environment
source /home/pn50212/anaconda3/etc/profile.d/conda.sh
conda activate fairchem

# Move to project directory
cd /DATA/user_scratch/pn50212/2024/12_AtomAttention

# Extract for eqV2_31M_omat with O adsorbate only (for testing)
python scripts/experiments/extract_equiformer_latent_25cao.py --model eqV2_31M_omat --adsorbate O

EndTime=$(date +%s)

echo Run time
echo $StartTime $EndTime | awk '{print $2-$1}' | xargs -I{} echo {} sec
echo $StartTime $EndTime | awk '{print ($2-$1)/60}' | xargs -I{} echo {} min
echo $StartTime $EndTime | awk '{print ($2-$1)/3600}' | xargs -I{} echo {} hour