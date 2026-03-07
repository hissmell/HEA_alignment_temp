#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2      # Cores per node
#SBATCH --partition=snu-gpu2     # Partition Name
##
#SBATCH --job-name=orb_latent_extraction
#SBATCH --time=24-00:00          # Runtime: Day-HH:MM
#SBATCH -o orb_extract.%N.%j.out         # STDOUT
#SBATCH -e orb_extract.%N.%j.err         # STDERR
#
#SBATCH --gres=gpu:l40s:1
##

hostname
date
module purge

StartTime=$(date +%s)
cd $SLURM_SUBMIT_DIR

# Activate Orb environment
source /home/pn50212/anaconda3/etc/profile.d/conda.sh
conda activate orb

# Move to project directory
cd /DATA/user_scratch/pn50212/2024/12_AtomAttention

# Extract Orb latent vectors for all 4 models and both adsorbates
python scripts/experiments/extract_orb_latent_25cao.py --all

EndTime=$(date +%s)

echo Run time
echo $StartTime $EndTime | awk '{print $2-$1}' | xargs -I{} echo {} sec
echo $StartTime $EndTime | awk '{print ($2-$1)/60}' | xargs -I{} echo {} min
echo $StartTime $EndTime | awk '{print ($2-$1)/3600}' | xargs -I{} echo {} hour