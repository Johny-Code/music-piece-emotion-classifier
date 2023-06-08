#!/bin/bash -l
#
#SBATCH -A eixam_users2
#SBATCH -J xlnet_grid_search
#SBATCH -o logs/xlnet_grid_search."%j".out
#SBATCH -e logs/xlnet_grid_search."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH -p medium

# activate python venv
source .././emotion_env/bin/activate

# train model
python3 models/lyric/train_xlnet.py --grid_search
