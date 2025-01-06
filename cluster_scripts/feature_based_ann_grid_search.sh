#!/bin/bash -l
#
#SBATCH -J feature_based_ann_grid_search
#SBATCH -o logs/feature_based_ann_grid_search."%j".out
#SBATCH -e logs/feature_based_ann_grid_search."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node101 OR node302 OR node108
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

# train model
python3 models/lyric/train_ann.py --grid_search
