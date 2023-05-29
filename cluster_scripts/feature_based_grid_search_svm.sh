#!/bin/bash -l
#
#SBATCH -J feature_based_grid_search_svm
#SBATCH -o logs/feature_based_grid_search_svm."%j".out
#SBATCH -e logs/feature_based_grid_search_svm."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node101 OR node302 OR node108
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

# simple svm run
python models/lyric/train_svm.py --grid_search