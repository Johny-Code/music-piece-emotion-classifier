#!/bin/bash -l
#
#SBATCH -A ixam_users2
#SBATCH -J feature_based_ann
#SBATCH -o logs/feature_based_ann."%j".out
#SBATCH -e logs/feature_based_ann."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node101 OR node302 OR node108
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

# train model
python3 models/lyric/train_ann.py --simple_run
