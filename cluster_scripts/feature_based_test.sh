#!/bin/bash -l
#
#SBATCH -J feature_based_test
#SBATCH -o logs/feature_based_test."%j".out
#SBATCH -e logs/feature_based_test."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node101 OR node302 OR node108
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

# extract features
python3 tools/extract_features_from_lyric.py

# train model
python3 models/lyric/train_svm.py
