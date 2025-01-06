#!/bin/bash -l
#
#SBATCH -A eixam_users2
#SBATCH -J extract_features
#SBATCH -o logs/extract_features."%j".out
#SBATCH -e logs/extract_features."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node122
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

# extract features
python3 tools/extract_features_from_lyric.py