#!/bin/bash -l
#
#SBATCH -J extract_features
#SBATCH -o logs/extract_features."%j".out
#SBATCH -e logs/extract_features."%j".err
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