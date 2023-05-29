#!/bin/bash -l
#
#SBATCH -J fasttext_gridsearch
#SBATCH -o logs/fasttext_gridsearch."%j".out
#SBATCH -e logs/fasttext_gridsearch."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node101 OR node302 OR node108
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

python3 models/lyric/fasttext.py --grid_search