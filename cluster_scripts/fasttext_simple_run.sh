#!/bin/bash -l
#
#SBATCH -A eixam_users2
#SBATCH -J fasttext_simple_run
#SBATCH -o logs/fasttext_simple_run."%j".out
#SBATCH -e logs/fasttext_simple_run."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node122
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

python3 models/lyric/train_fasttext.py --simple_run