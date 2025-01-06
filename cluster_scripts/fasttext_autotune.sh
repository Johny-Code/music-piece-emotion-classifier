#!/bin/bash -l
#
#SBATCH -J fasttext_autotune
#SBATCH -o logs/fasttext_autotune."%j".out
#SBATCH -e logs/fasttext_autotune."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node101 OR node302 OR node108
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

python3 models/lyric/train_fasttext.py --autotune