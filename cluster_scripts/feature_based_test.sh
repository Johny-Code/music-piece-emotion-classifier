#!/bin/bash -l
#
#SBATCH -J my-hello-world
#SBATCH -o my-hello-world."%j".out
#SBATCH -e my-hello-world."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node101 OR node302 OR node108
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

# extract features
python3 tools/extract_features_from_lyrics.py

# train model
python3 models/lyric/train_svm.py
