#!/bin/bash -l
#
#SBATCH -J xlnet_simple_run
#SBATCH -o logs/xlnet_simple_run."%j".out
#SBATCH -e logs/xlnet_simple_run."%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --nodelist=node101 OR node302 OR node108
#SBATCH -p short

# activate python venv
source .././emotion_env/bin/activate

# train model
python3 models/lyric/train_xlnet_v2.py --fine_tune
