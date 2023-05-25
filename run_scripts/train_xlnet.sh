#!/bin/bash -l
#
#SBATCH -J my-hello-world
#SBATCH -o my-hello-world.”%j".out
#SBATCH -e my-hello-world.”%j".err
#
#SBATCH --mail-user s175502@student.pg.edu.pl
#SBATCH --mail-type=ALL
#
#SBATCH --mem=16M
#SBATCH -c 16
#SBATCH -p short


./../emotion_env/activate

python ./models/lyrics/train_xlnet.py

