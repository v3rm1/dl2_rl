#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=dl2_ppo
#SBATCH --mem=72000
module load Python/3.6.4-foss-2018a
module load CUDA/9.1.85
module load Boost/1.66.0-foss-2018a-Python-3.6.4
module load TensorFlow/1.12.0-foss-2018a-Python-3.6.4
pip install gym
pip install atari-py
python ./proximal_policy_optimization/main.py
mv *.out slurm/
