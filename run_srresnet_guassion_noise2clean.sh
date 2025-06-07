#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=SGNC128
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --account=a_bai
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhiyong.ma@student.uq.edu.au
#SBATCH --no-requeue
#SBATCH -o outputs/SGNC128/logs/output.log
#SBATCH -e outputs/SGNC128/logs/error.log

source /home/uqzwan39/.bashrc
conda activate pt2
export PYTHONPATH=$PYTHONPATH:/scratch/user/uqzwan39/zhiyong.ma/duc
srun --export=PATH,TERM,HOME,LANG /bin/bash run_srresnet_noise_clean.sh
