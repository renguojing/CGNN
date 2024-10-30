#!/bin/bash
#SBATCH --job-name=LP-eg
#SBATCH -p GPUMAT01
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
echo $(hostname) $CUDA_VISIBLE_DEVICES

PD=data/db-wb1
anchor_ratio=0.9
edge_ratio=0.9

for method in node2vec SC SVD GAE GAT GraphSAGE
do
python lp_eg.py \
--anchor ${PD}/anchor${anchor_ratio}.pkl \
--split_dir ${PD}/split${edge_ratio}.pkl \
--method ${method}
done