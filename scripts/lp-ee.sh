#!/bin/bash
#SBATCH --job-name=LP-ee
#SBATCH -p GPUAIC02
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
echo $(hostname) $CUDA_VISIBLE_DEVICES

PD=data/fb-tt
anchor_ratio=0.9
edge_ratio=0.9

for method in SC node2vec SVD GAE GAT GraphSAGE
do
python lp_ee.py \
--anchor ${PD}/anchor${anchor_ratio}.pkl \
--split_dir ${PD}/split${edge_ratio}.pkl \
--method ${method}
done