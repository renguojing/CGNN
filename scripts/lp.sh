#!/bin/bash
#SBATCH --job-name=LP
#SBATCH -p GPUAIC02
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
echo $(hostname) $CUDA_VISIBLE_DEVICES

PD=data/db-wb1
edge_ratio=0.9

for method in CN JC AA RA PA CCPA
do
for layer in 1 2
do
python lp.py \
--split_dir ${PD}/split${edge_ratio}.pkl \
--method ${method} \
--normalize True \
--layer ${layer}
done
echo
done

for method in SC SVD node2vec GAE GAT GraphSAGE
do
for layer in 1 2
do
python lp.py \
--split_dir ${PD}/split${edge_ratio}.pkl \
--method ${method} \
--layer ${layer}
done
echo
done