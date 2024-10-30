PD=data/fb-tt
edge_ratio=0.9

for anchor_ratio in 0.3 0.6 0.9
do
python train.py \
--anchor ${PD}/anchor${anchor_ratio}.pkl \
--split_path ${PD}/split${edge_ratio}.pkl \
--dim 128 \
--lr 0.001 \
--pre_epochs 0 \
--epochs 1000 \
--margin 0.9 \
--alpha 0.2 \
--verbose 0 \
--edge_type gcn \
--nodeattr svd \
--expand_edges False
done