# CGNN: Link Prediction in Multilayer Networks via Cross-Network Embedding
This repository contains the code of paper:  
 >G. Ren, X. Ding, X.-K. Xu, and H.-F. Zhang, “Link Prediction in Multilayer Networks via Cross-Network Embedding”, AAAI, vol. 38, no. 8, pp. 8939-8947, Mar. 2024.

#### Requirements
python                    >=3.8.12

networkx                  >=2.6.3

numpy                     >=1.20.3

scikit-learn              >=0.24.2

torch                     >=1.9.0

torch-geometric           >=2.0.4

#### Examples
If you want to run CGNN algorithm on Facebook-Twitter dataset with 90% training edges and 90% interlayer links, run the following command in the home directory of this project:
`python train.py --anchor data/fb-tt/anchor0.9.pkl --split_path data/fb-tt/split0.9.pkl` 

## Reference  
If you are interested in our researches, please cite our papers:  

@article{Ren_CGNN_2024, 
title={Link Prediction in Multilayer Networks via Cross-Network Embedding}, 
volume={38}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/28742}, 
DOI={10.1609/aaai.v38i8.28742}, 
number={8}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Ren, Guojing and Ding, Xiao and Xu, Xiao-Ke and Zhang, Hai-Feng}, 
year={2024}, 
month={Mar.}, 
pages={8939-8947} }
