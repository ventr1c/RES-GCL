# Transductive
# Smoothed BGRL
## Nettack
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'nettack' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_BGRL_Nettack_0509.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'nettack' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_BGRL_Nettack_0509.log 2>&1 &

## Random attack 
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 >> ./logs/Cora/Smooth_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'BGRL' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers/Smooth_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'BGRL' --attack 'random' --device 1 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Physics/Smooth_BGRL_Random.log 2>&1 &
## PRBCD
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'PRBCD' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_BGRL_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_BGRL_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'BGRL' --attack 'PRBCD' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers/Smooth_BGRL_PRBCD.log 2>&1 &
## Certified Accuracy
nohup python -u run_certify_node.py --dataset 'Cora' --encoder_model 'BGRL' --device 1 --num_repeat 3 --n0 20 --n 200 --alpha 0.01 > ./logs/Cora/Certify_BGRL_new.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --device 2 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Pubmed/Certify_BGRL_re5n020n200alpha1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'ogbn-arxiv' --encoder_model 'BGRL' --device 3 --num_repeat 3 --n0 20 --n 200 --alpha 0.01 >> ./logs/ogbn-arxiv/Certify_BGRL_0520.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Physics' --encoder_model 'BGRL' --device 2 --num_repeat 3 --n0 20 --n 200 --alpha 0.01 > ./logs/Physics_Certify_BGRL_0520.log 2>&1 &
# Base BGRL
## Nettack
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'nettack' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_BGRL_Nettack_0508.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'nettack' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_BGRL_Nettack_0608.log 2>&1 &

## Random attack 
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'random' --device 0 --num_repeat 5 --num_sample 20 >> ./logs/Cora/Base_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'random' --device 0 --num_repeat 5 --num_sample 20 >> ./logs/Pubmed/Base_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'BGRL' --attack 'random' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'BGRL' --attack 'random' --device 2 --num_repeat 3 --num_sample 20 > ./logs/Physics/Base_BGRL_Random1.log 2>&1 &
## PRBCD
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'PRBCD' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_BGRL_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'PRBCD' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_BGRL_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'BGRL' --attack 'PRBCD' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_BGRL_PRBCD.log 2>&1 &



# Inductive
##  Smooth
### Random 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'random' --device 0 --if_smoothed --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Smooth_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'random' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Pubmed/Induc_Smooth_BGRL_Random256.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'BGRL' --attack 'random' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Smooth_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'ogbn-arxiv' --encoder_model 'BGRL' --attack 'random' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Smooth_BGRL_Random.log 2>&1 &

## Base 
### Random 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'random' --device 0 --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Base_BGRL_Random1.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'random' --device 2 --num_repeat 3 --num_sample 20 >> ./logs/Pubmed/Induc_Base_BGRL_Random1.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'BGRL' --attack 'random' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Base_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'ogbn-arxiv' --encoder_model 'BGRL' --attack 'random' --device 1 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Induc_Base_BGRL_Random.log 2>&1 &

### PRBCD 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'PRBCD' --device 1 --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Base_BGRL_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'PRBCD' --device 0 --num_repeat 3 --num_sample 20 >> ./logs/Pubmed/Induc_Base_BGRL_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'BGRL' --attack 'PRBCD' --device 3 --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Base_BGRL_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Physics' --encoder_model 'BGRL' --attack 'PRBCD' --device 2 --num_repeat 3 --num_sample 20 > ./logs/Physics/Induc_Base_BGRL_PRBCD.log 2>&1 &
