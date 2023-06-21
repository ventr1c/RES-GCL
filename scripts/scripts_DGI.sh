# Transductive
# Smoothed DGI
## Nettack
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'nettack' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_DGI_Nettack_0509_full.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'nettack' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_DGI_Nettack_0509.log 2>&1 &

## Random attack 
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 >> ./logs/Cora/Smooth_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'DGI' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers/Smooth_DGI_Random_iter1500.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'DGI' --attack 'random' --device 0 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Physics/Smooth_DGI_Random.log 2>&1 &

## PRBCD
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'PRBCD' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'DGI' --attack 'PRBCD' --device 1 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Physics/Smooth_DGI_PRBCD_0520_night.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'DGI' --attack 'PRBCD' --device 1 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/ogbn-arxiv/Smooth_DGI_PRBCD_0520_night.log 2>&1 &

# Base DGI
## Nettack
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'nettack' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_DGI_Nettack_0508.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'nettack' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_DGI_Nettack_0508.log 2>&1 &

## Random attack 
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'random' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'random' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'DGI' --attack 'random' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'DGI' --attack 'random' --device 0 --num_repeat 3 --num_sample 20 > ./logs/Physics/Base_DGI_Random.log 2>&1 &

## PRBCD
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'PRBCD' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'PRBCD' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'DGI' --attack 'PRBCD' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'DGI' --attack 'PRBCD' --device 1 --num_repeat 3 --num_sample 20 > ./logs/ogbn-arxiv/Base_DGI_PRBCD_0521.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'DGI' --attack 'PRBCD' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Physics/Base_DGI_PRBCD_0521.log 2>&1 &

# Inductive
##  Smooth
### Random 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'DGI' --attack 'random' --device 0 --if_smoothed --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Smooth_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'random' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Pubmed/Induc_Smooth_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'DGI' --attack 'random' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Smooth_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'ogbn-arxiv' --encoder_model 'DGI' --attack 'random' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Induc_Smooth_DGI_Random.log 2>&1 &
### PRBCD 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'DGI' --attack 'PRBCD' --device 1 --if_smoothed --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Smooth_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'PRBCD' --device 0 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Pubmed/Induc_Smooth_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'DGI' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Smooth_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Physics' --encoder_model 'DGI' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Physics/Induc_Smooth_DGI_PRBCD.log 2>&1 &

## Base 
### Random 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'DGI' --attack 'random' --device 0 --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Base_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'random' --device 2 --num_repeat 3 --num_sample 20 >> ./logs/Pubmed/Induc_Base_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'DGI' --attack 'random' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Base_DGI_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'ogbn-arxiv' --encoder_model 'DGI' --attack 'random' --device 2 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Induc_Base_DGI_Random.log 2>&1 &


### PRBCD 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'DGI' --attack 'PRBCD' --device 2 --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Base_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'PRBCD' --device 0 --num_repeat 3 --num_sample 20 >> ./logs/Pubmed/Induc_Base_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'DGI' --attack 'PRBCD' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Base_DGI_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Physics' --encoder_model 'DGI' --attack 'PRBCD' --device 3 --num_repeat 3 --num_sample 20 > ./logs/Physics/Induc_Base_DGI_PRBCD.log 2>&1 &

nohup python -u run_smooth_node_induc1.py --dataset 'ogbn-arxiv' --encoder_model 'DGI' --attack 'PRBCD' --device 3 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Induc_Base_DGI_PRBCD.log 2>&1 &


## Certified Accuracy
nohup python -u run_certify_node.py --dataset 'Cora' --encoder_model 'DGI' --device 1 --num_repeat 3 --n0 20 --n 200 --alpha 0.01 > ./logs/Cora_Certify_DGI_new.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Pubmed' --encoder_model 'DGI' --device 2 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Pubmed_Certify_DGI_re5n020n200alpha1_1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Computers' --encoder_model 'DGI' --device 2 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Computers_Certify_DGI_re5n020n200alpha1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Physics' --encoder_model 'DGI' --device 3 --num_repeat 1 --n0 20 --n 200 --alpha 0.01 > ./logs/Physics_Certify_DGI_0521.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'ogbn-arxiv' --encoder_model 'DGI' --device 1 --num_repeat 2 --n0 20 --n 200 --alpha 0.01 > ./logs/ogbn-arxiv_Certify_DGI_new.log 2>&1 &