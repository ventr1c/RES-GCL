# Transductive
# Smoothed Grace
## Nettack
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'nettack' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_Grace_Nettack_0509.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'nettack' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_Grace_Nettack_0429.log 2>&1 &

## Random attack 
nohup python -u run_smooth_node_auto.py --dataset 'Cora' --encoder_model 'Grace_auto' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_Grace_Random256_auto.log 2>&1 &
nohup python -u run_smooth_node_auto.py --dataset 'ogbn-arxiv' --encoder_model 'Grace_auto' --attack 'random' --device 2 --num_repeat 5 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Base_Grace_Random1.log 2>&1

nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 >> ./logs/Cora/Smooth_Grace_Random256.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_Grace_Random256.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers/Smooth_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Smooth_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 2 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Smooth_Grace_Random_0424_512_hid512proj256.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 1 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Smooth_Grace_Random_0423_512.log 2>&1 &
## PRBCD
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_Grace_PRBCD1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_Grace_PRBCD_0508.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers/Smooth_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 1 --if_smoothed --num_repeat 1 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Smooth_Grace_PRBCD_0423_1024_hid512_proj256.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Smooth_Grace_PRBCD.log 2>&1 &

## Certified Accuracy
nohup python -u run_certify_node.py --dataset 'Cora' --encoder_model 'Grace' --device 1 --num_repeat 2 --n0 20 --n 200 --alpha 0.01 > ./logs/Cora/Certify_Grace_new_1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Pubmed' --encoder_model 'Grace' --device 2 --num_repeat 2 --n0 20 --n 200 --alpha 0.01 > ./logs/Pubmed_Certify_Grace_re5n020n200alpha1_1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Computers' --encoder_model 'Grace' --device 2 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Computers_Certify_Grace_re5n020n200alpha1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Physics' --encoder_model 'Grace' --device 2 --num_repeat 3 --n0 20 --n 200 --alpha 0.01 --cont_batch_size 2048 > ./logs/Physics_Certify_Grace_0520.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --device 3 --num_repeat 2 --n0 20 --n 200 --alpha 0.01 --cont_batch_size 512 > ./logs/ogbn-arxiv/Certify_Grace_0425_512.log 2>&1 &

## Random attack global
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random_global' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_Grace_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random_global' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_Grace_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random_global' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Smooth_Grace_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random_global' --device 3 --if_smoothed --num_repeat 2 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Smooth_Grace_random_global_0424_512_hid512proj256.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random_global' --device 1 --if_smoothed --num_repeat 1 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Smooth_Grace_random_global_0423_512.log 2>&1 &

# Base Grace
## Nettack
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'nettack' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_Grace_Nettack_0508.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'nettack' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_Grace_Nettack_0429.log 2>&1 &

## Random attack 
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --num_repeat 5 --num_sample 20 >> ./logs/Cora/Base_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --num_repeat 5 --num_sample 20 >> ./logs/Pubmed/Base_Grace_Random1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace' --attack 'random' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 2 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Base_Grace_Random_0423_512_hid512proj256.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Base_Grace_Random.log 2>&1 &
## PRBCD
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --num_repeat 5 --num_sample 20 >> ./logs/Cora/Base_Grace_PRBCD1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --num_repeat 5 --num_sample 20 >> ./logs/Pubmed/Base_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 0 --num_repeat 3 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Base_Grace_PRBCD_0424_512_hid512_proj256.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Base_Grace_PRBCD.log 2>&1 &

## Random attack global
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random_global' --device 3 --num_repeat 5 --num_sample 20 >> ./logs/Cora/Base_Grace_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random_global' --device 2 --num_repeat 5 --num_sample 20 >> ./logs/Pubmed/Base_Grace_random_global1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random_global' --device 1 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Base_Grace_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random_global' --device 1 --num_repeat 2 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Base_Grace_random_global_0508_512_hid512proj256.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace' --attack 'random_global' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_Grace_random_global.log 2>&1 &






# Inductive
##  Smooth
### Random 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Smooth_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Pubmed/Induc_Smooth_Grace_Random256.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Smooth_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Smooth_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 2 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Smooth_Grace_Random.log 2>&1 &

### PRBCD 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Smooth_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'PRBCD' --device 0 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Pubmed/Induc_Smooth_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Smooth_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Physics' --encoder_model 'Grace' --attack 'PRBCD' --device 0 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Physics/Induc_Smooth_Grace_PRBCD.log 2>&1 &


## Base 
### Random 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Base_Grace_Random1.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 2 --num_repeat 3 --num_sample 20 >> ./logs/Pubmed/Induc_Base_Grace_Random1.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Base_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Induc_Base_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 2 --num_repeat 2 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Base_Grace_Random.log 2>&1 &

### PRBCD 
nohup python -u run_smooth_node_induc1.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 0 --num_repeat 3 --num_sample 20 >> ./logs/Cora/Induc_Base_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --num_repeat 3 --num_sample 20 >> ./logs/Pubmed/Induc_Base_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Computers' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --num_repeat 3 --num_sample 20 > ./logs/Computers/Induc_Base_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Induc_Base_Grace_PRBCD.log 2>&1 &
nohup python -u run_smooth_node_induc1.py --dataset 'Physics' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Induc_Base_Grace_PRBCD.log 2>&1 &







nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace_batch' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 >> ./logs/Cora/Smooth_Grace_batch.log 2>&1 &
