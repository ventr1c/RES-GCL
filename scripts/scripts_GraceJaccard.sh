# # Transductive
# # Smoothed Grace-Jaccard
# ## Nettack
# nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace-Jaccard' --attack 'nettack' --device 1 --num_repeat 2 --num_sample 20 > ./logs/Cora/Base_Grace-Jaccard_Nettack1.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace-Jaccard' --attack 'nettack' --device 1 --num_repeat 2 --num_sample 20 > ./logs/Pubmed/Base_Grace-Jaccard_Nettack1.log 2>&1 &

# ## Random attack 
# nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace-Jaccard' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 >> ./logs/Cora/Smooth_Grace-Jaccard_Random.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace-Jaccard' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_Grace-Jaccard_Random_1.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace-Jaccard' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers/Smooth_Grace-Jaccard_Random.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace-Jaccard' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 --cont_batch_size > ./logs/ogbn-arxiv/Smooth_Grace-Jaccard_Random.log 2>&1 &
# ## PRBCD
# nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace-Jaccard' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_Grace-Jaccard_PRBCD1.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace-Jaccard' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_Grace-Jaccard_PRBCD.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace-Jaccard' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers/Smooth_Grace-Jaccard_PRBCD.log 2>&1 &

# Base Grace-Jaccard
## Nettack
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace-Jaccard' --attack 'nettack' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_Grace-Jaccard_Nettack.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace-Jaccard' --attack 'nettack' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Pubmed/Base_Grace-Jaccard_Nettack.log 2>&1 &

## Random attack 
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace-Jaccard' --attack 'random' --device 3 --num_repeat 5 --num_sample 20 >> ./logs/Cora/Base_Grace-Jaccard_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace-Jaccard' --attack 'random' --device 2 --num_repeat 5 --num_sample 20 >> ./logs/Pubmed/Base_Grace-Jaccard_Random1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace-Jaccard' --attack 'random' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_Grace-Jaccard_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace-Jaccard' --attack 'random' --device 0 --num_repeat 5 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Base_Grace-Jaccard_Random_0424_512.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace-Jaccard' --attack 'random' --device 2 --num_repeat 3 --cont_batch_size 2048 > ./logs/Physics/Base_Grace-Jaccard_Random.log 2>&1 &
## PRBCD
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace-Jaccard' --attack 'PRBCD' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_Grace-Jaccard_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace-Jaccard' --attack 'PRBCD' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_Grace-Jaccard_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace-Jaccard' --attack 'PRBCD' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_Grace-Jaccard_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace-Jaccard' --attack 'PRBCD' --device 3 --num_repeat 3 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Base_Grace-Jaccard_PRBCD_0424_512.log 2>&1 &

## Random attack global
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace-Jaccard' --attack 'random_global' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_Grace-Jaccard_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace-Jaccard' --attack 'random_global' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_Grace-Jaccard_random_global1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace-Jaccard' --attack 'random_global' --device 2 --num_repeat 3 --cont_batch_size 2048 > ./logs/Physics/Base_Grace-Jaccard_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace-Jaccard' --attack 'random_global' --device 1 --num_repeat 3 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Base_Grace-Jaccard_random_global_0508_512.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace-Jaccard' --attack 'random_global' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Computers/Base_Grace-Jaccard_random_global.log 2>&1 &
