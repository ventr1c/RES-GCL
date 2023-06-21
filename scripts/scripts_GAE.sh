## Nettack
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'GAE' --attack 'nettack' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Cora/GAE_Nettack.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'GAE' --attack 'nettack' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/GAE_Nettack.log 2>&1 &

## Random attack 
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'GAE' --attack 'random' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Cora/GAE_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'GAE' --attack 'random' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/GAE_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'GAE' --attack 'random' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Computers/GAE_Random1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'GAE' --attack 'random' --device 1 --num_repeat 3 --num_sample 20 > ./logs/ogbn-arxiv/GAE_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'GAE' --attack 'random' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Physics/GAE_Random.log 2>&1 &

## PRBCD
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'GAE' --attack 'PRBCD' --device 3 --num_repeat 4 --num_sample 20 > ./logs/Cora/GAE_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'GAE' --attack 'PRBCD' --device 1 --num_repeat 4 --num_sample 20 > ./logs/Pubmed/GAE_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'GAE' --attack 'PRBCD' --device 2 --num_repeat 4 --num_sample 20 > ./logs/Computers/GAE_PRBCD.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'GAE' --attack 'PRBCD' --device 1 --num_repeat 3 --num_sample 20 > ./logs/ogbn-arxiv/GAE_PRBCD32.log 2>&1 &

nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'GAE' --attack 'PRBCD' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Physics/GAE_PRBCD.log 2>&1 &


## Random attack global
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'GAE' --attack 'random_global' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Cora/GAE_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'GAE' --attack 'random_global' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/GAE_random_global.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'GAE' --attack 'random_global' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Computers/GAE_random_global1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'GAE' --attack 'random_global' --device 0 --num_repeat 3 --num_sample 20 > ./logs/ogbn-arxiv/GAE_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'GAE' --attack 'random_global' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Physics/GAE_random_global.log 2>&1 &