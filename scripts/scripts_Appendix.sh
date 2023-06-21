# Robustness
## Random Global
## Base GRACE
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random_global' --device 3 --num_repeat 5 --num_sample 20 >> ./logs/Cora/Base_Grace_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random_global' --device 2 --num_repeat 5 --num_sample 20 >> ./logs/Pubmed/Base_Grace_random_global1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random_global' --device 1 --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/Physics/Base_Grace_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random_global' --device 1 --num_repeat 2 --num_sample 20 --cont_batch_size 512 > ./logs/ogbn-arxiv/Base_Grace_random_global_0508_512_hid512proj256.log 2>&1 &
## Base BGRL
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'random_global' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_BGRL_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'random_global' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_BGRL_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'BGRL' --attack 'random_global' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Physics/Base_BGRL_random_global.log 2>&1 &
## Base DGI
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'random_global' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_DGI_random_global_0521.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'random_global' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_DGI_random_global_0521.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'DGI' --attack 'random_global' --device 3 --num_repeat 3 --num_sample 20  > ./logs/Physics/Base_DGI_random_global_0521.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'DGI' --attack 'random_global' --device 3 --num_repeat 3 --num_sample 20 > ./logs/ogbn-arxiv/Base_DGI_random_global_0521.log 2>&1 &
## RES BGRL
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'random_global' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_BGRL_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'random_global' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_BGRL_random_global.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'BGRL' --attack 'random_global' --device 1 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Physics/Smooth_BGRL_random_global.log 2>&1 &
## RES DGI
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'random_global' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_DGI_random_global_new.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'random_global' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_DGI_random_global_new.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'DGI' --attack 'random_global' --device 3 --if_smoothed --num_repeat 3 --num_sample 20 > ./logs/Physics/Smooth_DGI_random_global_new.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'DGI' --attack 'random_global' --device 3 --if_smoothed --num_repeat 2 --num_sample 20 > ./logs/ogbn-arxiv/Smooth_DGI_random_global_new.log 2>&1 &