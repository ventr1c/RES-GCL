# Smoothed Grace
## Nettack
python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'nettack' --device 3 --if_smoothed --num_repeat 5 --num_sample 50 
python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'nettack' --device 1 --if_smoothed --num_repeat 5 --num_sample 50 

## PRBCD
python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 5 --num_sample 50
python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 5 --num_sample 50 
python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 5 --num_sample 50 
python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 1 --if_smoothed --num_repeat 5 --num_sample 50 --cont_batch_size 512
python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'PRBCD' --device 2 --if_smoothed --num_repeat 5 --num_sample 50 --cont_batch_size 2048 

## Random attack
python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random_global' --device 1 --if_smoothed --num_repeat 5 --num_sample 20
python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random_global' --device 0 --if_smoothed --num_repeat 5 --num_sample 50 
python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random_global' --device 2 --if_smoothed --num_repeat 5 --num_sample 50 --cont_batch_size 2048 
python -u run_smooth_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random_global' --device 3 --if_smoothed --num_repeat 5 --num_sample 50 --cont_batch_size 512 

## Certified Accuracy
python -u run_certify_node.py --dataset 'Cora' --encoder_model 'Grace' --device 1 --num_repeat 1 --n0 20 --n 200 --alpha 0.01 
python -u run_certify_node.py --dataset 'Pubmed' --encoder_model 'Grace' --device 2 --num_repeat 1 --n0 20 --n 200 --alpha 0.01
python -u run_certify_node.py --dataset 'Computers' --encoder_model 'Grace' --device 2 --num_repeat 1 --n0 20 --n 200 --alpha 0.01 
python -u run_certify_node.py --dataset 'Physics' --encoder_model 'Grace' --device 2 --num_repeat 1 --n0 20 --n 200 --alpha 0.01 --cont_batch_size 2048 
python -u run_certify_node.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --device 3 --num_repeat 1 --n0 20 --n 200 --alpha 0.01 --cont_batch_size 512 





