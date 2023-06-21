## CLGA
nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'Grace' --attack 'CLGA' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_Grace_CLGA10.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'Grace' --attack 'CLGA' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_Grace_CLGA10.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'GAE' --attack 'CLGA' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Cora/GAE_CLGA10.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'Grace-Jaccard' --attack 'CLGA' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Cora/Grace-Jaccard_CLGA10.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Cora' --encoder_model 'Node2vec' --attack 'CLGA' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Cora/Node2vec_CLGA10.log 2>&1 &

nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'Ariel' --attack 'CLGA' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 >> ./logs/Cora/Smooth_Ariel_CLGA.log 2>&1 &

nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'CLGA' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_BGRL_CLGA10.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'CLGA' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_BGRL_CLGA10.log 2>&1 &

nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'DGI' --attack 'CLGA' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_DGI_CLGA10_0521.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'DGI' --attack 'CLGA' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_DGI_CLGA10_0521.log 2>&1 &

## CLGA
nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'CLGA' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_Grace_CLGA10.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'CLGA' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 >> ./logs/Pubmed/Smooth_Grace_CLGA10.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'GAE' --attack 'CLGA' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/GAE_CLGA10.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'Grace-Jaccard' --attack 'CLGA' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Grace-Jaccard_CLGA10.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Pubmed' --encoder_model 'Node2vec' --attack 'CLGA' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Node2vec_CLGA10.log 2>&1 &

nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'Ariel' --attack 'CLGA' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 >> ./logs/Pubmed/Smooth_Ariel_CLGA.log 2>&1 &

nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'CLGA' --device 0 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_BGRL_CLGA10.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'CLGA' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_BGRL_CLGA10.log 2>&1 &

nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'CLGA' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Base_DGI_CLGA10_0521.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'CLGA' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_DGI_CLGA10_0521.log 2>&1 &
