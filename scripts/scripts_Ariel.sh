nohup python -u run_ariel.py --dataset 'Cora' --encoder_model 'Ariel' --attack 'random' --device 0 --num_repeat 3 --num_sample 20 > ./logs/Cora/Smooth_Ariel_Random.log 2>&1 &
nohup python -u run_ariel.py --dataset 'Cora' --encoder_model 'Ariel' --attack 'PRBCD' --device 0 --num_repeat 3 --num_sample 20 > ./logs/Cora/Smooth_Ariel_PRBCD.log 2>&1 &
nohup python -u run_ariel.py --dataset 'Cora' --encoder_model 'Ariel' --attack 'nettack' --device 0 --num_repeat 2 --num_sample 20 > ./logs/Cora/Smooth_Ariel_nettack.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'Ariel' --attack 'CLGA' --device 0 --num_repeat 3 --num_sample 20 > ./logs/Cora/Smooth_Ariel_CLGA.log 2>&1 &

nohup python -u run_smooth_node_clga.py --dataset 'Physics' --encoder_model 'Ariel' --attack 'random' --device 0  --num_repeat 3 --num_sample 20 > ./logs/Physics/Ariel_Random.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Physics' --encoder_model 'Ariel' --attack 'PRBCD' --device 0  --num_repeat 3 --num_sample 20 > ./logs/Physics/Ariel_PRBCD.log 2>&1 &

nohup python -u run_smooth_node_clga.py --dataset 'ogbn-arxiv' --encoder_model 'Ariel' --attack 'random' --device 1  --num_repeat 3 --num_sample 20 > ./logs/ogbn-arxiv/Ariel_Random.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'ogbn-arxiv' --encoder_model 'Ariel' --attack 'PRBCD' --device 0  --num_repeat 3 --num_sample 20 > ./logs/ogbn-arxiv/Ariel_PRBCD.log 2>&1 &

nohup python -u run_ariel.py --dataset 'Pubmed' --encoder_model 'Ariel' --attack 'random' --device 1 --num_repeat 3 --num_sample 20 > ./logs/Pubmed/Smooth_Ariel_Random.log 2>&1 &
nohup python -u run_ariel.py --dataset 'Pubmed' --encoder_model 'Ariel' --attack 'PRBCD' --device 0 --num_repeat 3 --num_sample 20 > ./logs/Pubmed/Smooth_Ariel_PRBCD.log 2>&1 &
nohup python -u run_ariel.py --dataset 'Pubmed' --encoder_model 'Ariel' --attack 'nettack' --device 0 --num_repeat 2 --num_sample 20 > ./logs/Pubmed/Smooth_Ariel_nettack.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Pubmed' --encoder_model 'Ariel' --attack 'CLGA' --device 0 --num_repeat 3 --num_sample 20 > ./logs/Pubmed/Smooth_Ariel_CLGA.log 2>&1 &


nohup python -u run_ariel.py --dataset 'Cora' --encoder_model 'Ariel' --attack 'random_global' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Cora/Smooth_Ariel_random_global.log 2>&1 &
nohup python -u run_ariel.py --dataset 'Pubmed' --encoder_model 'Ariel' --attack 'random_global' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Pubmed/Smooth_Ariel_random_global.log 2>&1 &
nohup python -u run_ariel.py --dataset 'Physics' --encoder_model 'Ariel' --attack 'random_global' --device 2 --num_repeat 3 --num_sample 20 > ./logs/Physics/Smooth_Ariel_random_global.log 2>&1 &
nohup python -u run_ariel.py --dataset 'ogbn-arxiv' --encoder_model 'Ariel' --attack 'random_global' --device 2 --num_repeat 3 --num_sample 20 > ./logs/ogbn-arxiv/Smooth_Ariel_random_global.log 2>&1 &
