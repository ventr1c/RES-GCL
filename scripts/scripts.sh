nohup python -u run_robust_acc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'nettack' --device 0 --if_smoothed --num_repeat 1 > ./logs/Cora_Smooth_Grace_Nettack_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'nettack' --device 0 --num_repeat 1 > ./logs/Cora_Base_Grace_Nettack_beta.log 2>&1 &

nohup python -u run_robust_acc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'nettack' --device 2 --if_smoothed --num_repeat 1> ./logs/Pubmed_Smooth_Grace_Nettack_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'nettack' --device 2 --num_repeat 1 > ./logs/Pubmed_Base_Grace_Nettack_beta.log 2>&1 &

nohup python -u run_robust_acc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'nettack' --device 0 --if_smoothed --num_repeat 1 > ./logs/Citeseer_Smooth_Grace_Nettack_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'nettack' --device 0 --num_repeat 1 > ./logs/Citeseer_Base_Grace_Nettack_beta.log 2>&1 &

nohup python -u run_robust_acc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 > ./logs/Cora_Smooth_Grace_Random_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 5 > ./logs/Cora_Base_Grace_Random_beta.log 2>&1 &

nohup python -u run_robust_acc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 > ./logs/Pubmed_Smooth_Grace_Random_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --num_repeat 5 > ./logs/Pubmed_Base_Grace_Random_beta.log 2>&1 &

nohup python -u run_robust_acc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 > ./logs/Citeseer_Smooth_Grace_Random_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 0 --num_repeat 5 > ./logs/Citeseer_Base_Grace_Random_beta.log 2>&1 &

nohup python -u run_robust_acc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'none' --if_smoothed --device 0 > ./logs/Cora_Smooth_Grace_Clean_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'none' --if_smoothed --device 0 > ./logs/Citeseer_Smooth_Grace_Clean_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'none' --if_smoothed --device 1 > ./logs/Pubmed_Smooth_Grace_Clean_beta.log 2>&1 &

nohup python -u run_robust_acc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'none' --device 0 > ./logs/Cora_Base_Grace_Clean_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'none' --device 0 > ./logs/Citeseer_Base_Grace_Clean_beta.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'none' --device 1 > ./logs/Pubmed_Base_Grace_Clean_beta.log 2>&1 &

python -u run_graphcl.py --dataset 'MUTAGS' --encoder_model 'GraphCL' --attack 'none' --device 2



nohup python -u run_robust_acc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed > ./logs/Cora_Smooth_Grace_random_dense.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed > ./logs/Citeseer_Smooth_Grace_random_dense.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed > ./logs/Pubmed_Smooth_Grace_random_dense.log 2>&1 &

nohup python -u run_graphcl.py --dataset 'PROTEINS' --encoder_model 'GraphCL' --attack 'random' --device 2 > ./logs/PROTEINS_Base_GraphCL_random.log 2>&1 &
nohup python -u run_graphcl.py --dataset 'MUTAG' --encoder_model 'GraphCL' --attack 'random' --device 2 > ./logs/MUTAG_Base_GraphCL_random.log 2>&1 &

nohup python -u run_graphcl.py --dataset 'PROTEINS' --encoder_model 'GraphCL' --attack 'random' --device 2 if_smoothed > ./logs/PROTEINS_Smooth_GraphCL_random.log 2>&1 &
nohup python -u run_graphcl.py --dataset 'MUTAG' --encoder_model 'GraphCL' --attack 'random' --device 2 if_smoothed > ./logs/MUTAG_Smooth_GraphCL_random.log 2>&1 &


nohup python -u run_gnn.py --dataset 'Cora' --model 'GNNGuard' --attack 'none' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GNNGuard_clean.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Cora' --model 'GNNGuard' --attack 'random' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GNNGuard_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Cora' --model 'GNNGuard' --attack 'nettack' --device 0 --if_smoothed --num_repeat 1 > ./logs/Cora_GNNGuard_nettack.log 2>&1 &

nohup python -u run_gnn.py --dataset 'Pubmed' --model 'GNNGuard' --attack 'none' --device 1 --if_smoothed --num_repeat 5 > ./logs/Pubmed_GNNGuard_clean.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Pubmed' --model 'GNNGuard' --attack 'random' --device 1 --if_smoothed --num_repeat 5 > ./logs/Pubmed_GNNGuard_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Pubmed' --model 'GNNGuard' --attack 'nettack' --device 1 --if_smoothed --num_repeat 1 > ./logs/Pubmed_GNNGuard_nettack.log 2>&1 &

nohup python -u run_gnn.py --dataset 'Citeseer' --model 'GNNGuard' --attack 'none' --device 2 --if_smoothed --num_repeat 5 > ./logs/Citeseer_GNNGuard_clean.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Citeseer' --model 'GNNGuard' --attack 'random' --device 2 --if_smoothed --num_repeat 5 > ./logs/Citeseer_GNNGuard_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Citeseer' --model 'GNNGuard' --attack 'nettack' --device 2 --if_smoothed --num_repeat 1 > ./logs/Citeseer_GNNGuard_nettack.log 2>&1 &
# nohup python -u run_gnn.py --dataset 'Cora' --model 'RobustGCN' --attack 'none' --device 0 --if_smoothed > ./logs/Cora_RobustGCN_clean.log 2>&1 &