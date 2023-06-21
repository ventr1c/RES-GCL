# Transductive
# Smoothed Grace
## Nettack
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'nettack' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora_Smooth_Grace_Nettack_beta_0316.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'nettack' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed_Smooth_Grace_Nettack_beta_0316.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'nettack' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Citeseer_Smooth_Grace_Nettack_beta_0316.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace' --attack 'nettack' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers_Smooth_Grace_Nettack_beta.log 2>&1 &

## Random attack 
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora_Smooth_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed_Smooth_Grace_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers_Smooth_Grace_Random_beta1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Photo' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Photo_Smooth_Grace_Random.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Physics_Smooth_Grace_Random_beta.log 2>&1 &

nohup python -u run_smooth_node_induc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Cora_Base_Grace_Random_beta_induc.log 2>&1 &
nohup python -u run_smooth_node_induc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 2 --num_repeat 5 --num_sample 20 > ./logs/Pubmed_Base_Grace_Random_beta_induc.log 2>&1 &
nohup python -u run_smooth_node_induc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 5 --num_sample 20 > ./logs/Citeseer_Base_Grace_Random_beta_induc.log 2>&1 &

nohup python -u run_smooth_node_induc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora_Smooth_Grace_Random_beta_induc.log 2>&1 &
nohup python -u run_smooth_node_induc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed_Smooth_Grace_Random_beta_induc.log 2>&1 &
nohup python -u run_smooth_node_induc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Citeseer_Smooth_Grace_Random_beta_induc.log 2>&1 &

nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora_Smooth_BGRL_Random2.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed_Smooth_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'BGRL' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers_Smooth_BGRL.log 2>&1 &

nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Cora_Smooth_DGI_Random1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Pubmed_Smooth_DGI_Random1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'DGI' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/Computers_Smooth_DGI_Random1.log 2>&1 &

# Ablation Study
nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'random_drop' > ./logs/Ablation_Cora_1Struc_NoStruc_HaveMC1.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'keep_structure' > ./logs/Ablation_Cora_1Struc_HaveStruc_HaveMC1.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'random_drop' > ./logs/Ablation_Cora_0Struc_NoStruc_HaveMC1.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'keep_structure' > ./logs/Ablation_Cora_0Struc_HaveStruc_HaveMC1.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 --if_keep_structure1 --sample_way 'random_drop' > ./logs/Ablation_Cora_2Struc_NoStruc_HaveMC1.log 2>&1 &

nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'random_drop' > ./logs/Ablation_Cora_1Struc_NoStruc_NoMC1.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'random_drop' > ./logs/Ablation_Cora_0Struc_NoStruc_NoMC1.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'keep_structure' > ./logs/Ablation_Cora_1Struc_HaveStruc_NoMC1.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'keep_structure' > ./logs/Ablation_Cora_0Struc_HaveStruc_NoMC1.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 1 --if_keep_structure1 --sample_way 'random_drop' > ./logs/Ablation_Cora_2Struc_NoStruc_NoMC1.log 2>&1 &

nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'random_drop' > ./logs/Ablation_Citeseer_1Struc_NoStruc_HaveMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'keep_structure' > ./logs/Ablation_Citeseer_1Struc_HaveStruc_HaveMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'random_drop' > ./logs/Ablation_Citeseer_0Struc_NoStruc_HaveMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'keep_structure' > ./logs/Ablation_Citeseer_0Struc_HaveStruc_HaveMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 --if_keep_structure1 --sample_way 'random_drop' > ./logs/Ablation_Citeseer_2Struc_NoStruc_HaveMC.log 2>&1 &

nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'random_drop' > ./logs/Ablation_Citeseer_1Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'random_drop' > ./logs/Ablation_Citeseer_0Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'keep_structure' > ./logs/Ablation_Citeseer_1Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'keep_structure' > ./logs/Ablation_Citeseer_0Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 1 --if_keep_structure1 --sample_way 'random_drop' > ./logs/Ablation_Citeseer_2Struc_NoStruc_NoMC.log 2>&1 &

nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_1Struc_NoStruc_HaveMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'keep_structure' > ./logs/Ablation_Pubmed_1Struc_HaveStruc_HaveMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_0Struc_NoStruc_HaveMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'keep_structure' > ./logs/Ablation_Pubmed_0Struc_HaveStruc_HaveMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 --if_keep_structure1 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_2Struc_NoStruc_HaveMC.log 2>&1 &

nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_1Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_0Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'keep_structure' > ./logs/Ablation_Pubmed_1Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'keep_structure' > ./logs/Ablation_Pubmed_0Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 1 --if_keep_structure1 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_2Struc_NoStruc_NoMC.log 2>&1 &

# Node-level Base encoder
nohup python -u run_robust_acc.py --dataset 'Cora' --encoder_model 'Grace' --attack 'nettack' --device 0 --num_repeat 5 > ./logs/Cora_Base_Grace_Nettack_beta_rev.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'nettack' --device 2 --num_repeat 5 > ./logs/Pubmed_Base_Grace_Nettack_beta_rev.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'nettack' --device 0 --num_repeat 5 > ./logs/Citeseer_Base_Grace_Nettack_beta_rev.log 2>&1 &

nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 5 > ./logs/Cora_Base_Grace_Random.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --num_repeat 5 > ./logs/Pubmed_Base_Grace_Random_beta_rev.log 2>&1 &
nohup python -u run_robust_acc.py --dataset 'Citeseer' --encoder_model 'Grace' --attack 'random' --device 0 --num_repeat 5 > ./logs/Citeseer_Base_Grace_Random_beta_rev.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'Grace' --attack 'random' --device 0 --num_repeat 5 > ./logs/Computers_Base_Grace_Random_beta1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Photo' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 5 > ./logs/Photo_Base_Grace_Random_beta.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'WikiCS' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 5 > ./logs/WikiCS_Base_Grace_Random_beta.log 2>&1 &
# nohup python -u run_smooth_node.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 1 --num_repeat 5 > ./logs/Physics_Base_Grace_Random_beta.log 2>&1 &


nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'BGRL' --attack 'random' --device 3 --num_repeat 5 > ./logs/Cora_Base_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'BGRL' --attack 'random' --device 2 --num_repeat 5 > ./logs/Pubmed_Base_BGRL_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'BGRL' --attack 'random' --device 0 --num_repeat 5 > ./logs/Computers_Base_BGRL_Random.log 2>&1 &

nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'DGI' --attack 'random' --device 3 --num_repeat 5 > ./logs/Cora_Base_DGI_Random1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'DGI' --attack 'random' --device 2 --num_repeat 5 > ./logs/Pubmed_Base_DGI_Random1.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'DGI' --attack 'random' --device 0 --num_repeat 5 > ./logs/Computers_Base_DGI_Random1.log 2>&1 &

# Calculate Certified Accuracy
nohup python -u run_certify_node.py --dataset 'Cora' --encoder_model 'Grace' --device 1 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Cora_Certify_Grace_re5n020n200alpha1_1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Citeseer' --encoder_model 'Grace' --device 1 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Citeseer_Certify_Grace_re5n020n200alpha1_1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Pubmed' --encoder_model 'Grace' --device 2 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Pubmed_Certify_Grace_re5n020n200alpha1_1.log 2>&1 &

# Traditional method
nohup python -u run_smooth_node.py --dataset 'Cora' --encoder_model 'GAE' --attack 'random' --device 1 --num_repeat 5 > ./logs/Cora_GAE_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Pubmed' --encoder_model 'GAE' --attack 'random' --device 3 --num_repeat 5 > ./logs/Pubmed_GAE_Random.log 2>&1 &
nohup python -u run_smooth_node.py --dataset 'Computers' --encoder_model 'GAE' --attack 'random' --device 2 --num_repeat 5 > ./logs/Computers_GAE_Random.log 2>&1 &


# Graph-level Smoothed encoder
nohup python -u run_graphcl.py --dataset 'PROTEINS' --encoder_model 'GraphCL' --attack 'random' --device 1 --if_smoothed --num_repeat 5 > ./logs/PROTEINS_Smooth_GraphCL_random_beta_rev1.log 2>&1 &
nohup python -u run_graphcl.py --dataset 'MUTAG' --encoder_model 'GraphCL' --attack 'random' --device 2 --if_smoothed --num_repeat 5 > ./logs/MUTAG_Smooth_GraphCL_random_beta_rev1.log 2>&1 &
nohup python -u run_graphcl.py --dataset 'ENZYMES' --encoder_model 'GraphCL' --attack 'random' --device 0 --if_smoothed --num_repeat 5 > ./logs/ENZYMES_Smooth_GraphCL_random_beta_rev1.log 2>&1 &
nohup python -u run_graphcl.py --dataset 'COLLAB' --encoder_model 'GraphCL' --attack 'random' --device 3 --if_smoothed --num_repeat 5 > ./logs/COLLAB_Smooth_GraphCL_random_beta_rev1.log 2>&1 &

python -u run_graphcl.py --dataset 'MUTAG' --encoder_model 'GraphCL' --attack 'random' --device 0 --num_repeat 5
# Graph-level Base encoder
nohup python -u run_graphcl.py --dataset 'PROTEINS' --encoder_model 'GraphCL' --attack 'random' --device 2 --num_repeat 5 > ./logs/PROTEINS_Base_GraphCL_random_beta_rev1.log 2>&1 &
nohup python -u run_graphcl.py --dataset 'MUTAG' --encoder_model 'GraphCL' --attack 'random' --device 2 --num_repeat 5 > ./logs/MUTAG_Base_GraphCL_random_beta_rev1.log 2>&1 &
nohup python -u run_graphcl.py --dataset 'COLLAB' --encoder_model 'GraphCL' --attack 'random' --device 3 --num_repeat 5 > ./logs/COLLAB_Base_GraphCL_random_beta_rev1.log 2>&1 &
nohup python -u run_graphcl.py --dataset 'ENZYMES' --encoder_model 'GraphCL' --attack 'random' --device 3 --num_repeat 5 > ./logs/ENZYMES_Base_GraphCL_random_beta_rev1.log 2>&1 &

# Calculate Certified Accuracy
nohup python -u run_certify_node.py --dataset 'Cora' --encoder_model 'Grace' --device 1 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Cora_Certify_Grace_re5n020n200alpha1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Citeseer' --encoder_model 'Grace' --device 1 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Citeseer_Certify_Grace_re5n020n200alpha1.log 2>&1 &
nohup python -u run_certify_node.py --dataset 'Pubmed' --encoder_model 'Grace' --device 2 --num_repeat 5 --n0 20 --n 200 --alpha 0.01 > ./logs/Pubmed_Certify_Grace_re5n020n200alpha1.log 2>&1 &
# 
nohup python -u run_gnn.py --dataset 'Cora' --model 'GNNGuard' --attack 'none' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GNNGuard_clean.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Cora' --model 'GNNGuard' --attack 'random' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GNNGuard_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Cora' --model 'GNNGuard' --attack 'nettack' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GNNGuard_nettack.log 2>&1 &

nohup python -u run_gnn.py --dataset 'Pubmed' --model 'GNNGuard' --attack 'none' --device 1 --if_smoothed --num_repeat 5 > ./logs/Pubmed_GNNGuard_clean.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Pubmed' --model 'GNNGuard' --attack 'random' --device 1 --if_smoothed --num_repeat 5 > ./logs/Pubmed_GNNGuard_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Pubmed' --model 'GNNGuard' --attack 'nettack' --device 2 --if_smoothed --num_repeat 5 > ./logs/Pubmed_GNNGuard_nettack.log 2>&1 &

nohup python -u run_gnn.py --dataset 'Citeseer' --model 'GNNGuard' --attack 'none' --device 2 --if_smoothed --num_repeat 5 > ./logs/Citeseer_GNNGuard_clean.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Citeseer' --model 'GNNGuard' --attack 'random' --device 2 --if_smoothed --num_repeat 5 > ./logs/Citeseer_GNNGuard_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Citeseer' --model 'GNNGuard' --attack 'nettack' --device 3 --if_smoothed --num_repeat 5 > ./logs/Citeseer_GNNGuard_nettack.log 2>&1 &


nohup python -u run_gnn.py --dataset 'Cora' --model 'GCN' --attack 'none' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GCN_clean.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Cora' --model 'GCN' --attack 'random' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GCN_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Cora' --model 'GCN' --attack 'nettack' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GCN_nettack.log 2>&1 &

nohup python -u run_gnn.py --dataset 'Pubmed' --model 'GCN' --attack 'none' --device 1 --if_smoothed --num_repeat 5 > ./logs/Pubmed_GCN_clean.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Pubmed' --model 'GCN' --attack 'random' --device 1 --if_smoothed --num_repeat 5 > ./logs/Pubmed_GCN_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Pubmed' --model 'GCN' --attack 'nettack' --device 1 --if_smoothed --num_repeat 5 > ./logs/Pubmed_GCN_nettack.log 2>&1 &

nohup python -u run_gnn.py --dataset 'Citeseer' --model 'GCN' --attack 'none' --device 2 --if_smoothed --num_repeat 5 > ./logs/Citeseer_GCN_clean.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Citeseer' --model 'GCN' --attack 'random' --device 2 --if_smoothed --num_repeat 5 > ./logs/Citeseer_GCN_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Citeseer' --model 'GCN' --attack 'nettack' --device 2 --if_smoothed --num_repeat 1 > ./logs/Citeseer_GCN_nettack.log 2>&1 &
# nohup python -u run_gnn.py --dataset 'Cora' --model 'RobustGCN' --attack 'none' --device 0 --if_smoothed > ./logs/Cora_RobustGCN_clean.log 2>&1 &

nohup python -u run_gnn.py --dataset 'Cora' --model 'GCN' --attack 'random' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GCN_random.log 2>&1 &
nohup python -u run_gnn.py --dataset 'Cora' --model 'GCN' --attack 'nettack' --device 0 --if_smoothed --num_repeat 5 > ./logs/Cora_GCN_nettack.log 2>&1 &

nohup python -u run_Node2Vec.py --dataset 'Cora' --attack 'random' --num_repeat 5 --device 1 > Cora_Node2Vec_Random.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Citeseer' --attack 'random' --num_repeat 5 --device 1 > Citeseer_Node2Vec_Random.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Pubmed' --attack 'random' --num_repeat 5 --device 1 > Pubmed_Node2Vec_Random.log 2>&1 &

python train_edge_noise.py --dataset cora --outdir ./outdir --predictfile ./preidctdir/predict_cora.log  --certifyfile ./certifydir/certify_cora.log

scale * sqrt(2*log(2/alpha))/beta


nohup python threshold.py --fn Cora --a 10 > threshold_cora_10.log 2>&1 &
nohup python threshold.py --fn Cora --a 20 > threshold_cora_20.log 2>&1 &
nohup python threshold.py --fn Cora --a 30 > threshold_cora_30.log 2>&1 &
nohup python threshold.py --fn Cora --a 40 > threshold_cora_40.log 2>&1 &
nohup python threshold.py --fn Cora --a 50 > threshold_cora_50.log 2>&1 &
nohup python threshold.py --fn Cora --a 60 > threshold_cora_60.log 2>&1 &
nohup python threshold.py --fn Cora --a 70 > threshold_cora_70.log 2>&1 &
nohup python threshold.py --fn Cora --a 80 > threshold_cora_80.log 2>&1 &
nohup python threshold.py --fn Cora --a 90 > threshold_cora_90.log 2>&1 &

nohup python threshold.py --fn Citeseer --a 10 > threshold_Citeseer_10.log 2>&1 &
nohup python threshold.py --fn Citeseer --a 20 > threshold_Citeseer_20.log 2>&1 &
nohup python threshold.py --fn Citeseer --a 30 > threshold_Citeseer_30.log 2>&1 &
nohup python threshold.py --fn Citeseer --a 40 > threshold_Citeseer_40.log 2>&1 &
nohup python threshold.py --fn Citeseer --a 50 > threshold_Citeseer_50.log 2>&1 &
nohup python threshold.py --fn Citeseer --a 60 > threshold_Citeseer_60.log 2>&1 &
nohup python threshold.py --fn Citeseer --a 70 > threshold_Citeseer_70.log 2>&1 &
nohup python threshold.py --fn Citeseer --a 80 > threshold_Citeseer_80.log 2>&1 &
nohup python threshold.py --fn Citeseer --a 90 > threshold_Citeseer_90.log 2>&1 &

nohup python threshold.py --fn Pubmed --a 10 > threshold_Pubmed_10.log 2>&1 &
nohup python threshold.py --fn Pubmed --a 20 > threshold_Pubmed_20.log 2>&1 &
nohup python threshold.py --fn Pubmed --a 30 > threshold_Pubmed_30.log 2>&1 &
nohup python threshold.py --fn Pubmed --a 40 > threshold_Pubmed_40.log 2>&1 &
nohup python threshold.py --fn Pubmed --a 50 > threshold_Pubmed_50.log 2>&1 &
nohup python threshold.py --fn Pubmed --a 60 > threshold_Pubmed_60.log 2>&1 &
nohup python threshold.py --fn Pubmed --a 70 > threshold_Pubmed_70.log 2>&1 &
nohup python threshold.py --fn Pubmed --a 80 > threshold_Pubmed_80.log 2>&1 &
nohup python threshold.py --fn Pubmed --a 90 > threshold_Pubmed_90.log 2>&1 &