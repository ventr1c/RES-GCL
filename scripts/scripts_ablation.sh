## CLGA
nohup python -u run_smooth_node_flip.py --dataset 'Physics' --encoder_model 'Grace_flip' --attack 'random' --device 0 --num_repeat 3 --num_sample 20 --if_smoothed > ./logs/Physics/Smooth_Grace_nettack_flip.log 2>&1 &
nohup python -u run_smooth_node_flip.py --dataset 'Physics' --encoder_model 'Grace_flip' --attack 'nettack' --device 0 --num_repeat 3 --num_sample 20 --if_smoothed > ./logs/Physics/Smooth_Grace_nettack_flip.log 2>&1 &

nohup python -u run_smooth_node_flip.py --dataset 'Cora' --encoder_model 'Grace_flip' --attack 'nettack' --device 0 --num_repeat 3 --num_sample 20 --if_smoothed > ./logs/Cora/Smooth_Grace_nettack_flip1.log 2>&1 &
nohup python -u run_smooth_node_flip.py --dataset 'Pubmed' --encoder_model 'Grace_flip' --attack 'nettack' --device 0 --num_repeat 3 --num_sample 20 --if_smoothed > ./logs/Pubmed/Smooth_Grace_nettack_flip_0428.log 2>&1 &

# without noise
nohup python -u run_smooth_node_flip.py --dataset 'Cora' --encoder_model 'Grace_flip' --attack 'nettack' --if_keep_structure1 --sample_way 'random_drop' --device 2 --num_repeat 3 --num_sample 20 --if_smoothed > ./logs/Cora/Smooth_Grace_nettack_flip_2S.log 2>&1 &
nohup python -u run_smooth_node_flip.py --dataset 'Pubmed' --encoder_model 'Grace_flip' --attack 'nettack' --if_keep_structure1 --sample_way 'random_drop' --device 1 --num_repeat 3 --num_sample 20 --if_smoothed > ./logs/Pubmed/Smooth_Grace_nettack_flip_2S.log 2>&1 &




## Ablation
### Pubmed
nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_1Struc_NoStruc_HaveMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'keep_structure' > ./logs/Ablation_Pubmed_1Struc_HaveStruc_HaveMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_0Struc_NoStruc_HaveMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'keep_structure' > ./logs/Ablation_Pubmed_0Struc_HaveStruc_HaveMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 --if_keep_structure1 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_2Struc_NoStruc_HaveMC.log 2>&1 &

nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 0 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_1Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_0Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'keep_structure' > ./logs/Ablation_Pubmed_1Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'keep_structure' > ./logs/Ablation_Pubmed_0Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Pubmed' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 1 --if_keep_structure1 --sample_way 'random_drop' > ./logs/Ablation_Pubmed_2Struc_NoStruc_NoMC.log 2>&1 &
### Physics
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'random_drop' > ./logs/Ablation/Physics_1Struc_NoStruc_HaveMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'keep_structure' > ./logs/Ablation/Physics_1Struc_HaveStruc_HaveMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'random_drop' > ./logs/Ablation/Physics_0Struc_NoStruc_HaveMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 --if_ignore_structure2 --sample_way 'keep_structure' > ./logs/Ablation/Physics_0Struc_HaveStruc_HaveMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 --if_keep_structure1 --sample_way 'random_drop' > ./logs/Ablation/Physics_2Struc_NoStruc_HaveMC.log 2>&1 &

nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'random_drop' --cont_batch_size 2048 > ./logs/Ablation_Physics_1Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'random_drop' --cont_batch_size 2048 > ./logs/Ablation_Physics_0Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'keep_structure' --cont_batch_size 2048 > ./logs/Ablation_Physics_1Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'keep_structure' --cont_batch_size 2048 > ./logs/Ablation_Physics_0Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_keep_structure1 --sample_way 'random_drop' --cont_batch_size 2048 > ./logs/Ablation_Physics_2Struc_NoStruc_NoMC.log 2>&1 &


nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'PRBCD' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'random_drop' --cont_batch_size 2048 > ./logs/Ablation_Physics_1Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'PRBCD' --attack 'random' --device 1 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'random_drop' --cont_batch_size 2048 > ./logs/Ablation_Physics_0Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'PRBCD' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'keep_structure' --cont_batch_size 2048 > ./logs/Ablation_Physics_1Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'PRBCD' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'keep_structure' --cont_batch_size 2048 > ./logs/Ablation_Physics_0Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Physics' --encoder_model 'PRBCD' --attack 'random' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_keep_structure1 --sample_way 'random_drop' --cont_batch_size 2048 > ./logs/Ablation_Physics_2Struc_NoStruc_NoMC.log 2>&1 &


# Cora
nohup python -u run_smooth_node_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 --if_keep_structure1 --sample_way 'random_drop'  > ./logs/Ablation_Cora_NoRES.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 5 --num_sample 20 --sample_way 'random_drop'  > ./logs/Ablation_Cora_HaveRES.log 2>&1 &

nohup python -u run_smooth_node_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 1 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'random_drop'  > ./logs/Ablation_Cora_1Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'random_drop'  > ./logs/Ablation_Cora_0Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --sample_way 'keep_structure'  > ./logs/Ablation_Cora_1Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_ignore_structure2 --sample_way 'keep_structure'  > ./logs/Ablation_Cora_0Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'Cora' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 5 --num_sample 1 --if_keep_structure1 --sample_way 'random_drop'  > ./logs/Ablation_Cora_2Struc_NoStruc_NoMC.log 2>&1 &

# Arxiv
nohup python -u run_smooth_node_ablation.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 0 --if_smoothed --num_repeat 2 --num_sample 20 --if_keep_structure1 --sample_way 'random_drop' --cont_batch_size 512  > ./logs/Ablation_ogbn-arxiv_NoRES_0522.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 2 --num_sample 20 --sample_way 'random_drop' --cont_batch_size 512  > ./logs/Ablation_ogbn-arxiv_HaveRES_0522.log 2>&1 &

nohup python -u run_smooth_node_ablation.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 1 --if_smoothed --num_repeat 1 --num_sample 1 --sample_way 'random_drop' --cont_batch_size 512 > ./logs/Ablation_ogbn-arxiv_1Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 1 --num_sample 1 --if_ignore_structure2 --sample_way 'random_drop' --cont_batch_size 512 > ./logs/Ablation_ogbn-arxiv_0Struc_NoStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 1 --num_sample 1 --sample_way 'keep_structure' --cont_batch_size 512 > ./logs/Ablation_ogbn-arxiv_1Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 1 --num_sample 1 --if_ignore_structure2 --sample_way 'keep_structure' --cont_batch_size 512 > ./logs/Ablation_ogbn-arxiv_0Struc_HaveStruc_NoMC.log 2>&1 &
nohup python -u run_smooth_node_ablation.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'PRBCD' --device 3 --if_smoothed --num_repeat 1 --num_sample 1 --if_keep_structure1 --sample_way 'random_drop' --cont_batch_size 512 > ./logs/Ablation_ogbn-arxiv_2Struc_NoStruc_NoMC.log 2>&1 &
