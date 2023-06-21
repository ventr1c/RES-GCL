nohup python -u run_smooth_node_clga.py --dataset 'ogbn-arxiv' --encoder_model 'Grace' --attack 'random' --device 3 --if_smoothed --num_repeat 3 --num_sample 20 --cont_batch_size 2048 > ./logs/ogbn-arxiv/Smooth_Grace_Random_0420.log 2>&1 &

# PRBCD
nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'Grace' --attack 'CLGA' --device 3 --num_repeat 5 --num_sample 20 > ./logs/Cora/Base_Grace_CLGA10.log 2>&1 &
nohup python -u run_smooth_node_clga.py --dataset 'Cora' --encoder_model 'Grace-Jaccard' --attack 'PRBCD' --device 3 --num_repeat 3 --num_sample 20 > ./logs/Physics/Grace-Jaccard_PRBCD.log 2>&1 &