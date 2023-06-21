nohup python -u run_certify_graph.py --dataset ogbg-molhiv --encoder_model BGRL-G2L --device_id 0 --num_repeat 1 --n0 20 --n 200 --alpha 0.01 --batch_size 1024 > ./logs/ogbg-molhiv/certify_BGRL_0521.log 2>&1 &
nohup python -u run_certify_graph.py --dataset PROTEINS --encoder_model BGRL-G2L --device_id 0 --num_repeat 2 --n0 50 --n 200 --alpha 0.01 --batch_size 1024 > ./logs/PROTEINS/certify_BGRL3.log 2>&1 &
nohup python -u run_certify_graph.py --dataset MUTAG --encoder_model BGRL-G2L --device_id 0 --num_repeat 5 --n0 50 --n 200 --alpha 0.01 --batch_size 128 > ./logs/MUTAG/certify_BGRL2.log 2>&1 &


nohup python -u run_certify_graph.py --dataset ogbg-molhiv --encoder_model GraphCL --device_id 1 --num_repeat 1 --n0 50 --n 200 --alpha 0.01 --batch_size 256 > ./logs/ogbg-molhiv/certify_GraphCL_0506_batch256.log 2>&1 &
nohup python -u run_certify_graph.py --dataset PROTEINS --encoder_model GraphCL --device_id 3 --num_repeat 2 --n0 50 --n 200 --alpha 0.01 --batch_size 1024 > ./logs/PROTEINS/certify_GraphCL3.log 2>&1 &
nohup python -u run_certify_graph.py --dataset MUTAG --encoder_model GraphCL --device_id 3 --num_repeat 5 --n0 50 --n 200 --alpha 0.01 --batch_size 128 > ./logs/MUTAG/certify_GraphCL2.log 2>&1 &

# Random
nohup python -u run_smooth_graph.py --dataset 'MUTAG' --encoder_model 'GraphCL' --attack 'random_global' --device 2 --num_repeat 5 --num_sample 20 --batch_size 128 > ./logs/MUTAG/Base_GraphCL_Random.log 2>&1 &
nohup python -u run_smooth_graph.py --dataset 'MUTAG' --encoder_model 'BGRL-G2L' --attack 'random_global' --device 1 --num_repeat 5 --num_sample 20 --batch_size 128 > ./logs/MUTAG/Base_BGRL-G2L_Random.log 2>&1 &
nohup python -u run_smooth_graph.py --dataset 'MUTAG' --encoder_model 'GraphCL' --attack 'random_global' --device 2 --if_smoothed --num_repeat 5 --num_sample 20 --batch_size 128 > ./logs/MUTAG/Smooth_GraphCL_Random1.log 2>&1 &
nohup python -u run_smooth_graph.py --dataset 'MUTAG' --encoder_model 'BGRL-G2L' --attack 'random_global' --device 1 --if_smoothed --num_repeat 5 --num_sample 20 --batch_size 128 > ./logs/MUTAG/Smooth_BGRL-G2L_Random.log 2>&1 &

nohup python -u run_smooth_graph.py --dataset 'PROTEINS' --encoder_model 'BGRL-G2L' --attack 'random_global' --device 0 --num_repeat 5 --num_sample 20 > ./logs/PROTEINS/Base_BGRL-G2L_Random_0521.log 2>&1 &
nohup python -u run_smooth_graph.py --dataset 'PROTEINS' --encoder_model 'GraphCL' --attack 'random_global' --device 0 --num_repeat 5 --num_sample 20 > ./logs/PROTEINS/Base_GraphCL_Random_0521.log 2>&1 &
nohup python -u run_smooth_graph.py --dataset 'PROTEINS' --encoder_model 'BGRL-G2L' --attack 'random_global' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/PROTEINS/Smoothed_BGRL-G2L_Random_0521_1.log 2>&1 &
nohup python -u run_smooth_graph.py --dataset 'PROTEINS' --encoder_model 'GraphCL' --attack 'random_global' --device 0 --if_smoothed --num_repeat 5 --num_sample 20 > ./logs/PROTEINS/Smoothed_GraphCL_Random_0521_1.log 2>&1 &

nohup python -u run_smooth_graph.py --dataset 'ogbg-molhiv' --encoder_model 'GraphCL' --attack 'random_global' --device 1 --if_smoothed --num_repeat 3 --num_sample 20 --batch_size 1024 > ./logs/ogbg-molhiv/Smooth_GraphCL_Random_0521.log 2>&1 &
nohup python -u run_smooth_graph.py --dataset 'ogbg-molhiv' --encoder_model 'GraphCL' --attack 'random_global' --device 3 --num_repeat 3 --num_sample 20 --batch_size 1024 > ./logs/ogbg-molhiv/Base_GraphCL_Random_0521.log 2>&1 &
nohup python -u run_smooth_graph.py --dataset 'ogbg-molhiv' --encoder_model 'BGRL-G2L' --attack 'random_global' --device 3 --num_repeat 3 --num_sample 20 --batch_size 1024 > ./logs/ogbg-molhiv/Base_BGRL-G2L_Random_0521.log 2>&1 &
nohup python -u run_smooth_graph.py --dataset 'ogbg-molhiv' --encoder_model 'BGRL-G2L' --attack 'random_global' --device 0 --if_smoothed --num_repeat 3 --num_sample 20 --batch_size 128 > ./logs/ogbg-molhiv/Smooth_BGRL-G2L_Random_0521.log 2>&1 &