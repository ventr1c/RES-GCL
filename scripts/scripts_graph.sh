python -u run_certify_graph.py --dataset ogbg-molhiv --encoder_model BGRL-G2L --device_id 0 --num_repeat 1 --n0 20 --n 200 --alpha 0.01 --batch_size 1024
python -u run_certify_graph.py --dataset ogbg-molhiv --encoder_model GraphCL --device_id 1 --num_repeat 1 --n0 50 --n 200 --alpha 0.01 --batch_size 1024 

