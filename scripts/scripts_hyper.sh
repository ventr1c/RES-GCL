
#!/bin/bash
# alphas=(0.001 0.01 0.05 0.10)
# ns=(200 500 1000)

# alphas=(0.0001 0.001 0.01 0.05 0.10)
betas=(0.001 0.1 0.3 0.5 0.9)
ns=(20 50 200 500 1000)

# for beta in ${betas[@]};
# do 
#     for n in ${ns[@]};
#     do
#         echo $n $alpha
#         python -u run_smooth_node_hyper.py \
#             --dataset='Cora' \
#             --encoder_model='Grace' \
#             --attack='PRBCD' \
#             --device=2 \
#             --if_smoothed \
#             --num_repeat=3 \
#             --num_sample=${n} \
#             --prob=${beta} 
#     done
# done
for beta in ${betas[@]};
do 
    for n in ${ns[@]};
    do
        echo $n $alpha
        python -u run_smooth_node_hyper.py \
            --dataset='Physics' \
            --encoder_model='Grace' \
            --attack='PRBCD' \
            --device=2 \
            --if_smoothed \
            --num_repeat=3 \
            --num_sample=${n} \
            --prob=${beta} \
            --cont_batch_size=2048 
    done
done

# for alpha in ${alphas[@]};
# do 
#     for n in ${ns[@]};
#     do
#         python -u run_certify_node_hyper.py \
#             --dataset='Cora' \
#             --encoder_model='Grace' \
#             --device=0 \
#             --num_repeat=1 \
#             --n0=20 \
#             --n=${n} \
#             --alpha=${alpha}
#     done
# done

# for alpha in ${alphas[@]};
# do 
#     for n in ${ns[@]};
#     do
#         echo $n $alpha
#         python -u run_certify_node_hyper.py \
#             --dataset='Cora' \
#             --encoder_model='Grace' \
#             --device=3 \
#             --num_repeat=3 \
#             --n0=20 \
#             --n=${n} \
#             --alpha=${alpha} \
#     done
# done