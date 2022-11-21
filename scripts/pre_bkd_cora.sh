device_id=$2
# cont_weights=(0.001 0.01 0.1 1 2 10 50 100)
cont_weight=$1
# num_epochs=(1000 2000)
num_epoch=3000
# learning_rate=(0.0001 0.0002 0.0005 0.001)
drop_edge_rate_1=(0.3 0.4 0.5)
drop_edge_rate_2=(0 0.2 0.3 0.4)
# drop_feature_rate_1=(0.4 0.3 0.5)
# drop_feature_rate_2=(0.4 0.3 0.5)
drop_feature_rate_1=(0.4)
drop_feature_rate_2=(0 0.4 0.3 0.5)
learning_rate=0.0002
weight_decay=0.00001
cont_batch_size=0
num_hidden=64
num_proj_hidden=64
cl_num_proj_hidden=64
vs_number=5
homo_loss_weights=(0 100)
homo_boost_thrd=0.5 
trojan_epochs=200
tau=0.4
selection_method='cluster_degree'

# for cont_weight in ${cont_weights[@]};
# do
#     for epoch in ${num_epochs[@]};
#     do
#         for der1 in ${drop_edge_rate_1[@]};
#         do
#             for der2 in ${drop_edge_rate_2[@]};
#             do
#                 for dfr1 in ${drop_feature_rate_1[@]};
#                 do
#                     for dfr2 in ${drop_feature_rate_2[@]};
#                     do
#                         if [ $der1<=$der2 ] ; then 
#                             python -u ../run_pre_noisy.py \
#                                 --cl_lr=${learning_rate}\
#                                 --cl_num_hidden=128\
#                                 --cl_num_proj_hidden=128\
#                                 --cl_num_layers=2\
#                                 --cl_activation='rele'\
#                                 --cl_base_model='GCNConv'\
#                                 --cont_weight=${cont_weight}\
#                                 --add_edge_rate_1=0\
#                                 --add_edge_rate_2=0\
#                                 --drop_edge_rate_1=${der1}\
#                                 --drop_edge_rate_2=${der2}\
#                                 --drop_feat_rate_1=${dfr1}\
#                                 --drop_feat_rate_2=${dfr2}\
#                                 --tau=${tau}\
#                                 --cl_num_epochs=${epoch}\
#                                 --cl_weight_decay=${weight_decay}\
#                                 --cont_batch_size=${cont_batch_size}\
#                                 --vs_number=5\
#                                 --homo_loss_weight=100\
#                                 --homo_boost_thrd=0.5\
#                                 --trojan_epochs=200\
#                         fi
#                     done
#                 done
#             done
#         done
#     done
# done
for homo_loss_weight in ${homo_loss_weights[@]};
do
    for der1 in ${drop_edge_rate_1[@]};
    do
        for der2 in ${drop_edge_rate_2[@]};
        do
            for dfr1 in ${drop_feature_rate_1[@]};
            do
                for dfr2 in ${drop_feature_rate_2[@]};
                do
                    echo "Begin epxeriment: cont_weight: ${cont_weight} epoch:${epoch} der1:${der1} der2:${der2} dfr1:${dfr1} dfr2:${dfr2}"
                    python -u ./run_pre_bkd.py \
                        --dataset='Cora'\
                        --cl_lr=${learning_rate}\
                        --num_hidden=64\
                        --num_proj_hidden=64\
                        --cl_num_proj_hidden=64\
                        --cl_num_layers=2\
                        --cl_activation='relu'\
                        --cl_base_model='GCNConv'\
                        --cont_weight=${cont_weight}\
                        --add_edge_rate_1=0\
                        --add_edge_rate_2=0\
                        --drop_edge_rate_1=${der1}\
                        --drop_edge_rate_2=${der2}\
                        --drop_feat_rate_1=${dfr1}\
                        --drop_feat_rate_2=${dfr2}\
                        --tau=${tau}\
                        --cl_num_epochs=${num_epoch}\
                        --cl_weight_decay=${weight_decay}\
                        --cont_batch_size=${cont_batch_size}\
                        --vs_number=5\
                        --homo_loss_weight=${homo_loss_weight}\
                        --homo_boost_thrd=0.5\
                        --trojan_epochs=200\
                        --device_id=${device_id}
                done
            done
        done
    done
done