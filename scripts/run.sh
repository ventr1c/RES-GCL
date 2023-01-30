device_id=2
device_id2=0
cont_weights=(0.001 0.01 0.1 1 2)
# cont_weights=(10 50 100)
# nohup sh pre_noisy.sh ${cont_weight} > noisy_cora_
for cont_weight in ${cont_weights[@]};
do
    # filename1="noisy_cora_$(echo $cont_weight)_noMLP.txt"
    # nohup bash ./scripts/pre_noisy_cora.sh ${cont_weight} ${device_id} > ./${filename1} 2>&1 & 
    filename2="bkd_cora_$(echo $cont_weight)_noMLP.txt"
    nohup bash ./scripts/pre_bkd_cora.sh ${cont_weight} ${device_id2} > ./${filename2} 2>&1 & 
    echo $filename
done

# for cont_weight in ${cont_weights[@]};
# do
#     filename1="3_noisy_pubmed_$(echo $cont_weight).txt"
#     nohup bash ./scripts/pre_noisy_pubmed.sh ${cont_weight} ${device_id} > ./logs/${filename1} 2>&1 &
#     # filename2="4_bkd_pubmed_$(echo $cont_weight).txt"
#     # bash ./scripts/pre_bkd_pubmed.sh ${cont_weight} ${device_id2} > ./logs/${filename2} 2>&1 &
#     # echo $filename
# done