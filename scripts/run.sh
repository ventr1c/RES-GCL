device_id=0
device_id2=3
# cont_weights=(0.001 0.01 0.1 1 2)
cont_weights=(10 50 100)
# nohup sh pre_noisy.sh ${cont_weight} > noisy_cora_
# for cont_weight in ${cont_weights[@]};
# do
#     filename1="2_noisy_cora_$(echo $cont_weight).txt"
#     nohup bash ./scripts/pre_noisy_cora.sh ${cont_weight} ${device_id} > ./logs/${filename1} 2>&1 & 
#     filename2="2_bkd_cora_$(echo $cont_weight).txt"
#     nohup bash ./scripts/pre_bkd_cora.sh ${cont_weight} ${device_id2} > ./logs/${filename2} 2>&1 & 
#     # echo $filename
# done

for cont_weight in ${cont_weights[@]};
do
    # filename1="noisy_pubmed_$(echo $cont_weight).txt"
    # bash ./scripts/pre_noisy_pubmed.sh ${cont_weight} ${device_id} > ./logs/${filename1}
    filename2="bkd_pubmed_$(echo $cont_weight).txt"
    bash ./scripts/pre_bkd_pubmed.sh ${cont_weight} ${device_id2} > ./logs/${filename2}
    # echo $filename
done