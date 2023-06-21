# Transductive
# Node2vec
nohup python -u run_Node2Vec.py --dataset 'Cora' --encoder_model 'Node2vec' --attack 'nettack' --device 1 --num_repeat 5 > ./logs/Cora/Node2vec_nettack.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Pubmed' --encoder_model 'Node2vec' --attack 'nettack' --device 1 --num_repeat 3 > ./logs/Pubmed/Node2vec_nettack1.log 2>&1 &

## Random attack 
nohup python -u run_Node2Vec.py --dataset 'Cora' --encoder_model 'Node2vec' --attack 'random' --device 1 --num_repeat 5 > ./logs/Cora/Node2vec_Random1.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Pubmed' --encoder_model 'Node2vec' --attack 'random' --device 3 --num_repeat 5 > ./logs/Pubmed/Node2vec_Random.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Computers' --encoder_model 'Node2vec' --attack 'random' --device 2 --num_repeat 5 > ./logs/Computers/Node2vec_Random.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'ogbn-arxiv' --encoder_model 'Node2vec' --attack 'random' --device 0 --num_repeat 3 > ./logs/ogbn-arxiv/Node2vec_Random.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Physics' --encoder_model 'Node2vec' --attack 'random' --device 0 --num_repeat 3 > ./logs/Physics/Node2vec_Random.log 2>&1 &
## PRBCD
nohup python -u run_Node2Vec.py --dataset 'Cora' --encoder_model 'Node2vec' --attack 'PRBCD' --device 1 --num_repeat 5 > ./logs/Cora/Node2vec_PRBCD.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Pubmed' --encoder_model 'Node2vec' --attack 'PRBCD' --device 3 --num_repeat 5 > ./logs/Pubmed/Node2vec_PRBCD.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Computers' --encoder_model 'Node2vec' --attack 'PRBCD' --device 2 --num_repeat 5 > ./logs/Computers/Node2vec_PRBCD.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'ogbn-arxiv' --encoder_model 'Node2vec' --attack 'PRBCD' --device 1 --num_repeat 3 > ./logs/ogbn-arxiv/Node2vec_PRBCD.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Physics' --encoder_model 'Node2vec' --attack 'PRBCD' --device 1 --num_repeat 3 > ./logs/Physics/Node2vec_PRBCD.log 2>&1 &

## Random attack global 
nohup python -u run_Node2Vec.py --dataset 'Cora' --encoder_model 'Node2vec' --attack 'random_global' --device 2 --num_repeat 5 > ./logs/Cora/Node2vec_random_global1.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Pubmed' --encoder_model 'Node2vec' --attack 'random_global' --device 1 --num_repeat 5 > ./logs/Pubmed/Node2vec_random_global.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'ogbn-arxiv' --encoder_model 'Node2vec' --attack 'random_global' --device 0 --num_repeat 3 > ./logs/ogbn-arxiv/Node2vec_random_global.log 2>&1 &
nohup python -u run_Node2Vec.py --dataset 'Physics' --encoder_model 'Node2vec' --attack 'random_global' --device 0 --num_repeat 3 > ./logs/Physics/Node2vec_random_global.log 2>&1 &
# nohup python -u run_Node2Vec.py --dataset 'Computers' --encoder_model 'Node2vec' --attack 'random_global' --device 0 --num_repeat 5 > ./logs/Computers/Node2vec_random_global.log 2>&1 &
