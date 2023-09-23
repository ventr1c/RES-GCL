# Certifiably Robust Graph Contrastive Learning
An official PyTorch implementation of "Certifiably Robust Graph Contrastive Learning" (NeurIPS 2023). [[paper]]()

## Running the code
For instance, to check the performance of our RES-GRACE on clean Cora graph, run the following code:

```
python run_smooth_node.py --if_smoothed --encoder_model GRACE --dataset Cora --attack none
```

For PRBCD-perturbed Cora graph, run the following code:

```
python run_smooth_node.py --if_smoothed --encoder_model GRACE --dataset Cora --attack PRBCD
```
