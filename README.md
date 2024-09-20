
# Efficient Relation-aware Neighborhood Aggregation in Graph Neural Networks via Tensor Decomposition

This is a PyTorch implementation of the paper.

## Requirements

To install the requirements, please use the "installation.sh" file. Run the following:
```
chmod +x installation.sh
./installation.sh
```

## Train a model from scratch an evaluate

To train TGCN on one of the datasets run the main.py file with the corresponding arguments. 

### Example: FB15k-237 

To train on FB15k-237 run the below commande:
```
python main.py --dataset fb15k-237 --gpu 0 --evaluate-after 60000 --n-epochs 60010 --graph-batch-size 100000 --dim-r 125 \
--lr 0.005 --dr-input 0.0 --dr-hid1 0.1 --dr-hid2 0.0 --dr-output 0.2 --dr-decoder 0.3
```

### Example: WN18RR

To train on WN18RE run the below commande:
```
python main.py --dataset wn18rr --gpu 0 --evaluate-after 34000 --n-epochs 34010 --graph-batch-size 50000 --dim-r 125 \
--lr 0.001 --dr-input 0.0 --dr-hid1 0.0 --dr-hid2 0.0 --dr-output 0.3 --dr-decoder 0.3
```

