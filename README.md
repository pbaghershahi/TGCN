
# Efficient Relation-aware Neighborhood Aggregation in Graph Neural Networks via Tensor Decomposition

This is a PyTorch implementation of the paper.

## Requirements

To install the requirements, please use the "installation.sh" file. Run the following:
```
chmod +x installation.sh
./installation.sh
```

## Train a model from scratch and evaluate

To train TGCN on one of the datasets run the main.py file with the corresponding config file. There is a config file for every used dataset in the ```config``` directory. The default parameters are used for the paper's main experiments. The other settings of hyper-parameters are available in the paper which you can change the config file.

### Example: FB15k-237 

To train on FB15k-237 run the below commande:
```
python main.py --config-from-file ./config/fb15k_237.yaml
```

### Example: WN18RR

To train on WN18RE run the below commande:
```
python main.py --config-from-file ./config/wn18rr.yaml
```

## Supported Datasets
Dataset | path |
:--- | :---: |
Fb15k-237 | ```./config/enzymes.yaml``` |
WN18RR | ```./config/proteins.yaml``` |

## Supported Decoding Methods
Methods | method name |
:--- | :---: |
TuckER | ```tucker``` |
DistMult | ```distmult``` |

To change the decoding function use the ```--decoder``` argument.

