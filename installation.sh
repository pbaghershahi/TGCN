#!/bin/bash

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121;
pip install --no-index pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$2.1.0+cu121.html;
pip install torch-geometric;
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torcheval;
pip install torchmetrics;
pip install matplotlib;
pip install pandas;
pip install ipdb;
pip install gdown;
pip install notebook;
pip install pyyaml;
pip install -U "ray[data,train,tune,serve]";
pip uninstall fsspec -y;
pip install --force-reinstall -v "fsspec==2024.3.1"
printf "\033c";
