#!/bin/bash

apt-get update;
apt-get install vim;
apt-get install git;
apt-get install screen;
apt-get install htop;
cd /workspace;
python -m venv env;
source env/bin/activate;
git clone https://pbaghershahi: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@github.com/pbaghershahi/TGCN.git;
mv TGCN/ tgcn;
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118;
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu118.html;
pip install -U "ray[data,train,tune,serve]";
screen -S tgcn;
