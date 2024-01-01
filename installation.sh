#!/bin/bash

apt-get update;
apt-get install -y vim;
apt-get install -y git;
apt-get install -y screen;
apt-get install -y htop;
cd /workspace;
python -m venv env;
source env/bin/activate;
git clone https://pbaghershahi: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@github.com/pbaghershahi/TGCN.git;
mv TGCN/ tgcn;
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118;
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0%2Bcu118.html;
pip install -U "ray[data,train,tune,serve]";
