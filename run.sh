#!/bin/bash 

# Experiment for barlow twins

# nohup python train.py --config configs/barlow_twins.c10.yaml --gpu 2 --model resnet18 --epochs 350 --epochs_lin 100 --save_path bt.c10.r18.e350.pth > logs/bt.c10.r18.e350.log &

# nohup python train.py --config configs/barlow_twins.c100.yaml --gpu 3 --model resnet18 --epochs 350 --epochs_lin 100 --save_path bt.c100.r18.e350.pth > logs/bt.c100.r18.e350.log &

# nohup python train.py --config configs/barlow_twins.c10.yaml --gpu 4 --model resnet50 --epochs 350 --epochs_lin 100 --save_path bt.c10.r50.e350.pth > logs/bt.c10.r50.e350.log &

# nohup python train.py --config configs/barlow_twins.c100.yaml --gpu 5 --model resnet50 --epochs 350 --epochs_lin 100 --save_path bt.c100.r50.e350.pth > logs/bt.c100.r50.e350.log &

# # Experiment for byol 

# nohup python train.py --config configs/byol.c10.yaml --gpu 3 --model resnet18 --epochs 350 --epochs_lin 100 --save_path byol.c10.r18.e350.pth > logs/byol.c10.r18.e350.log &

# nohup python train.py --config configs/byol.c100.yaml --gpu 3 --model resnet18 --epochs 350 --epochs_lin 100 --save_path byol.c100.r18.e350.pth > logs/byol.c100.r18.e350.log &

# nohup python train.py --config configs/byol.c10.yaml --gpu 3 --model resnet50 --epochs 350 --epochs_lin 100 --save_path byol.c10.r50.e350.pth > logs/byol.c10.r50.e350.log &

# nohup python train.py --config configs/byol.c100.yaml --gpu 2 --model resnet50 --epochs 350 --epochs_lin 100 --save_path byol.c100.r50.e350.pth > logs/byol.c100.r50.e350.log &

# Experiment for simsiam 

# nohup python train.py --config configs/simsiam.c10.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path simsiam.c10.r18.e350.pth > logs/simsiam.c10.r18.e350.log &

# nohup python train.py --config configs/simsiam.c100.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path simsiam.c100.r18.e350.pth > logs/simsiam.c100.r18.e350.log &

# nohup python train.py --config configs/simsiam.c10.yaml --gpu 7 --model resnet50 --epochs 350 --epochs_lin 100 --save_path simsiam.c10.r50.e350.pth > logs/simsiam.c10.r50.e350.log &

# nohup python train.py --config configs/simsiam.c100.yaml --gpu 4 --model resnet50 --epochs 350 --epochs_lin 100 --save_path simsiam.c100.r50.e350.pth > logs/simsiam.c100.r50.e350.log &

# Experiment for simclr

nohup python train.py --config configs/simclr.c10.yaml --mlp_type linear --gpu 2 --model resnet18 --epochs 350 --epochs_lin 100 --save_path simclr.c10.r18.e350.pth > logs/simclr.c10.r18.e350.log &

nohup python train.py --config configs/simclr.c100.yaml --mlp_type linear --gpu 2 --model resnet18 --epochs 350 --epochs_lin 100 --save_path simclr.c100.r18.e350.pth > logs/simclr.c100.r18.e350.log &

nohup python train.py --config configs/simclr.c10.yaml --mlp_type linear --gpu 4 --model resnet50 --epochs 350 --epochs_lin 100 --save_path simclr.c10.r50.e350.pth > logs/simclr.c10.r50.e350.log &

nohup python train.py --config configs/simclr.c100.yaml --mlp_type linear --gpu 6 --model resnet50 --epochs 350 --epochs_lin 100 --save_path simclr.c100.r50.e350.pth > logs/simclr.c100.r50.e350.log &
