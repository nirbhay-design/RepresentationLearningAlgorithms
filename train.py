import sys, random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.network import Network, MLP, BYOL_mlp
from train_utils import yaml_loader, train_supcon, train_triplet, train_simsiam, \
                        train_byol, model_optimizer, \
                        loss_function, \
                        load_dataset

import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group
import os 

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = "4084"
    init_process_group(backend = 'nccl', rank = rank, world_size = world_size)

def train_network(
        train_algo,
        model,
        mlp,
        train_dl,
        train_dl_mlp,
        test_dl,
        loss,
        loss_mlp,
        optimizer,
        mlp_optimizer,
        opt_lr_schedular,
        eval_every,
        n_epochs,
        n_epochs_mlp,
        rank,
        eval_id,
        return_logs,
        target_net,
        pred_net,
        ema_tau):
    
    if train_algo == "supcon" or train_algo == "simclr":
        train_supcon(
            train_algo,
            model,
            mlp,
            train_dl,
            train_dl_mlp,
            test_dl,
            loss,
            loss_mlp,
            optimizer,
            mlp_optimizer,
            opt_lr_schedular,
            eval_every,
            n_epochs,
            n_epochs_mlp,
            rank,
            eval_id,
            return_logs)
    elif train_algo == "triplet":
        train_triplet(
            model,
            mlp,
            train_dl,
            train_dl_mlp,
            test_dl,
            loss,
            loss_mlp,
            optimizer,
            mlp_optimizer,
            opt_lr_schedular,
            eval_every,
            n_epochs,
            n_epochs_mlp,
            rank,
            eval_id,
            return_logs)
    elif train_algo == "simsiam":
        train_simsiam(
            model,
            mlp,
            train_dl,
            train_dl_mlp,
            test_dl,
            loss,
            loss_mlp,
            optimizer,
            mlp_optimizer,
            opt_lr_schedular,
            eval_every,
            n_epochs,
            n_epochs_mlp,
            rank,
            eval_id,
            return_logs)
    elif train_algo == 'byol':
        train_byol(
            model,
            target_net,
            pred_net,
            mlp,
            train_dl,
            train_dl_mlp,
            test_dl,
            loss,
            loss_mlp,
            optimizer,
            mlp_optimizer,
            opt_lr_schedular,
            ema_tau,
            eval_every,
            n_epochs,
            n_epochs_mlp,
            rank,
            eval_id,
            return_logs)

def main_single():
    algo = config['train_algo']
    
    pred_net = None
    target_net = None
    ema_tau = None
    if algo == 'byol':
        target_net = Network(**config['model_params'])
        pred_net = BYOL_mlp(**config["byol_pred_params"])
        ema_tau = config['ema_tau']

    model = Network(**config['model_params'])
    mlp = MLP(model.classifier_infeatures, config['num_classes'], config['mlp_type'])

    optimizer = model_optimizer(model, config['opt'], pred_net, **config['opt_params'])
    mlp_optimizer = model_optimizer(mlp, config['mlp_opt'], **config['mlp_opt_params'])

    opt_lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['schedular_params'])

    loss, loss_mlp = loss_function(loss_type = config['loss'], **config.get('loss_params', {}))
    
    train_dl, train_dl_mlp, test_dl, train_ds, test_ds = load_dataset(
        dataset_name = config['data_name'],
        distributed = False,
        **config['data_params'])
    
    print(f"# of Training Images: {len(train_ds)}")
    print(f"# of Testing Images: {len(test_ds)}")


    return_logs = config['return_logs']
    eval_every = config['eval_every']
    n_epochs = config['n_epochs']
    n_epochs_mlp = config['n_epochs_mlp']
    device = config['gpu_id']

    train_network(
        config['train_algo'],
        model,
        mlp,
        train_dl,
        train_dl_mlp,
        test_dl,
        loss,
        loss_mlp,
        optimizer,
        mlp_optimizer,
        opt_lr_schedular,
        eval_every,
        n_epochs,
        n_epochs_mlp,
        device,
        device,
        return_logs,
        target_net,
        pred_net,
        ema_tau
    )

if __name__ == "__main__":
    config = yaml_loader(sys.argv[1])
    
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("environment: ")
    print(f"YAML: {sys.argv[1]}")
    for key, value in config.items():
        print(f"==> {key}: {value}")

    print("-"*50)

    main_single()