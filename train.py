import sys, random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.network import Network, MLP
from train_utils import yaml_loader, train_supcon, train_simclr, train_triplet, \
                        model_optimizer, \
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
        test_dl,
        loss,
        optimizer,
        mlp_optimizer,
        opt_lr_schedular,
        eval_every,
        n_epochs,
        n_epochs_mlp,
        rank,
        eval_id,
        return_logs):
    
    if train_algo == "supcon":
        train_supcon(
            model,
            mlp,
            train_dl,
            test_dl,
            loss,
            optimizer,
            mlp_optimizer,
            opt_lr_schedular,
            eval_every,
            n_epochs,
            n_epochs_mlp,
            rank,
            eval_id,
            return_logs)
    elif train_algo == "simclr":
        train_simclr(
            model,
            mlp,
            train_dl,
            test_dl,
            loss,
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
            test_dl,
            loss,
            optimizer,
            mlp_optimizer,
            opt_lr_schedular,
            eval_every,
            n_epochs,
            n_epochs_mlp,
            rank,
            eval_id,
            return_logs)
    
def main_dist(rank, world_size, config):

    ddp_setup(rank, world_size)

    model = Network(**config['model_params'])
    mlp = MLP(model.classifier_infeatures, config['num_classes'], config['mlp_type'])
    print(f"############################# RANK: {rank} #############################")
    model = model.to(rank)
    mlp = mlp.to(rank)

    model = DDP(model, device_ids = [rank])
    mlp = DDP(mlp, device_ids=[rank])

    optimizer = model_optimizer(model, config['opt'], **config['opt_params'])
    mlp_optimizer = model_optimizer(mlp, config['mlp_opt'], **config['mlp_opt_params'])

    opt_lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['schedular_params'])

    loss = loss_function(loss_type = config['loss'], **config.get('loss_params', {}))
    
    train_dl, test_dl, train_ds, test_ds = load_dataset(
        dataset_name=config['data_name'],
        distributed = config['distributed'],
        **config['data_params'])
    
    if rank == 0:
        print(f"# of Training Images: {len(train_ds)}")
        print(f"# of Testing Images: {len(test_ds)}")


    return_logs = config['return_logs']
    eval_every = config['eval_every']
    n_epochs = config['n_epochs']
    n_epochs_mlp = config['n_epochs_mlp']

    eval_id = 0
    
    
    train_network(
        config['train_algo'],
        model,
        mlp,
        train_dl,
        test_dl,
        loss,
        optimizer,
        mlp_optimizer,
        opt_lr_schedular,
        eval_every,
        n_epochs,
        n_epochs_mlp,
        rank,
        eval_id,
        return_logs
    )

    destroy_process_group()

def main_single():
    model = Network(**config['model_params'])
    mlp = MLP(model.classifier_infeatures, config['num_classes'], config['mlp_type'])

    optimizer = model_optimizer(model, config['opt'], **config['opt_params'])
    mlp_optimizer = model_optimizer(mlp, config['mlp_opt'], **config['mlp_opt_params'])

    opt_lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['schedular_params'])

    loss = loss_function(loss_type = config['loss'], **config.get('loss_params', {}))
    
    train_dl, test_dl, train_ds, test_ds = load_dataset(
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
        test_dl,
        loss,
        optimizer,
        mlp_optimizer,
        opt_lr_schedular,
        eval_every,
        n_epochs,
        n_epochs_mlp,
        device,
        device,
        return_logs
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

    distributed = config['distributed']

    if distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(main_dist, args=(world_size, config), nprocs=world_size)
    else:
        main_single()