import sys, random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.network import Network, MLP, BYOL_mlp, VAE_linear
from train_utils import yaml_loader, train_supcon, train_triplet, train_simsiam, \
                        train_byol, train_barlow_twins, train_DARe, train_DiAl, model_optimizer, \
                        loss_function, \
                        load_dataset

import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group
import os 
import argparse 

def get_args():
    parser = argparse.ArgumentParser(description="Training script")

    # basic experiment settings
    parser.add_argument("--config", type=str, default = "configs/nodel.c10.yaml", required=True, help="config file")
    parser.add_argument("--save_path", type=str, default="model.pth", required=True, help="path to save model")
    parser.add_argument("--gpu", type=int, default = 0, help="gpu_id")
    parser.add_argument("--model", type=str, default="resnet50", help="resnet18/resnet50")
    parser.add_argument("--verbose", action="store_true", help="verbose or not")
    parser.add_argument("--epochs", type=int, default = None, help="epochs for SSL pretraining")
    parser.add_argument("--epochs_lin", type=int, default = None, help="epochs for linear probing")
    parser.add_argument("--opt", type=str, default=None, help="SGD/ADAM/AdamW")
    parser.add_argument("--lr", type=float, default = None, help="lr for SSL")

    args = parser.parse_args()
    return args

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = "4084"
    init_process_group(backend = 'nccl', rank = rank, world_size = world_size)

def train_network(**kwargs):
    train_algo = kwargs['train_algo']
    kwargs.pop("train_algo")
    if train_algo == "supcon" or train_algo == "simclr":
        kwargs["train_algo"] = train_algo
        return train_supcon(**kwargs)
    elif train_algo == "triplet":
        return train_triplet(**kwargs)
    elif train_algo == "simsiam":
        return train_simsiam(**kwargs)
    elif train_algo == 'byol':
        return train_byol(**kwargs)
    elif train_algo == "barlow_twins":
        return train_barlow_twins(**kwargs)
    elif train_algo == "dare":
        return train_DARe(**kwargs)
    elif train_algo == "dial":
        return train_DiAl(**kwargs)
    return None 

def main_single():
    train_algo = config['train_algo']

    model = Network(**config['model_params'])
    mlp = MLP(model.classifier_infeatures, config['num_classes'], config['mlp_type'])

    pred_net = None 
    if train_algo == "byol":
        pred_net = BYOL_mlp(**config["byol_pred_params"])
    elif train_algo == "dare":
        pred_net = VAE_linear(input_dim = model.classifier_infeatures, output_dim = config['vae_linear_out'])

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

    tsne_name = "_".join(config["model_save_path"].split('/')[-1].split('.')[:-1]) + ".png"
    # tsne_name = "_".join(sys.argv[1].split('/')[-1].split('.')[:-1]) + f"_{config['model_params']['model_name']}.png"

    ## defining parameter configs for each training algorithm
    param_config = {"train_algo": train_algo, "model": model, "mlp": mlp, "train_loader": train_dl, "train_loader_mlp": train_dl_mlp,
        "test_loader": test_dl, "lossfunction": loss, "lossfunction_mlp": loss_mlp, "optimizer": optimizer, 
        "mlp_optimizer": mlp_optimizer, "opt_lr_schedular": opt_lr_schedular, "eval_every": eval_every, 
        "n_epochs": n_epochs, "n_epochs_mlp": n_epochs_mlp, "device_id": device, "eval_id": device, "tsne_name": tsne_name, "return_logs": return_logs}
    
    if train_algo == 'simclr' or train_algo == 'supcon':
        pass # no need to add anything
    elif train_algo == 'triplet' or train_algo == "barlow_twins" or train_algo == "simsiam":
        pass # no change for triplet margin loss 
    # elif train_algo == 'simsiam':
    #     mlp_opt_lr_schedular = optim.lr_scheduler.StepLR(mlp_optimizer, **config['mlp_schedular_params'])
    #     param_config["mlp_opt_lr_schedular"] = mlp_opt_lr_schedular
    elif train_algo == 'byol':
        target_net = Network(**config['model_params'])
        ema_tau = config['ema_tau']

        param_config.pop("model")
        param_config["online_model"] = model 
        param_config["target_model"] = target_net 
        param_config["online_pred_model"] = pred_net 
        param_config["ema_beta"] = ema_tau
    elif train_algo == "dare":
        param_config["vae_linear"] = pred_net 

    final_model = train_network(**param_config)

    torch.save(final_model.state_dict(), config["model_save_path"])
    print("Model weights saved")

if __name__ == "__main__":
    args = get_args()
    config = yaml_loader(args.config)

    config["config"] = args.config
    config['gpu_id'] = args.gpu
    config['model_params']['model_name'] = args.model
    config["return_logs"] = args.verbose
    config["model_save_path"] = os.path.join(config.get("model_save_path", "saved_models"), args.save_path)

    if args.opt:
        config["opt"] = args.opt
        if args.opt in ["ADAM", "AdamW"]:
            config["opt_params"].pop("momentum", -1)
            config["opt_params"].pop("nesterov", -1)
    if args.lr:
        config["opt_params"]["lr"] = args.lr 
    if args.epochs:
        config["n_epochs"] = args.epochs
        config["schedular_params"]["T_max"] = args.epochs
    if args.epochs_lin:
        config["n_epochs_mlp"] = args.epochs_lin
    
    # setting random seeds 
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