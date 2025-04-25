from src.losses import *
from src.network import Network, MLP
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
import yaml, sys, random, numpy as np
from yaml.loader import SafeLoader
from src.data import *
import math
import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 


def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
    
    return config_data

def progress(current, total, **kwargs):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    data_ = ""
    for meter, data in kwargs.items():
        data_ += f"{meter}: {round(data,2)}|"
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}|{data_}",end='\r')
    if (current == total):
        print()

def make_tsne_for_dataset(model, loader, device, algo, return_logs = False, tsne_name = None):
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            if algo == 'simsiam':
                feats, _, _ = model(x)
            else:
                feats, _ = model(x)

            all_features.append(feats)
            all_labels.append(y)

            if return_logs:
                progress(idx+1,loader_len)

    features = torch.vstack(all_features).detach().cpu().numpy()
    labels = torch.hstack(all_labels).detach().cpu().numpy()

    make_tsne_plot(features, labels, name = tsne_name)

def evaluate(model, mlp, loader, device, return_logs=False, algo=None):
    model.eval()
    mlp.eval()
    correct = 0;samples =0
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            if algo == 'simsiam':
                feats, _, _ = model(x)
            else:
                feats, _ = model(x)
            scores = mlp(feats)

            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)

            correct += (predictions == y).sum()
            samples += predictions.size(0)
        
            if return_logs:
                progress(idx+1,loader_len)
                # print('batches done : ',idx,end='\r')
        accuracy = round(float(correct / samples), 3)
    return accuracy 

def train_mlp(
    model, mlp, train_loader, test_loader, 
    lossfunction, mlp_optimizer, n_epochs, eval_every,
    device_id, eval_id, return_logs=False, algo=None, mlp_schedular=None):

    tval = {'trainacc':[],"trainloss":[]}
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    mlp = mlp.to(device)
    for epochs in range(n_epochs):
        model.eval()
        mlp.train()
        curacc = 0
        cur_mlp_loss = 0
        len_train = len(train_loader)
        for idx , (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                if algo == "simsiam":
                    feats, _, _ = model(data)
                else:
                    feats, proj_feat = model(data)
            scores = mlp(feats.detach())      
            
            loss_sup = lossfunction(scores, target)

            mlp_optimizer.zero_grad()
            loss_sup.backward()
            mlp_optimizer.step()

            cur_mlp_loss += loss_sup.item() / (len_train)
            scores = F.softmax(scores,dim = 1)
            _,predicted = torch.max(scores,dim = 1)
            correct = (predicted == target).sum()
            samples = scores.shape[0]
            curacc += correct / (samples * len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_sup=loss_sup.item(), GPU = device_id)
        
        if mlp_schedular is not None:
            mlp_schedular.step()
        
        if epochs % eval_every == 0 and device_id == eval_id:
            cur_test_acc = evaluate(model, mlp, test_loader, device, return_logs, algo=algo)
            print(f"[GPU{device_id}] Test Accuracy at epoch: {epochs}: {cur_test_acc}")
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_mlp_loss))
        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss_sup: {cur_mlp_loss:.3f}")
    
    if device_id == eval_id:
        final_test_acc = evaluate(model, mlp, test_loader, device, return_logs, algo=algo)
        print(f"[GPU{device_id}] Final Test Accuracy: {final_test_acc}")

    return mlp, tval

def train_supcon(
        train_algo, model, mlp, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, 
        optimizer, mlp_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    

    print(f"### {train_algo} Training begins")
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, target) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)
            if train_algo == "supcon":
                target = target.to(device)
            
            feats, proj_feat = model(data)
            feats_cap, proj_feat_cap = model(data_cap)
            
            if train_algo == "supcon":
                x_full = torch.cat([proj_feat,proj_feat_cap], dim = 0)
                target_full = torch.cat([target, target])
                loss_con = lossfunction(x_full, target_full)
            else:
                loss_con = lossfunction(proj_feat, proj_feat_cap)
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
            
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    print("### TSNE starts")
    make_tsne_for_dataset(model, test_loader, device_id, train_algo, return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")
    train_mlp(
        model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs)

    return model

def train_simsiam(
        model, mlp, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, 
        optimizer, mlp_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    

    print(f"### simsiam Training begins")
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, target) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)
            
            _, feats, pred_feat = model(data) # z, p
            _, feats_cap, pred_feat_cap = model(data_cap)
            
            loss_con = 0.5 * (lossfunction(pred_feat, feats_cap.detach()) + lossfunction(pred_feat_cap, feats.detach()))
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
            
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    print("### TSNE starts")
    make_tsne_for_dataset(model, test_loader, device_id, 'simsiam', return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")

    train_mlp(
        model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs, algo='simsiam')

    return model

def train_byol(
        online_model, target_model, online_pred_model, mlp, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, 
        optimizer, mlp_optimizer, opt_lr_schedular, ema_beta, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    

    print(f"### byol Training begins")
    device = torch.device(f"cuda:{device_id}")
    ema = EMA(ema_beta, n_epochs)
    online_model = online_model.to(device)
    target_model = target_model.to(device)
    online_pred_model = online_pred_model.to(device)

    for epochs in range(n_epochs):
        online_model.train()
        online_pred_model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, target) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)

            data_all = torch.cat([data, data_cap], dim = 0)

            _, online_proj = online_model(data_all) # y, z
            online_pred = online_pred_model(online_proj) # q
            with torch.no_grad():
                _, target_proj = target_model(data_all) # y, z

            online_pred_feat, online_pred_feat_cap = online_pred.chunk(2, dim = 0)
            target_proj_feat, target_proj_feat_cap = target_proj.chunk(2, dim = 0)
            
            loss_con = lossfunction(online_pred_feat, target_proj_feat_cap.detach()) + \
                        lossfunction(online_pred_feat_cap, target_proj_feat.detach())
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
        target_model = ema(online_model, target_model, epochs)
            
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    print("### TSNE starts")
    make_tsne_for_dataset(online_model, test_loader, device_id, 'byol', return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")

    train_mlp(
        online_model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs, algo='byol')

    return online_model

def train_triplet(
        model, mlp, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, 
        optimizer, mlp_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    
    print(f"### Triplet Training begins")

    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (a, a_t, p, p_t, n, n_t) in enumerate(train_loader):
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)
            
            af, apf = model(a)
            pf, ppf = model(p)
            nf, npf = model(n)
            
            loss_con = lossfunction(
                z_a = apf, z_p = ppf, z_n=npf)
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
              
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    print("### TSNE starts")
    make_tsne_for_dataset(model, test_loader, device_id, 'triplet', return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")

    train_mlp(
        model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs)

    return model

def train_barlow_twins(
        model, mlp, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, 
        optimizer, mlp_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    
    print(f"### Barlow Twins Training begins")

    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, target) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)
            
            feats, proj_feat = model(data)
            feats_cap, proj_feat_cap = model(data_cap)

            loss_con = lossfunction(proj_feat, proj_feat_cap)
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
              
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    print("### TSNE starts")
    make_tsne_for_dataset(model, test_loader, device_id, 'barlow_twins', return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")

    train_mlp(
        model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs)

    return model

def train_DARe(
        model, mlp, vae_linear, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, 
        optimizer, mlp_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    
    print(f"### DARe Training begins")

    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    vae_linear = vae_linear.to(device)
    for epochs in range(n_epochs):
        model.train()
        vae_linear.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, target) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)
            
            feats, proj_feat = model(data)
            feats_cap, proj_feat_cap = model(data_cap)

            mu, log_var = vae_linear(feats)
            mu_cap, log_var_cap = vae_linear(feats_cap)

            loss_con = lossfunction(proj_feat, proj_feat_cap, mu, mu_cap, log_var, log_var_cap)
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
              
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    print("### TSNE starts")
    make_tsne_for_dataset(model, test_loader, device_id, 'DARe', return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")

    train_mlp(
        model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs)

    return model

def loss_function(loss_type = 'supcon', **kwargs):
    print(f"loss function: {loss_type}")
    loss_mlp = nn.CrossEntropyLoss()
    if loss_type == "simclr":
        return SimCLR(**kwargs), loss_mlp
    elif loss_type == 'supcon':
        return SupConLoss(**kwargs), loss_mlp
    elif loss_type == "triplet":
        return TripletMarginLoss(**kwargs), loss_mlp
    elif loss_type == "simsiam":
        return SimSiamLoss(), loss_mlp
    elif loss_type == 'byol':
        return BYOLLoss(), loss_mlp
    elif loss_type == "barlow_twins":
        return BarlowTwinLoss(**kwargs), loss_mlp
    elif loss_type == "dare":
        return DAReLoss(**kwargs), loss_mlp
    else:
        print("{loss_type} Loss is Not Supported")
        return None 
    
def model_optimizer(model, opt_name, model2 = None, **opt_params):
    print(f"using optimizer: {opt_name}")

    if model2 is None:
        params = model.parameters()
    else:
        params = list(model.parameters()) + list(model2.parameters())

    if opt_name == "SGD":
        return optim.SGD(params, **opt_params)
    elif opt_name == "ADAM":
        return optim.Adam(params, **opt_params)
    elif opt_name == "AdamW":
        return optim.AdamW(params, **opt_params)
    else:
        print("{opt_name} not available")
        return None

def load_dataset(dataset_name, **kwargs):
    if dataset_name == "cifar10":
        return Cifar10DataLoader(**kwargs)
    if dataset_name == 'cifar100':
        return Cifar100DataLoader(**kwargs)
    else:
        print(f"{dataset_name} is not supported")
        return None

class EMA():
    def __init__(self, tau, K):
        self.tau_base = tau
        self.tau = tau 
        self.K = K

    def __call__(self, online, target, k):
        for online_wt, target_wt in zip(online.parameters(), target.parameters()):
            target_wt.data = self.tau * online_wt.data + (1 - self.tau) * target_wt.data
        self.tau = 1 - (1 - self.tau_base) * (math.cos(math.pi * k / self.K) + 1) / 2
        return copy.deepcopy(target) 

def make_tsne_plot(X, y, name):
    tsne = TSNE(n_components=2, random_state=0)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10')  # Color by labels
    plt.title("t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(label="Labels")
    plt.savefig(f"plots/{name}")
    plt.close()