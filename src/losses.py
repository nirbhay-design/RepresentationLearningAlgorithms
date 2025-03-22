import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

class SupConLossBruteForce(nn.Module):
    def __init__(self, 
                 sim = 'cosine', 
                 tau = 1.0):
        super().__init__()
        self.tau = tau
        self.sim = sim 

    def forward(self, features, labels):
        features = F.normalize(features, dim = -1, p = 2)
        loss = 0
        for idx, i in enumerate(features):
            total_sim = 0
            pos_samples_wt = []
            num_pos = 0
            for jdx, j in enumerate(features):
                if idx != jdx:
                    sim_ij = torch.exp(torch.sum(i*j) / self.tau)
                    total_sim += sim_ij 
                    if labels[idx] == labels[jdx]: # positive_sample
                        pos_samples_wt.append(sim_ij)
                        num_pos += 1

            log_softmax = -torch.log(torch.tensor(pos_samples_wt) / total_sim).sum() / (num_pos + 1e-5)
            loss += log_softmax
        return loss / features.shape[0]      

class SupConLoss(nn.Module):
    def __init__(self,  
                 sim = 'cosine', 
                 tau = 1.0):
        super().__init__()
        self.tau = tau
        self.sim = sim 

    def forward(self, features, labels):
        B, _ = features.shape
        # calculate pair wise similarity 
        sim_mat = self.calculate_sim_matrix(features)
        # division by temperature
        sim_mat = F.log_softmax(sim_mat / self.tau, dim = -1) 
        sim_mat = sim_mat.clone().fill_diagonal_(torch.tensor(0.0))
        # calculating pair wise equal labels for pos pairs
        labels = labels.unsqueeze(1)
        labels_mask = (labels == labels.T).type(torch.float32)
        labels_mask.fill_diagonal_(torch.tensor(0.0))
        # calculating num of positive pairs for each sample
        num_pos = torch.sum(labels_mask, dim = -1)
        # masking out the negative pair log_softmax value
        pos_sim_mat = sim_mat * labels_mask 
        # summing log_softmax value over all positive pairs
        pos_pair_sum = torch.sum(pos_sim_mat, dim = -1)
        # averaging out the log_softmax value, epsilon = 1e-5 is to avoid division by zero
        pos_pairs_avg = torch.div(pos_pair_sum, num_pos + 1e-5)
        # final loss over all features in batch
        loss = -pos_pairs_avg.sum() / B
        return loss

    def calculate_sim_matrix(self, features):
        sim_mat = None
        if self.sim == "mse":
            sim_mat = -torch.cdist(features, features)
        elif self.sim == "cosine":
            features = F.normalize(features, dim = -1, p = 2)
            sim_mat = F.cosine_similarity(features[None, :, :], features[:, None, :], dim = -1)
        else: # bhattacharya coefficient
            features = F.normalize(features, dim = -1, p = 2)
            features = F.softmax(features, dim = -1)
            sqrt_feats = torch.sqrt(features) # sqrt of prob dist 
            sim_mat = sqrt_feats @ sqrt_feats.T
            
        # filling diagonal with -torch.inf as it will be cancel out while doing softmax
        sim_mat.fill_diagonal_(-torch.tensor(torch.inf))
        return sim_mat      
    
class SupConClsLoss(nn.Module): # combination of SupConLoss with CrossEntropyLoss
    def __init__(self, sim = 'cosine', tau = 1.0):
        super().__init__()
        self.tau = tau
        self.sim = sim 
        self.ce = nn.CrossEntropyLoss()
        self.supcon = SupConLoss(sim = sim, tau = tau)
    
    def forward(self, features, features_cap, scores, labels): # features: [B, 128], scores: [B, num_classes]
        x_full = torch.cat([features,features_cap], dim = 0)
        label_full = torch.cat([labels, labels])
        
        return self.supcon(x_full, label_full), self.ce(scores, labels)
    
class TripletMarginLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super().__init__()
        self.margin = margin 

    def forward(self, z_a, z_p, z_n):
        dist1 = torch.norm(z_a - z_p, dim = -1)
        dist2 = torch.norm(z_a - z_n, dim = -1)
        loss_added_margin = dist1 - dist2 + self.margin
        loss_added_margin[loss_added_margin < 0] = 0.0 
        loss_batch = torch.mean(loss_added_margin)

        return loss_batch 
    
class TripletMarginCELoss(nn.Module):
    def __init__(self, margin = 1.0):
        super().__init__()
        self.tml = TripletMarginLoss(margin = margin)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z_a, z_p, z_n, s_a, l_a):
        tml = self.tml(z_a, z_p, z_n)
        ce = self.ce(s_a, l_a)
        return tml, ce
    
class SimCLR(nn.Module):
    def __init__(self, sim = 'cosine', tau = 1.0):
        super().__init__()

        self.supcon = SupConLoss(sim, tau)

    def forward(self, x, x_cap):
        B = x.shape[0]
        device = x.device 

        fake_label = torch.arange(0, B, device = device)
        fake_labels = torch.cat([fake_label, fake_label])

        x_full = torch.cat([x,x_cap], dim = 0)

        return self.supcon(x_full, fake_labels)

class SimCLRClsLoss(nn.Module):
    def __init__(self, sim = 'cosine', tau = 1.0):
        super().__init__()
        self.tau = tau
        self.sim = sim 
        self.ce = nn.CrossEntropyLoss()
        self.simclr = SimCLR(sim = sim, tau = tau)
    
    def forward(self, features, features_cap, scores, labels):
        return self.simclr(features, features_cap), self.ce(scores, labels)

class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, z):
        p = F.normalize(p, dim = -1)
        z = F.normalize(z, dim = -1)

        return -(p * z).sum(dim = -1).mean()

class BYOLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, online, target):
        online = F.normalize(online, dim = -1)
        target = F.normalize(target, dim = -1)

        return -2 * (online * target).sum(dim = -1).mean()

class BarlowTwinLoss(nn.Module):
    def __init__(self, lambd = 0.1):
        super().__init__()
        self.lambd = lambd

    def forward(self, za, zb):
        N, D = za.shape

        za = (za - za.mean(0, keepdim=True)) / za.std(0, keepdim=True)
        zb = (zb - zb.mean(0, keepdim=True)) / zb.std(0, keepdim=True)

        C = torch.mm(za.T, zb) / N # DxD

        I = torch.eye(D, device=za.device)

        diff = (I - C).pow(2)

        diag_elem = torch.diag(diff)

        diff.fill_diagonal_(0.0)

        return diag_elem.sum() + self.lambd * diff.sum()

if __name__ == "__main__":
    scl = SupConClsLoss()
    tml = TripletMarginCELoss()
    sclr = SimCLRClsLoss()

    features = torch.rand(10,5)
    features_cap = torch.rand(10,5)
    features_p = torch.rand(10,5)
    features_n = torch.rand(10,5)
    scores_p = torch.rand(10,3)
    scores_n = torch.rand(10,3)
    scores = torch.rand(10,3)
    labels = torch.randint(low = 0, high = 2, size = (10,))

    print(scl(features, features_cap, scores, labels))
    print(sclr(features, features_cap, scores, labels))
    print(tml(features, features_p, features_n, scores, labels))

    # implement bhattacharya as measure of similarity

    bt = BarlowTwinLoss()

    za = torch.rand(2,3)
    zb = torch.rand(2,3)

    print(bt(za, zb))