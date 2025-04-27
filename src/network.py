import torch 
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, num_classes, mlp_type="linear"):
        super().__init__()
        if mlp_type == "linear":
            print("===> using linear mlp")
            self.mlp = nn.Sequential(
                nn.Linear(in_features, num_classes)
            )
        else:
            print("===> using hiddin mlp")
            self.mlp = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.mlp(x)

class BYOL_mlp(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.mlp(x)

class Network(nn.Module):
    def __init__(self, model_name = 'resnet18', pretrained = False, proj_dim = 128, algo_type="supcon", pred_dim = 512, byol_hidden = 4096, barlow_hidden = 8192):
        super().__init__()
        if model_name == 'resnet50':
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif model_name  == 'resnet18':
            model = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        else:
            print(f"{model_name} model type not supported")
            model = None

        module_keys = list(model._modules.keys())
        self.feat_extractor = nn.Sequential()
        for key in module_keys[:-1]:
            if key == "maxpool": # don't add maxpool layer
                continue
            module_key = model._modules.get(key, nn.Identity())
            self.feat_extractor.add_module(key, module_key)

        if not pretrained:
            in_feat = self.feat_extractor.conv1.in_channels
            out_feat = self.feat_extractor.conv1.out_channels
            self.feat_extractor.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, bias=False)

        self.classifier_infeatures = model._modules.get(module_keys[-1], nn.Identity()).in_features
        
        if algo_type == "simsiam":
            prev_dim = self.classifier_infeatures
            self.proj = nn.Sequential(
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim)
            )
            self.pred = nn.Sequential(
                nn.Linear(prev_dim, pred_dim, bias=False),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(),
                nn.Linear(pred_dim, prev_dim)
            )
        elif algo_type == 'byol':
            self.proj = BYOL_mlp(in_features = self.classifier_infeatures, hidden_dim = byol_hidden, out_features = proj_dim)
        elif algo_type == "barlow_twins":
            self.proj = nn.Sequential(
                nn.Linear(self.classifier_infeatures, barlow_hidden, bias=False),
                nn.BatchNorm1d(barlow_hidden, bias=False),
                nn.ReLU(),
                nn.Linear(barlow_hidden, barlow_hidden, bias=False),
                nn.BatchNorm1d(barlow_hidden),
                nn.ReLU(),
                nn.Linear(barlow_hidden, proj_dim)
            )
        else:
            self.proj = nn.Linear(self.classifier_infeatures, proj_dim)

        self.algo_type = algo_type

    def forward(self, x):
        features = self.feat_extractor(x).flatten(1)
        proj_features = self.proj(features)

        if self.algo_type == 'simsiam':
            pred_features = self.pred(proj_features)
            return features, proj_features, pred_features # features, proj - z, pred - p
        return features, proj_features # 2048/512, 128 proj

class VAE_linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.ip = input_dim
        self.out = output_dim
        self.linear_mu = nn.Linear(self.ip, self.out)
        self.linear_var = nn.Linear(self.ip, self.out)

    def forward(self, x):
        mu =  self.linear_mu(x)
        log_var = self.linear_var(x)
        return mu, log_var

if __name__ == "__main__":
    network = Network(model_name = 'resnet50', pretrained=False, algo_type='byol', byol_hidden = 4096, proj_dim = 256)
    mlp = MLP(network.classifier_infeatures, num_classes=10, mlp_type='hidden')
    x = torch.rand(2,3,224,224)
    feat, proj_feat = network(x)
    print(feat.shape, proj_feat.shape)
    score = mlp(feat)
    print(score.shape)

    print(network)

    print(network.proj.mlp[-1].out_features)

    # contrastive loss on proj_feat, representations are feat, MLP on feat 