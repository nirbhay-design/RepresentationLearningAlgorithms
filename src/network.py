import torch 
import torchvision 
import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class Network(nn.Module):
    def __init__(self, model_name = 'resnet18', pretrained = False, proj_dim = 128):
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
            module_key = model._modules.get(key, nn.Identity())
            self.feat_extractor.add_module(key, module_key)

        self.classifier_infeatures = model._modules.get(module_keys[-1], nn.Identity()).in_features
        self.proj = nn.Linear(self.classifier_infeatures, proj_dim)

    def forward(self, x):
        features = self.feat_extractor(x).flatten(1)
        proj_features = self.proj(features)
        return features, proj_features

if __name__ == "__main__":
    network = Network(model_name = 'resnet50', pretrained=False)
    mlp = MLP(network.classifier_infeatures, num_classes=10)
    x = torch.rand(2,3,224,224)
    feat, proj_feat = network(x)
    print(feat.shape, proj_feat.shape)
    score = mlp(feat)
    print(score.shape)