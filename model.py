import torchvision
import torch
import torch.nn as nn


def get_resnet(model='resnet18'): 
    match model: 
        case 'resnet18':  
            model = torchvision.models.resnet18()
        case 'resnet34':
            model = torchvision.models.resnet34()
        case 'resnet50':
            model = torchvision.models.resnet50()
        case 'resnet101':
            model = torchvision.models.resnet101()
    
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        nn.Linear(in_features, 10),
    )

    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    return model


if __name__ == "__main__":
    model = get_resnet()
    
    print(model)
