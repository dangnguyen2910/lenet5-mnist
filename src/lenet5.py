import torchvision
import torch
import torch.nn as nn
import math


# def get_resnet(model='resnet18'): 
#     match model: 
#         case 'resnet18':  
#             model = torchvision.models.resnet18()
#         case 'resnet34':
#             model = torchvision.models.resnet34()
#         case 'resnet50':
#             model = torchvision.models.resnet50()
#         case 'resnet101':
#             model = torchvision.models.resnet101()
    
#     in_features = model.fc.in_features
#     model.fc = torch.nn.Sequential(
#         nn.Linear(in_features, 10),
#     )

#     model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

#     return model


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        # h,w = self.calculate_conv_output_dim((28,28), padding=2, stride=1, kernel_size=5)

        self.avgpool = nn.AvgPool2d(2, stride=2)
        # h,w = self.calculate_pool_output_dim((h,w), padding=0, stride=2, kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        # h,w = self.calculate_conv_output_dim((h,w), padding=0, stride=1, kernel_size=5)
        
        # h,w = self.calculate_pool_output_dim((h,w), padding=0, stride=2, kernel_size=2)

        self.linear1 = nn.Linear(400, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    
    def forward(self, x): 
        x = self.relu(self.avgpool(self.conv1(x)))
        x = self.relu(self.avgpool(self.conv2(x)))
        x = self.flatten(x)
        x = self.linear3(self.linear2(self.linear1(x)))
        return x

    

    def calculate_conv_output_dim(self, dim, padding,  kernel_size, stride, dilation=0,):
        height, width = dim
        height_out = math.floor(((height + 2*padding - dilation * (kernel_size-1)-1)/stride) + 1)
        width_out = math.floor(((width + 2*padding - dilation * (kernel_size-1)-1)/stride) + 1)
        return height_out, width_out
    

    def calculate_pool_output_dim(self, dim, padding, kernel_size, stride): 
        height, width = dim
        height_out = math.floor(((height + 2*padding - kernel_size)/stride) + 1)
        width_out = math.floor(((width + 2*padding - kernel_size)/stride) + 1)
        return height_out, width_out


if __name__ == "__main__":
    model = LeNet5()
    input = torch.randn(1,1,28,28)
    print(model(input))
    
    # print(model)
