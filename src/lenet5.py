import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)

        self.avgpool = nn.AvgPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        

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

if __name__ == "__main__":
    model = LeNet5()
    input = torch.randn(1,1,28,28)
    print(model(input))
    