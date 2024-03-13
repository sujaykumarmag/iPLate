import torch
import torchvision
import torch.nn as nn



class CNN(nn.Module):

    def __init__(self,num_classes):
        super(CNN,self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=5,stride=2)
        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=12,kernel_size=5)
        

        self.lin1 = nn.Linear(in_features=23232,out_features=250)
        self.lin2 = nn.Linear(in_features=250,out_features=num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.cnn1(x)
        out = self.maxpool(out)
        out = self.cnn2(out)
        out = self.maxpool(out)

        out = out.view(x.shape[0],-1)

        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.relu(out)

        return out




