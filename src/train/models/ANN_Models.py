import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import numpy as np


class ANNMalware_Model1(nn.Module):
    def __init__(self, image_dim=32, num_of_classes=20):
        super().__init__()

        self.image_dim = image_dim
        self.num_of_classes = num_of_classes

        self.linear1_in_features = int(1 * self.image_dim)
        # reduce the neurons by 20% i.e. take 80% in_features
        self.linear1_out_features = int(self.linear1_in_features * 0.80)
        # reduce the neurons by 40%
        self.linear2_out_features = int(self.linear1_out_features * 0.60)

        self.classifier = nn.Sequential(
            nn.Linear(self.linear1_in_features, self.linear1_out_features),
            nn.BatchNorm1d(self.linear1_out_features),  # BatchNorm layer
            nn.ReLU(inplace=True),
            nn.Linear(self.linear1_out_features, self.linear2_out_features),
            nn.BatchNorm1d(self.linear2_out_features),  # BatchNorm layer
            nn.ReLU(inplace=True),
            nn.Linear(self.linear2_out_features, self.num_of_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class ANNMalware_Model2(nn.Module):
    def __init__(self, image_dim=32, num_of_classes=20):
        super().__init__()

        self.image_dim = image_dim
        self.num_of_classes = num_of_classes

        self.linear1_in_features = int(self.image_dim * self.image_dim)
        # reduce the neurons by 20% i.e. take 80% in_features
        self.linear1_out_features = int(self.linear1_in_features * 0.80)
        # reduce the neurons by 40%
        self.linear2_out_features = int(self.linear1_out_features * 0.60)
        # reduce the neurons by 20%
        self.linear3_out_features = int(self.linear1_out_features * 0.40)
        # reduce the neurons by 20%
        self.linear4_out_features = int(self.linear1_out_features * 0.20)

        self.classifier = nn.Sequential(
            nn.Linear(self.linear1_in_features, self.linear1_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear1_out_features, self.linear2_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear2_out_features, self.linear3_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear3_out_features, self.linear4_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear4_out_features, self.num_of_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    

class MalConv(nn.Module):
    # trained to minimize cross-entropy loss
    # criterion = nn.CrossEntropyLoss()
    def __init__(self, num_feats = 257, out_size=2, channels=128, window_size=512, embd_size=8):
        super(MalConv, self).__init__()
        self.embd = nn.Embedding(num_feats, embd_size, padding_idx=0)
        
        self.window_size = window_size
    
        self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        
        self.pooling = nn.AdaptiveMaxPool1d(1)
        
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
    
    def forward(self, x):
        
        x = self.embd(x.long())
        x = torch.transpose(x,-1,-2)
        
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))
        
        x = cnn_value * gating_weight
        
        x = self.pooling(x)
        
        #Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        
        return x