
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features
    activation = nn.Sigmoid(), nn.Softmax"""
    def __init__(self, embed_dim=384, num_cls=13,  activation=None):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(embed_dim, num_cls)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
      
        self.activation = activation
        self.arch = 'linear_classifier'


    def forward(self, x):
        # linear layer
        x = self.linear(x)
        #activation
        if self.activation: 
            x = self.activation(x)
        return x