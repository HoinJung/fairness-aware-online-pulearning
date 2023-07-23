import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
class Layer(nn.Module):
    def __init__(self, input_size,hidden_feature):
        super(Layer, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_feature)
        self.fc2 = nn.Linear(hidden_feature, 1)

        self.fc1.weight.data.fill_(0)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        
        
    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        f = self.fc2(h)
        
        return f, h


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class O_MLP(nn.Module):
    def __init__(self, input_size, hidden_feature,max_num_hidden_layers, hidden_nodes,device, num_class=1):
        super(O_MLP, self).__init__()
        
        b=0.99
        s=0.2
        
        self.device = device

        self.hidden_feature = hidden_feature
        self.hidden_nodes = hidden_nodes
        self.max_num_hidden_layers = max_num_hidden_layers

        self.hidden_layers = []
        self.output_layers = []
        self.hidden_layers.append(
        nn.Linear(input_size, self.hidden_nodes[0]))
        
        for i in range(max_num_hidden_layers - 1):
            self.hidden_layers.append(
                nn.Linear(self.hidden_nodes[i], self.hidden_nodes[i+1]))


        for i in range(max_num_hidden_layers):
            self.output_layers.append(
                nn.Linear(self.hidden_nodes[i], num_class))
        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)
        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers).fill_(1 / (self.max_num_hidden_layers + 1)),
                               requires_grad=False).to(self.device)
        self.b = Parameter(torch.tensor(
            b), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(
            s), requires_grad=False).to(self.device)
        self.zero_initial()
        
    def zero_initial(self):
        for i in range(self.max_num_hidden_layers):
            nn.init.kaiming_uniform_(self.output_layers[i].weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.hidden_layers[i].weight, mode='fan_in', nonlinearity='relu')

    def zero_gradient(self):
        for i in range(self.max_num_hidden_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)
        
    def forward(self, x):
        
        hidden_connections = []
        
        x = self.hidden_layers[0](x)
        
        hidden_connections.append(x)

        for i in range(1, self.max_num_hidden_layers):
            hidden_connections.append(
                self.hidden_layers[i](hidden_connections[i - 1]))
        
        output_class = []
        for i in range(self.max_num_hidden_layers):
            output_class.append(self.output_layers[i](hidden_connections[i]))

        pred_per_layer = torch.stack(output_class)
        
        
        return pred_per_layer

class Linear(torch.nn.Module):
     def __init__(self, input_size, output_dim=1,mode = 'online'):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_dim,bias=True)
        if mode=='online':
            self.linear.weight.data.fill_(0)
            self.linear.bias.data.fill_(0)
        else :
            nn.init.xavier_normal(self.linear.weight)
            
     def forward(self, x):
        outputs = self.linear(x)
        return outputs


        