import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

    
class MLP(nn.Module):
    def __init__(self, input_size,hidden_dim):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class Linear(torch.nn.Module):
    def __init__(self, input_size, output_dim=1, mode='offline'):
        super(Linear, self).__init__()
        
        self.linear = torch.nn.Linear(input_size, output_dim)
        
        # Handle the initialization for online mode
        if mode == 'online':
            self.linear.weight.data.fill_(0)
            self.linear.bias.data.fill_(0)
        else:
            nn.init.kaiming_normal_(self.linear.weight)
    
    def forward(self, x):
        x = self.linear(x)
        
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTM, self).__init__()
        
        n_layers = 2
        bidirectional = True
        dropout = 0.5
        output_dim = 1
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=0.5)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(x)
        # print(embedded.shape)
        # LSTM
        embedded = embedded.unsqueeze(0)  # This makes the shape [1, batch_size, embedding_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        

        # If bidirectional, concatenate the final forward (hidden[-2,:,:]) 
        # and backward (hidden[-1,:,:]) hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Apply dropout
        hidden = self.dropout(hidden)
            
        # Return the result of the fully connected layer
        return self.fc(hidden)

class O_MLP(nn.Module):
    def __init__(self, input_size, max_num_hidden_layers, hidden_nodes,device, num_class=1):
        super(O_MLP, self).__init__()
        
        
        b=0.99
        s=0.2
        
        self.device = device

        # self.hidden_feature = hidden_feature
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
        self.initial()
        
    def initial(self):
        for i in range(self.max_num_hidden_layers):
            nn.init.kaiming_normal_(self.output_layers[i].weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.hidden_layers[i].weight, mode='fan_in', nonlinearity='relu')

    def zero_gradient(self):
        for i in range(self.max_num_hidden_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)
    def zero_grad(self):
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

class O_LSTM(nn.Module):
    def __init__(self, input_size, max_num_hidden_layers, hidden_nodes, device, num_class=1):
        super(O_LSTM, self).__init__()

        b = 0.99
        s = 0.2

        self.device = device
        self.hidden_nodes = hidden_nodes
        self.max_num_hidden_layers = max_num_hidden_layers

        self.lstm_layers = []
        self.output_layers = []

        # Initialize LSTM layers and Output layers
        for i in range(max_num_hidden_layers):
            if i == 0:
                self.lstm_layers.append(nn.LSTMCell(input_size, self.hidden_nodes[i]))
            else:
                self.lstm_layers.append(nn.LSTMCell(self.hidden_nodes[i-1], self.hidden_nodes[i]))
            self.output_layers.append(nn.Linear(self.hidden_nodes[i], num_class))
        
        self.lstm_layers = nn.ModuleList(self.lstm_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)

        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers).fill_(1 / (self.max_num_hidden_layers + 1)),
                               requires_grad=False).to(self.device)
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)

        self.initial()

    def initial(self):
        for i in range(self.max_num_hidden_layers):
            nn.init.kaiming_normal_(self.output_layers[i].weight, mode='fan_in', nonlinearity='relu')
            # LSTM initialization can be more involved than this, but for simplicity:
            nn.init.kaiming_normal_(self.lstm_layers[i].weight_ih, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.lstm_layers[i].weight_hh, mode='fan_in', nonlinearity='relu')

    def zero_gradient(self):
        for i in range(self.max_num_hidden_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.lstm_layers[i].weight_ih.grad.data.fill_(0)
            self.lstm_layers[i].weight_hh.grad.data.fill_(0)
            self.lstm_layers[i].bias_ih.grad.data.fill_(0)
            self.lstm_layers[i].bias_hh.grad.data.fill_(0)

    def zero_grad(self):
        self.zero_gradient()
    def forward(self, x):
        batch_size = x.size(0)

        # Initialize the initial hidden and cell states
        h_t = [torch.zeros(batch_size, hidden_size).to(self.device) for hidden_size in self.hidden_nodes]
        c_t = [torch.zeros(batch_size, hidden_size).to(self.device) for hidden_size in self.hidden_nodes]

        hidden_connections = []
        
        # First LSTM layer
        h_t[0], c_t[0] = self.lstm_layers[0](x, (h_t[0], c_t[0]))
        hidden_connections.append(h_t[0])
        
        # Subsequent LSTM layers
        for i in range(1, self.max_num_hidden_layers):
            h_t[i], c_t[i] = self.lstm_layers[i](hidden_connections[i - 1], (h_t[i], c_t[i]))
            hidden_connections.append(h_t[i])
            
        output_class = []
        for i in range(self.max_num_hidden_layers):
            output_class.append(self.output_layers[i](hidden_connections[i]))

        pred_per_layer = torch.stack(output_class)
        
        return pred_per_layer
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class O_Transformer(nn.Module):
    def __init__(self, input_size, hidden_dim, max_num_transformer_layers, nhead, device, num_class=1):
        super(O_Transformer, self).__init__()

        self.device = device
        self.max_num_transformer_layers = max_num_transformer_layers
        self.nhead = nhead

        b = 0.99
        s = 0.2

        self.initial_linear = nn.Linear(input_size, hidden_dim, bias=True)
        nn.init.kaiming_normal_(self.initial_linear.weight, mode='fan_in', nonlinearity='relu')
        # self.initial_linear.weight.data.fill_(0.00)
        # self.initial_linear.bias.data.fill_(0.00)
        
        self.transformer_layers = []
        self.output_layers = []

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=self.nhead)
        
        for i in range(max_num_transformer_layers):
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # Using single layer for each transformer
            self.transformer_layers.append(transformer)
            self.output_layers.append(nn.Linear(hidden_dim, num_class))
        
        self.transformer_layers = nn.ModuleList(self.transformer_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)
        
        self.alpha = Parameter(torch.Tensor(self.max_num_transformer_layers).fill_(1 / (self.max_num_transformer_layers + 1)),
                               requires_grad=False).to(self.device)
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.initial()

    def initial(self):
        for i in range(self.max_num_transformer_layers):
            nn.init.kaiming_normal_(self.output_layers[i].weight, mode='fan_in', nonlinearity='relu')
            # Note: Transformers have their own initialization. 

    def zero_gradient(self):
        for i in range(self.max_num_transformer_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            for param in self.transformer_layers[i].parameters():
                param.grad.data.fill_(0)

    def forward(self, x):
        x = self.initial_linear(x)
        x = x.unsqueeze(0)  # Adding sequence length of 1
        x = x.permute(1, 0, 2)  # Adjusting dimensions for Transformer
        
        
        transformer_outputs = []
        for i in range(self.max_num_transformer_layers):
            transformer_output = self.transformer_layers[i](x)
            transformer_output = transformer_output.permute(1, 0, 2).squeeze()  # Readjusting dimensions for Linear layer
            transformer_outputs.append(transformer_output)
        
        output_class = []
        for i, transformer_output in enumerate(transformer_outputs):
            output_class.append(self.output_layers[i](transformer_output))

        pred_per_layer = torch.stack(output_class)

        return pred_per_layer

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout,dim_feedforward):
        super(Transformer, self).__init__()
        
        # Transformer Block
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout,dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Linear Layer
        self.fc = nn.Linear(d_model, 1)   # Produces a single toxicity score
        
    def forward(self, src):
        # Transformer requires input shape (S, N, E)
        src = src.unsqueeze(1)
        src = src.permute(1, 0, 2)
        
        # Get the encoder's output
        output = self.transformer_encoder(src)
        output = output.squeeze(0)
        # Taking the last token's representation to predict toxicity score
        # Alternatively, you can consider other strategies like average pooling across tokens
        
        output = self.fc(output)
        # output = output.unsqueeze(1)
        # print(output.shape)
        
        return output