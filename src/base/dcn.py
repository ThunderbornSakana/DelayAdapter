import torch
import torch.nn as nn


class CrossNet(nn.Module):
    def __init__(self, in_dim, layer_num=2, parameterization='vector'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # Weight in DCN.  (in_dim, 1)
            self.kernels = nn.Parameter(torch.Tensor(layer_num, in_dim, 1))
        elif self.parameterization == 'matrix':
            # Weight matrix in DCN-M.  (in_dim, in_dim)
            self.kernels = nn.Parameter(torch.Tensor(layer_num, in_dim, in_dim))
        else:
            raise ValueError("Parameterization should be vector or matrix.")
        self.bias = nn.Parameter(torch.Tensor(layer_num, in_dim, 1))
        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

    def forward(self, x):
        x_0 = x.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)
                # W * xi (bs, in_dim, 1) ^
                dot_ = xl_w + self.bias[i] # W * xi + b
                x_l = x_0 * dot_ + x_l # x0 · (W * xi + b) + xl (Hadamard)
            else:
                raise ValueError("Parameterization should be vector or matrix.")
        return torch.squeeze(x_l, dim=2)
    

"""
  @param cross_num - The number of cross layers.
  @param cross_param - How to parameterize cross net ("vector" or "matrix").
  @param hidden_units - Number of units in each layer of DNN.
"""
class DCN(nn.Module):
    def __init__(self, num_num, embed_size, cat_dims, output_size, cross_num=2,
                 cross_param='vector', hidden_units=[128, 128], dropout=0, predict=False):
        super(DCN, self).__init__()
        # Initialize the embedding table for categorical features.
        self.embedding = nn.ModuleList([nn.Embedding(c, embed_size) for c 
                                        in cat_dims])
        input_size = num_num + len(cat_dims) * embed_size
        self.hidden_units = hidden_units
        self.cross_num = cross_num
        # Initialize DNN layers.
        if len(hidden_units) > 0:
            layers = []
            last_hidden_size = input_size
            for hidden_size in hidden_units:
                layers.append(nn.Linear(last_hidden_size, hidden_size))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                last_hidden_size = hidden_size
            self.dnn = nn.Sequential(*layers)
        # Get output layer input dimension.
        if len(hidden_units) > 0 and cross_num > 0:
            pre_out_dim = input_size + hidden_units[-1]
        elif len(hidden_units) > 0:
            pre_out_dim = hidden_units[-1]
        elif cross_num > 0:
            pre_out_dim = input_size
        # Initialize cross net.
        if cross_num > 0:
            self.crossNet = CrossNet(input_size, cross_num, cross_param)
        self.out = nn.Linear(pre_out_dim, output_size)
        self.predict = predict

    def forward(self, cat, num=None):
        embedded = [embed(cat[:, i]) for i, embed in enumerate(self.embedding)]
        if num is not None:
            x = torch.cat([num] + embedded, dim=1)
        else:
            x = torch.cat(embedded, dim=1)
        pred = None
        # Deep & Cross.
        if len(self.hidden_units) > 0 and self.cross_num > 0:
            deep_out = self.dnn(x)
            cross_out = self.crossNet(x)
            pred = self.out(torch.cat((cross_out, deep_out), dim=-1))
        # Only Deep.
        elif len(self.hidden_units) > 0:
            deep_out = self.dnn(x)
            pred = self.out(deep_out)
        # Only Cross.
        elif self.cross_num > 0:
            cross_out = self.crossNet(x)
            pred = self.out(cross_out)
        if self.predict:
            pred = torch.sigmoid(pred)
        return pred
