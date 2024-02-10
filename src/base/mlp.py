import torch
import torch.nn as nn


"""
@param num_num - The number of numeric features.
@param hidden_units - The size of the hidden layers.
@param output_size - The size of the output layer.
@param cat_dims - List of categorical value dimensions.
@param embed_size - Categorical value embedding dimension.
"""
class MLP(nn.Module):
    def __init__(self, num_num, hidden_units, embed_size, cat_dims,
                 output_size=0, dropout=0, predict=False):
        super(MLP, self).__init__()
        # Initialize the embedding table for categorical features.
        self.embedding = nn.ModuleList([nn.Embedding(c, embed_size) for c
                                        in cat_dims])
        input_size = num_num + len(cat_dims) * embed_size
        # Initialize hidden layers.
        layers = []
        for hidden_size in hidden_units:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        # Whether to add the final layer for output.
        if output_size:
            layers.append(nn.Linear(hidden_units[-1], output_size))
        # Whether to produce predictions.
        if predict:
            layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, num=None):
        embedded = [embed(cat[:, i]) for i, embed in enumerate(self.embedding)]
        if num is not None:
            x = torch.cat([num] + embedded, dim=1)
        else:
            x = torch.cat(embedded, dim=1)

        return self.mlp(x)
