import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim):
        super(NumericalEmbedding, self).__init__()
        # Create a parameter for each numerical feature.
        self.embedding = nn.ParameterList([nn.Parameter(torch.randn(
            1, embed_dim) * 0.01) for _ in range(num_features)])

    def forward(self, num):
        # num should be of shape [batch_size, num_fields, 1].
        _, num_fields, _ = num.shape
        dense_kd_embed = []
        for i in range(num_fields):
            # Extract the feature and apply the corresponding embedding.
            feature = num[:, i, :]
            scaled_embed = feature * self.embedding[i]
            scaled_embed = scaled_embed.unsqueeze(1)
            dense_kd_embed.append(scaled_embed)
        # Concatenate to get [batch_size, num_fields, embed_dim].
        return torch.cat(dense_kd_embed, dim=1)
    

class AutoInt(torch.nn.Module):
    def __init__(self, num_num, cat_dims, embed_size, atten_embed_dim,
                 output_size, num_heads, num_layers, dropout=0, 
                 has_residual=True, predict=False):
        super(AutoInt, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(c, embed_size) for c
                                        in cat_dims])
        feat_num = num_num + len(cat_dims)
        self.numerical_embedding = NumericalEmbedding(num_num, embed_size)
        self.atten_embedding = torch.nn.Linear(embed_size, atten_embed_dim)
        self.embed_output_dim = feat_num * embed_size
        self.atten_output_dim = feat_num * atten_embed_dim
        self.has_residual = has_residual
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim,
                                        num_heads, dropout=dropout)
                                        for _ in range(num_layers)
        ])
        self.out = torch.nn.Linear(self.atten_output_dim, output_size)
        if has_residual:
            self.res_embedding = torch.nn.Linear(embed_size, atten_embed_dim)
        self.predict = predict

    def forward(self, cat, num=None):
        cat_embed = [embed(cat[:, i]).unsqueeze(1) for
                     i, embed in enumerate(self.embedding)]
        cat_embed = torch.cat(cat_embed, dim=1)
        if num is not None:
            num_embed = self.numerical_embedding(num.unsqueeze(-1))
            x = torch.cat([num_embed, cat_embed], dim=1)
        else:
            x = cat_embed
        atten_x = self.atten_embedding(x)
        cross_term = atten_x.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.res_embedding(x)
            cross_term += V_res
        cross_term = F.relu(cross_term).contiguous().view(
            -1, self.atten_output_dim)
        pred = self.out(cross_term)
        if self.predict:
            pred = torch.sigmoid(pred)
        return pred
