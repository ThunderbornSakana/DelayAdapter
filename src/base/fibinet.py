import torch
import torch.nn as nn
import itertools


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
    

"""
  Input shape
    - ``(batch_size, field_size, embedding_size)``.
  Output shape
    - ``(batch_size, field_size, embedding_size)``.
  Arguments
    @param field_size - The number of feature groups.
"""
class SENETLayer(nn.Module):
    def __init__(self, field_size, reduction_ratio=3):
        super(SENETLayer, self).__init__()
        reduction_size = max(1, field_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(field_size, reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(reduction_size, field_size, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        Z = torch.mean(x, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(x, torch.unsqueeze(A, dim=2))
        return V
    

"""
  Input shape
    - ``(batch_size, field_size, embedding_size)``.
  Output shape
    - ``(batch_size, field_size * (field_size - 1) / 2, embedding_size)``.
"""
class BilinearInteraction(nn.Module):
    def __init__(self, field_size, embedding_size, bilinear_type="each"):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.bilinear = nn.ModuleList()
        if self.bilinear_type == "all":
            self.bilinear = nn.Linear(
                embedding_size, embedding_size, bias=False)
        elif self.bilinear_type == "each":
            for _ in range(field_size):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        elif self.bilinear_type == "interaction":
            for _, _ in itertools.combinations(range(field_size), 2):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = torch.split(x, 1, dim=1)
        if self.bilinear_type == "all":
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.combinations(x, 2)]
        elif self.bilinear_type == "each":
            p = [torch.mul(self.bilinear[i](x[i]), x[j])
                 for i, j in itertools.combinations(range(len(x)), 2)]
        elif self.bilinear_type == "interaction":
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in
                 zip(itertools.combinations(x, 2), self.bilinear)]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=1)
    

"""
  @param bilinear_type - Can be "all" , "each" or "interaction".
  @param reduction_ratio - An integer in [1, inf).
"""
class FiBiNET(nn.Module):
    def __init__(self, num_num, cat_dims, embed_size, hidden_units,
                 dropout, output_size, bilinear_type='each', 
                 reduction_ratio=3, predict=False):
        super(FiBiNET, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(c, embed_size) for c
                                        in cat_dims])
        feat_num = num_num + len(cat_dims)
        self.numerical_embedding = NumericalEmbedding(num_num, embed_size)
        self.SE = SENETLayer(feat_num, reduction_ratio)
        self.bilinear_output_size = feat_num * (feat_num - 1) * embed_size
        self.bilinear = BilinearInteraction(feat_num, embed_size, bilinear_type)
        if len(hidden_units) > 0:
            layers = []
            last_hidden_size = self.bilinear_output_size
            for hidden_size in hidden_units:
                layers.append(nn.Linear(last_hidden_size, hidden_size))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                last_hidden_size = hidden_size
            self.dnn = nn.Sequential(*layers)
        self.out = nn.Linear(last_hidden_size, output_size)
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
        senet_output = self.SE(x)
        senet_bilinear_out = self.bilinear(senet_output)
        bilinear_out = self.bilinear(x)
        hid = torch.cat((senet_bilinear_out, bilinear_out), dim=1)
        hid = self.dnn(hid.view(-1, self.bilinear_output_size))
        pred = self.out(hid)
        if self.predict:
            pred = torch.sigmoid(pred)
        return pred