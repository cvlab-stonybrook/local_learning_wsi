import torch
from torch import nn
import torch.nn.functional as F


class SpatialPyramidPool(nn.Module):
    def __init__(self, out_pool_sizes=((1, 1), (2, 2), (4, 4)), pooling='max'):
        super(SpatialPyramidPool, self).__init__()
        self.out_pool_sizes = out_pool_sizes
        assert pooling in ('max', 'avg')
        pooling_class = nn.AdaptiveMaxPool2d if pooling == 'max' else nn.AdaptiveAvgPool2d
        self.poolings = nn.ModuleList([
            pooling_class(pool_size)
            for pool_size in self.out_pool_sizes
        ])
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.cat([
            self.flatten(pooling(x))
            for pooling in self.poolings
        ], dim=-1)
        return x

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, mid_dim, out_dim=1, flatten=False, dropout=0.):
        super(AttentionPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.flatten = flatten

        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.out_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.squeeze(0)

        H = x.view(x.shape[0], -1)  # d * L
        H = torch.transpose(H, 1, 0)  # L * d

        A = self.attention(H)  # L * K
        A = torch.transpose(A, 1, 0)  # K * L
        A = F.softmax(A, dim=1)  # softmax over L
        A = self.dropout(A)

        M = torch.mm(A, H)  # Kxd

        if self.flatten:
            M = M.flatten()
        return M.unsqueeze(0)


class CnnAttentionPooling(nn.Module):
    def __init__(self, feature_dim, mid_dim, out_dim=1, kernel_size=(3, 3), padding=(1, 1), flatten=False, dropout=0.):
        super(CnnAttentionPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.flatten = flatten

        self.attention = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.mid_dim, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.Conv2d(self.mid_dim, self.out_dim, kernel_size=kernel_size, padding=padding)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = b, c, h, w
        A = self.attention(x)  # b, 1, h, w
        A = torch.flatten(A, start_dim=1)  # b, h * w
        A = F.softmax(A, dim=1)  # softmax over h * w
        A = self.dropout(A).unsqueeze(-1)  # b, hw, 1

        x = x.reshape(*x.shape[:2], -1)  # b, c, hw
        x = torch.transpose(x, 1, 2)  # b, hw, c
        M = A * x
        M = M.sum(dim=1)

        if self.flatten:
            M = M.flatten()
        return M


class GatedAttentionPooling(nn.Module):
    def __init__(self, feature_dim, mid_dim, out_dim=1, flatten=False, dropout=0.):
        super(GatedAttentionPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.flatten = flatten

        self.attention_V = nn.Sequential(
            nn.Linear(self.feature_dim, self.mid_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.feature_dim, self.mid_dim),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.mid_dim, self.out_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.squeeze(0)

        H = x.view(x.shape[0], -1)  # d * L
        H = torch.transpose(H, 1, 0)  # L * d

        A_V = self.attention_V(H)  # L * mid
        A_U = self.attention_U(H)  # L * mid
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # L * K
        A = torch.transpose(A, 1, 0)  # K x L
        A = F.softmax(A, dim=1)  # softmax over L
        A = self.dropout(A)

        M = torch.mm(A, H)  # Kxd

        if self.flatten:
            M = M.flatten()

        return M.unsqueeze(0)