import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.batchnorm import BatchNorm2d
import torch_scatter
import math


class Trans_dim(nn.Module):
    def __init__(self, dim_in, dim_out, use_bn=False) -> None:
        super(Trans_dim, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)
        )
    def forward(self, cylinder_polar):

        return self.conv(cylinder_polar)

class Cross_Attn(nn.Module):
    def __init__(self, dim_cylinder, dim_polar, height=32):
        super(Cross_Attn, self).__init__()
        self.trans_dim = Trans_dim(dim_in=dim_cylinder, dim_out=dim_polar)
        # self.dd = nn.Sequential(
        #     nn.BatchNorm1d(dim_cylinder),
        #     nn.Linear(dim_cylinder, dim_cylinder),
        # )
        
        self.height = height
        # pe = torch.zeros(self.height, dim_cylinder)
        # height = torch.arange(0, self.height).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, dim_cylinder, 2) *
        #                      -(math.log(10000.0) / dim_cylinder))
        # pe[:, 0::2] = torch.sin(height * div_term)
        # pe[:, 1::2] = torch.cos(height * div_term)
        # pe = pe
        
        # leanrnable params
        pe = torch.randn(self.height, dim_cylinder)
        self.register_buffer('pe', pe)

    def forward(self, cylinder_fea, polar_fea):
        
        current_device = polar_fea.get_device()
        layer_polar = polar_fea  # [B, dim, x, y]
        layer_cylinder = cylinder_fea  # [B, dim, x, y, z]
        cylinder_features, cylinder_indices = layer_cylinder.features, layer_cylinder.indices
        
        # ppe = Variable(self.pe, requires_grad=True)
        # ppe_cylinder = ppe[cylinder_indices[:, 3].to(torch.int64), :]
        cylinder_features = cylinder_features #* ppe_cylinder
        unq, unq_inv, unq_cnt = \
            torch.unique(cylinder_indices[:,:3], return_inverse=True, return_counts=True, dim=0)
        cylinder_features = torch_scatter.scatter_max(cylinder_features,unq_inv, dim=0)[0]
        cylinder_polar_size = (polar_fea.shape[0], cylinder_features.shape[1],
                               polar_fea.shape[2],polar_fea.shape[3])
        cylinder_polar = torch.zeros(size=cylinder_polar_size, dtype=torch.float32).to(current_device)
        unq = unq.type(torch.int64)
        cylinder_polar[unq[:, 0], :, unq[:, 1], unq[:, 2]] = cylinder_features
        cylinder_polar = self.trans_dim(cylinder_polar)
        # TODO now have same shape of feature shape except of dim of feature
        assert cylinder_polar.shape == layer_polar.shape
        return cylinder_polar, layer_polar


class KD_Part(nn.Module):
    def __init__(self, dim_cylinder_list, dim_polar_list) -> None:
        super(KD_Part, self).__init__()
        self.cylinder_dim = dim_cylinder_list
        self.polar_dim = dim_polar_list
        self.trans_list = nn.ModuleList()
        for i in range(len(dim_cylinder_list)):
            self.trans_list.append(Cross_Attn(dim_cylinder_list[i], dim_polar_list[i]))

    def forward(self, cylinder_list, polar_list):
        cylinder_kd = []
        polar_kd = []
        for i in range(len(cylinder_list)):
            cylinder_polar, layer_polar = self.trans_list[i](cylinder_list[i], polar_list[i])
            cylinder_kd.append(cylinder_polar)
            polar_kd.append(layer_polar)
        return cylinder_kd, polar_kd