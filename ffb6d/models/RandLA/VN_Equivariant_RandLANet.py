#!/usr/bin/env python
"""Equivariant RandLANet With Vector Neuron"""
__author__      = "Haojie"
__status__ = "incomplete"

from data_robot.VN_RandLANet import Conv2d
# d_out=100
# mlp1 = Conv2d(in_size=10,out_size=d_out,kernel_size=(1,1),bn=False)
# print(mlp1)
#batch * n_points * n_channel*3
#batch * n_points * n_neighbors * n_channel * 3

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    '''
    find k-nearest neighbors idx
    :param x: (batch_size, xyz, num_points)
    :param k: (batch_size, xyz, num_points)
    :return: (batch_size, num_points, k)
    '''
    assert len(x.shape)==3
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    #print('pairwise distance',pairwise_distance.size())
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def vn_gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel*3
    '''
     gather the coordinates or features of neighboring points
    :param pc: (batch*npoint*channel*3)
    :param neighbor_idx: (batch_size, num_points, k)
    :return: (batch*npoint*nsamples*channel*3)
    '''

    assert len(pc.shape) == 4
    batch_size = pc.shape[0]
    num_points = pc.shape[1]
    d = pc.shape[2]
    index_input = neighbor_idx.reshape(batch_size, -1)
    features = torch.gather(pc, 1, index_input.unsqueeze(-1).unsqueeze(dim=-1).repeat(1, 1, d, 3)).contiguous()
    features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d, 3)  # batch*npoint*nsamples*channel*3
    return features

def vn_relative_pos_encoding(vn_pts_xyz, neigh_idx):
    '''
    get equivariant relative pos encoding
    :param vn_pts_xyz: batch*npoint*1*3
    :param neigh_idx: (batch_size, num_points, k)
    :return: batch*npoint*nsamples*4*3
    '''
    assert len(vn_pts_xyz.shape) == 4 and vn_pts_xyz.shape[-2]==1
    neighbor_xyz = vn_gather_neighbour(vn_pts_xyz, neigh_idx)
    xyz_tile = vn_pts_xyz.unsqueeze(dim=2).repeat(1, 1, neigh_idx.shape[-1], 1, 1)
    relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*1*3
    relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1*1 (rho_0)
    relative_dis = relative_dis.repeat(1, 1, 1, 1, 3)  # batch*npoint*nsamples*1*3 (rho_0*3)
    relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-2)  # batch*npoint*nsamples*4*3
    return relative_feature

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [ Batch, N_pts, ... ,N feat, 3]
        '''
        x_out = self.map_to_feat(x.transpose(-2, -1)).transpose(-2, -1)
        return x_out

class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_pts,..., n_channels, 3]
        '''
        d = self.map_to_dir(x.transpose(-2, -1)).transpose(-2, -1)
        dotprod = (x * d).sum(-2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(-2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                    mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + 1e-7)) * d))
        return x_out

class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, share_nonlinearity=False, negative_slope=0.2, ubn=False):
        super(VNLinearLeakyReLU, self).__init__()
        self.linear = VNLinear(in_channels,out_channels)
        self.relu = VNLeakyReLU(out_channels,share_nonlinearity,negative_slope)
        # use batch norm or not
        self.ubn = ubn
        if self.ubn:
            #todo: add batchnorm
            pass

    def forward(self, x):
        # Linear
        x = self.linear(x)
        # BatchNorm todo later
        if self.ubn:
            pass
        # LeakyReLU
        x = self.relu(x)
        return x


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2,ubn=False):
        super(VNStdFeature, self).__init__()
        self.normalize_frame = normalize_frame
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope, ubn=ubn)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, share_nonlinearity=share_nonlinearity,
                                     negative_slope=negative_slope, ubn=ubn)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''

        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(-2, -1)).transpose(-2, -1) #(B,N,C=3,3)

        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]
            # u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + 1e-7)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            # u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + 1e-7)

            # compute the cross product of the two output vectors
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(-2, -1)

        if len(z0.size()) == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        if len(z0.size()) == 4:
            x_std = torch.einsum('bmij,bmjk->bmik', x, z0)
        elif len(z0.size()) == 5:
            x_std = torch.einsum('bmnij,bmnjk->bmnik', x, z0)

        return x_std, z0

class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = VNLinear(d_in, d_in)
        self.mlp = VNLinear(d_in, d_out)
        self.mlp_relu = VNLeakyReLU(d_out)

    def forward(self, feature_set):#features set--> batch*npoint*n_samples*nchannels*3
        '''
        aggregate from neighboring features
        :param feature_set: (batch*npoint*n_samples*nchannels*3)
        :return: (batch*npoint*nchannels*3)
        '''
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=2)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=2, keepdim=True)
        f_agg = self.mlp(f_agg)
        f_agg = self.mlp_relu(f_agg) # out--> batch*npoint*1*nchannels*3
        f_agg = f_agg.squeeze(dim=-3) # batch*npoint*nchannels*3
        return f_agg


class VN_Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = VNLinearLeakyReLU(4, d_out//2)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)
        self.mlp2 = VNLinearLeakyReLU(d_out//2, d_out//2)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):
        '''
        :param xyz:  batch*npoint*1*3
        :param feature: batch*npoint* d_out/2 *3
        :param neigh_idx: batch*npoit*k
        :return: batch*npoint*d_out*3
        '''
        f_xyz = vn_relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*4*3
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = vn_gather_neighbour(feature.contiguous(),neigh_idx).contiguous()  # batch*npoint*nsamples*channel*3
        f_concat = torch.cat([f_neighbours, f_xyz], dim=-2)
        f_pc_agg = self.att_pooling_1(f_concat)  # batch*npoints*channel*3

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = vn_gather_neighbour(f_pc_agg.contiguous(), neigh_idx).contiguous()  # batch*npoint*nsamples*channel*3
        f_concat = torch.cat([f_neighbours, f_xyz], dim=-2)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

class VN_Dilated_res_block(nn.Module):
    '''
    residual connection
    '''
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp1 = VNLinearLeakyReLU(d_in, d_out//2)
        self.lfa = VN_Building_block(d_out)
        self.mlp2 = VNLinearLeakyReLU(d_out, d_out*2,)
        self.shortcut = VNLinearLeakyReLU(d_in, d_out*2)
        self.relu = VNLeakyReLU(in_channels=d_out*2,negative_slope=0.2)
    def forward(self, feature, xyz, neigh_idx):
        '''
        :param feature: batch*npoints*channel*3
        :param xyz: batch*npoints*1*3
        :param neigh_idx: batch*npoints*k
        :return: batch*npoints*2_d_out*3
        '''
        f_pc = self.mlp1(feature)
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # batch*npoints*d_out*3
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        f_pc = self.relu(f_pc+shortcut)
        return f_pc

def vn_random_sample(feature, pool_idx ,pool='average'):
    """
    :param feature: [B, N, d , 3] input features matrix
    :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
    the max_num is k (nearest neighbors)
    :return: pool_features = [B, N', d ,3] pooled features matrix
    todo Max_pooling or average_pooling: I prefer average pooling on (N') neighbors or directly select with N'
    since the max-pooling will be slow but slightly better; since we already aggregate the local information
    """
    if pool == 'average':
        assert len(pool_idx.shape)==3
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[2]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 1, pool_idx.unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, 1, d, 3))
        pool_features = pool_features.reshape(batch_size, -1, num_neigh, d, 3) # batch*npoints*num_neighbor*channel*3
        pool_features = pool_features.mean(dim=2, keepdim=True)  # batch*npoints*1*channel*3
        pool_features = pool_features.squeeze(dim=2)
    elif pool =='direct':
        assert len(pool_idx.shape)==2
        d = feature.shape[2]
        pool_features = torch.gather(feature, 1, pool_idx.unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, 1, d, 3))
    return pool_features

def nearest_interpolation(feature, interp_idx):
    """
    :param feature: [B, npoints, C, 3] input features matrix
    :param interp_idx: [B, up_num_points, 1] nearest neighbour index
    for each unsampled point find its nearest sampled point that owns a latent tensor
    :return: [B, up_num_points, C ,3] interpolated features matrix
    """
    batch_size = interp_idx.shape[0]
    up_num_points = interp_idx.shape[1]
    interp_idx = interp_idx.reshape(batch_size, up_num_points)
    interpolated_features = torch.gather(feature, 1, interp_idx.unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, 1, feature.shape[2], 3))
    return interpolated_features

def nearst_neighbor_match(xyz, sample_index):
    '''
    find the nereast neighbor between two sets of points
    :param xyz: (B,N,3)
    :param sampled_index (B,N')
    :return: nearest_nerighbor: (B,N); sampled_xyz (B, N',3)
    '''
    sampled_xyz = torch.gather(xyz, 1, sample_index.unsqueeze(dim=-1).repeat(1, 1, 3))
    nearest_neighbors = torch.argmax(torch.cdist(xyz, sampled_xyz), dim=-1)
    return nearest_neighbors, sampled_xyz


pts_xyz = torch.rand(10,1024,3)
neigh_idx = knn(pts_xyz.transpose(1,2),16) #(10,1024,16)
vn_pts_xyz = pts_xyz.unsqueeze(dim=-2) #(10, 1024, 1, 3)
print(vn_pts_xyz.size())
neighbor_xyz = vn_gather_neighbour(vn_pts_xyz,neigh_idx) #(10, 1024, 16, 1, 3)
#print(neighbor_xyz.size())
relative_pos_feature = vn_relative_pos_encoding(vn_pts_xyz,neigh_idx) #(10, 1024, 16, 4, 3)
print(relative_pos_feature.size())
vn_linear = VNLinear(in_channels=4,out_channels=10)
relative_pos_feature = vn_linear(relative_pos_feature)
print(relative_pos_feature.size())
vn_relu = VNLeakyReLU(in_channels=10)
relative_pos_feature = vn_relu(relative_pos_feature)
linear_relu = VNLinearLeakyReLU(in_channels=10,out_channels=10)
relative_pos_feature = linear_relu(relative_pos_feature)
print(relative_pos_feature.size())
attention_pooling = Att_pooling(d_in=10,d_out=10)
pooling_feature = attention_pooling(relative_pos_feature)
print('after pooling featues', pooling_feature.size())
block = VN_Building_block(d_out=20)
after_block_feature = block(xyz=vn_pts_xyz,feature=pooling_feature, neigh_idx=neigh_idx)
print('after block feature',after_block_feature.size())
res_block = VN_Dilated_res_block(d_in=10,d_out=20)
after_res_block = res_block(feature=pooling_feature, xyz=vn_pts_xyz, neigh_idx=neigh_idx)
print('after res block',after_res_block.size())
# todo by Haojie random_sample, attention, nearest_interpolation
sample = torch.randint(high=1024,size=(10,1024//4))
sample_neighbors = torch.gather(neigh_idx, dim=1,index=sample.unsqueeze(dim=-1).repeat(1,1,neigh_idx.shape[-1]))
print(sample_neighbors.size())
sampled_feature = vn_random_sample(after_block_feature,sample,pool='direct')
print('direct sampled features',sampled_feature.size())
sampled_feature = vn_random_sample(after_block_feature,sample_neighbors,pool='average')
print('averaged sampled features',sampled_feature.size())

nearest_neighbors, sampled_xyz = nearst_neighbor_match(pts_xyz,sample)
print('nearest neighbor',nearest_neighbors.size())
interpolation_f = nearest_interpolation(sampled_feature,nearest_neighbors)
print('nearest interpolation', interpolation_f.size())

#this mean is over points (instance level)
interpolation_f_mean_tile = interpolation_f.mean(dim=1,keepdim=True).expand(interpolation_f.size())
print('mean tile',interpolation_f_mean_tile.size())
std = VNStdFeature(in_channels=interpolation_f.shape[-2]*2)
invariant_f, z0 = std(torch.cat((interpolation_f,interpolation_f_mean_tile),dim=-2))
print('invariant f',invariant_f.size())
invariant_f = invariant_f.flatten(2)
print('invariant f',invariant_f.size())
# todo for downsteam test