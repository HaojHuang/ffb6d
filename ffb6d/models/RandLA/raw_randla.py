#!/usr/bin/env python
"""Equivariant RandLANet With Vector Neuron"""
__author__      = "Haojie, David"
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
    features = torch.gather(pc, 1, index_input.unsqueeze(-1).unsqueeze(dim=-1).repeat(1, 1, d,3)).contiguous()
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

pts_xyz = torch.rand(10,1024,3)
neigh_idx = knn(pts_xyz.transpose(1,2),16) #(10,1024,16)
vn_pts_xyz = pts_xyz.unsqueeze(dim=-2) #(10, 1024, 1, 3)
print(vn_pts_xyz.size())
neighbor_xyz = vn_gather_neighbour(vn_pts_xyz,neigh_idx) #(10, 1024, 16, 1, 3)
#print(neighbor_xyz.size())
relative_pos_feature = vn_relative_pos_encoding(vn_pts_xyz,neigh_idx) #(10, 1024, 16, 4, 3)
print(relative_pos_feature.size())
# todo by Haojie random_sample, attention, nearest_interpolation
