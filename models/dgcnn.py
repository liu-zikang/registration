#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.dcputil import quat2mat

_raw_features_sizes = {'xyz': 3, 'lxyz': 3, 'gxyz': 3, 'ppf': 4, 'pcf': 6}



def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

class Conv2d(torch.nn.Conv2d):

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:

    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)

def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(16, channels),
        }[norm]
    return norm(out_channels)

def get_graph_feature(data, feature_name, k=20):
    xyz = data[:, :3, :]

    idx = knn(xyz, k=k)  
    batch_size, num_points, _ = idx.size()
   
    idx_base = torch.arange(0, batch_size).to(xyz.device).view(-1, 1, 1) * num_points
    feature = torch.tensor([]).to(xyz.device)

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = xyz.size()

    xyz = xyz.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)

    # gxyz
    neighbor_gxyz = xyz.view(batch_size * num_points, -1)[idx, :]
    neighbor_gxyz = neighbor_gxyz.view(batch_size, num_points, k, num_dims)
    if 'gxyz' in feature_name:
        feature = torch.cat((feature, neighbor_gxyz), dim=3)

    # xyz
    xyz = xyz.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, xyz), dim=3)

    # lxyz
    if 'lxyz' in feature_name:
        neighbor_lxyz = neighbor_gxyz - xyz
        feature = torch.cat((feature, neighbor_lxyz), dim=3)

    # ppf
    if 'ppf' in feature_name:
        normal = data[:, 3:6, :]
        normal = normal.transpose(2, 1).contiguous()
        neighbor_norm = normal.view(batch_size * num_points, -1)[idx, :]
        neighbor_norm = neighbor_norm.view(batch_size, num_points, k, num_dims)
        nr_d = angle(normal.permute(0, 2, 1).contiguous()[:,:,None,:], neighbor_lxyz)
        ni_d = angle(neighbor_norm, neighbor_lxyz)
        nr_ni = angle(normal.permute(0, 2, 1).contiguous()[:,:,None,:], neighbor_norm)
        d_norm = torch.norm(neighbor_lxyz, dim=-1)
        ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # (B, npoint, n_sample, 4)
        feature = torch.cat((feature, ppf_feat), dim=3)

    # pcf
    if 'pcf' in feature_name:
        neighbor_gxyz_center = torch.mean(neighbor_gxyz, dim=2, keepdim=True)
        nrnc = neighbor_gxyz_center - xyz
        ncni = neighbor_gxyz - neighbor_gxyz_center
        ninr = xyz - neighbor_gxyz
        nrnc_norm = torch.norm(nrnc, dim=3)
        ncni_norm = torch.norm(ncni, dim=3)
        ninr_norm = torch.norm(ninr, dim=3)
        nr_angle = angle(nrnc, -ninr)
        nc_angle = angle(ncni, -nrnc)
        ni_angle = angle(ninr, -ncni)
        pcf_feat = torch.stack([nrnc_norm, ncni_norm, ninr_norm, nr_angle, nc_angle, ni_angle], dim=-1)
        feature = torch.cat((feature, pcf_feat), dim=3)

    feature = feature.permute(0, 3, 1, 2).contiguous()
    return feature

class DGCNN(nn.Module):
    def __init__(self, features, neighboursnum, emb_dims=512):
        super(DGCNN, self).__init__()

        self.features = features
        self.neighboursnum = neighboursnum
        self.sfe = SFEModule()
        self.ie = IEModule()
        self.stn1 = STN1_Net()
        self.stn2 = STN2_Net()
        self.stn3 = STN3_Net()
        self.stn4 = STN4_Net()
        raw_dim = sum([_raw_features_sizes[f] for f in self.features])  #

       
        self.conv1 = nn.Conv2d(raw_dim, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(512, emb_dims, kernel_size=1, bias=False)  # 512
        self.conv6 = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)
     

    def forward(self, xyz):

        xyz = xyz.permute(0, 2, 1).contiguous()  

        batch_size, num_dims, num_points = xyz.size()
        x = get_graph_feature(xyz, self.features, self.neighboursnum)   
        

        x = F.relu(self.bn1(self.conv1(x)))  
        x1 = x.max(dim=-1, keepdim=True)[0]  
        x1 = x1.squeeze(-1)
        x11=self.stn1(x1)
        x1 = x1.transpose(2, 1)
        x1 = torch.bmm(x1, x11) 
        x1 = x1.transpose(2,1)

        y_feature = self.sfe(x)
        y_feature = y_feature.max(dim=-1, keepdim=True)[0]    
        y_ft = y_feature.squeeze(-1)
        y_pt = self.ie(y_ft) 
 

        x = F.relu(self.bn2(self.conv6(x)))  
        x2 = x.max(dim=-1, keepdim=True)[0]
        x2 = x2.squeeze(-1)
        x22=self.stn2(x2)
        x2 = x1.transpose(2, 1)
        x2 = torch.bmm(x2, x22) 
        x2 = x2.transpose(2,1)

        x = F.relu(self.bn3(self.conv2(x)))  
        x3 = x.max(dim=-1, keepdim=True)[0]
        x3 = x3.squeeze(-1)
        x33=self.stn3(x3)
        x3 = x3.transpose(2, 1)
        x3 = torch.bmm(x3, x33) 
        x3 = x3.transpose(2,1)

        x = F.relu(self.bn4(self.conv3(x)))  
        x4 = x.max(dim=-1, keepdim=True)[0]
        x4 = x4.squeeze(-1)
        x44=self.stn4(x4)
        x4 = x4.transpose(2, 1)
        x4 = torch.bmm(x4, x44) 
        x4 = x4.transpose(2,1)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x_edge = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)

        return y_pt, x_edge

class SFEModule(nn.Module):
    def __init__(self, in_chan=64, out_chan=64, norm="GN"):
        super(SFEModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        
        feat = torch.mul(x, atten)
        
        x = x + feat
        
        feat = self.conv(x)
        
        return feat

class IEModule(nn.Module):  
    def __init__(self, channels=128):
        super(IEModule, self).__init__()
        self.conv1 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(channels)
        self.sie1 = subIEModule(channels)  
        self.sie2 = subIEModule(channels)
        self.sie3 = subIEModule(channels)
        self.sie4 = subIEModule(channels)

    def forward(self, x):
        batch_size, _, N = x.size()  

        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x)))

        x1 = self.sie1(x)
        x2 = self.sie2(x1)
        x3 = self.sie3(x2)
        x4 = self.sie4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x

class subIEModule(nn.Module):  
    def __init__(self, channels):
        super(subIEModule, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False) 
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  
        x_k = self.k_conv(x)  
        x_v = self.v_conv(x)  
      
        energy = torch.bmm(x_q, x_k)  
        attention = self.softmax(energy) 
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        x_r = torch.bmm(x_v, attention)  
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
       
        return x

class STN1_Net(nn.Module):
    def __init__(self):
        super(STN1_Net, self).__init__()
        self.conv1 = torch.nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.fc1= nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64*64)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(64).astype(np.float32))).view(1,64*64).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 64, 64) 
        return x

class STN2_Net(nn.Module):
    def __init__(self):
        super(STN2_Net, self).__init__()
        self.conv1 = torch.nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64*64)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(64).astype(np.float32))).view(1,64*64).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 64, 64) 
        return x

class STN3_Net(nn.Module):
    def __init__(self):
        super(STN3_Net, self).__init__()

        self.conv1 = torch.nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv1d(256, 512, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128*128)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(128).astype(np.float32))).view(1,128*128).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 128,128) 
        return x

class STN4_Net(nn.Module):
    def __init__(self):
        super(STN4_Net, self).__init__()

        self.conv1 = torch.nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv1d(256, 512, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256*256)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(256).astype(np.float32))).view(1,256*256).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 256, 256)
        return x

class Classify1(nn.Module):
    def __init__(self, emb_dims=20):
        super(Classify, self).__init__()
        self.conv1 = nn.Conv1d(emb_dims, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, y):
        x = x.permute(0, 2, 1).contiguous()
        y = y.permute(0, 2, 1).contiguous()
        batch_size, _, num_points = x.size()
        x = get_graph_feature_cross(x, y)   
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x_inlier = torch.sigmoid(self.conv3(x)).permute(0,2,1).contiguous()


        if torch.sum(torch.isnan(x_inlier)):
            print('discover nan value')
        return x_inlier

def knn_cross(x, y, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), y)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1).contiguous()

    dist = pairwise_distance.topk(k=k, dim=-1)[0]  # (batch_size, num_points, k)
    return dist

def get_graph_feature_cross(x, y, k=20):
    dist = knn_cross(x, y, k=k)  # (batch_size, num_points, k)
    dist = dist.permute(0, 2, 1).contiguous()
    return dist

class Classify2(nn.Module):
    def __init__(self, emb_dims=512):
        super(Classify2, self).__init__()
        self.conv00 = nn.Conv1d(emb_dims, emb_dims, kernel_size=1, bias=False)
        self.conv01 = nn.Conv1d(emb_dims, emb_dims, kernel_size=1, bias=False)
        self.bn00 = nn.BatchNorm1d(emb_dims)
        self.bn01 = nn.BatchNorm1d(emb_dims)
        self.conv1 = nn.Conv1d(emb_dims*2, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, y):
        batch_size, _, num_points = x.size()
        y = F.relu(self.bn00(self.conv00(y)))
        y = F.relu(self.bn01(self.conv01(y)))
        y = torch.max(y, dim=2, keepdim=True)[0].repeat(1, 1, num_points)
        x = torch.cat((x,y), dim=1)

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x_inlier = torch.sigmoid(self.conv3(x)).permute(0,2,1).contiguous()

        if torch.sum(torch.isnan(x_inlier)):
            print('discover nan value')
        return x_inlier

class Classify(nn.Module):
    def __init__(self, emb_dims=512):
        super(Classify, self).__init__()
        self.conv1 = nn.Conv1d(emb_dims, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.permute(0,2,1).contiguous()
        batch_size, _, num_points = x.size()
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x_inlier = torch.sigmoid(self.conv3(x)).permute(0,2,1).contiguous()

        if torch.sum(torch.isnan(x_inlier)):
            print('discover nan value')
        return x_inlier

class s_weight(nn.Module):
    def __init__(self, emb_dims=512):
        super(s_weight, self).__init__()
        self.prepool = nn.Sequential(
            nn.Conv1d(emb_dims+1, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),

            nn.Linear(512, 256),

            nn.Linear(256, 1),
        )

    def forward(self, src, tgt):
        src_padded = F.pad(src, (0, 1), mode='constant', value=0)
        ref_padded = F.pad(tgt, (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])

        return beta

