import torch
import numpy as np
from numpy import linalg as LA
import os.path as osp
from torch_geometric.data import Data

def spheres2graph(filename, truncated_num, resolution):
    sa = np.loadtxt(filename, delimiter=' ')
    spheres_num = len(sa)
    if truncated_num > spheres_num:
        raise ValueError('The sphere number exceeds the max sphere number in the file!')
    # normalize radii
    if sa[0,3] > 0:
        r_max = sa[0,3]
    else:
        r_max = 1
    pos = torch.from_numpy(sa[:truncated_num,0:3]).float();
    node_features = torch.from_numpy(sa[:truncated_num,3]/r_max).float()
    node_features = torch.from_numpy(sa[:truncated_num,3]).float()
    node_features = torch.unsqueeze(node_features,-1)
    node_features = torch.cat((pos,node_features),1)
      
    #node_features = torch.ones([truncated_num])
    # build edge_index and edge_attr
    edge_index = torch.tensor([],dtype=torch.long)
    edge_attr = torch.tensor([],dtype=torch.float)
    for sph_i in range(truncated_num):
        for sph_j in range(sph_i, truncated_num):
            c_dist = LA.norm(sa[sph_i,0:3]-sa[sph_j,0:3])
            r_sum = sa[sph_i,3]+sa[sph_j,3]
            if np.absolute(c_dist-r_sum) < resolution:
                temp_tensor = torch.tensor([[sph_i, sph_j],[sph_j, sph_i]],dtype=torch.long)
                edge_index = torch.cat((edge_index, temp_tensor),1)
    edge_num = edge_index.shape[1]
    pseudo = torch.zeros([edge_num, 3])
    for edge_i in range(edge_num):
        i = edge_index[0,edge_i]
        j = edge_index[1, edge_i]
        pseudo[edge_i,0] = (sa[j,0]-sa[i,0]+1)/2
        pseudo[edge_i,1] = (sa[j,1]-sa[i,1]+1)/2
        pseudo[edge_i,2] = (sa[j,2]-sa[i,2]+1)/2        
    edge_attr = torch.zeros([edge_index.shape[1],1])
    adj_mat = torch.zeros([truncated_num, truncated_num], dtype=torch.float)
    edges_num = edge_index.shape[1]/2
    if (edges_num < 512*2):
        zero_adj = True
        return edge_index, node_features, edge_attr, pseudo, zero_adj
    for edge_i in range(int(edges_num)):
        adj_mat[ edge_index[0, edge_i*2], edge_index[1, edge_i*2]]=1
        adj_mat[ edge_index[0, edge_i*2+1],edge_index[1, edge_i*2+1]]=1
    node_degrees = adj_mat.sum(1)
    for col_i in range(truncated_num):
        if node_degrees[col_i]>0:
            adj_mat[:,col_i] = adj_mat[:,col_i]/node_degrees[col_i]
    zero_adj = False
    for edge_i in range(edge_index.shape[1]):
        start_v = edge_index[0,edge_i]
        target_v = edge_index[1, edge_i]
        edge_attr[edge_i,0]=adj_mat[target_v, start_v]
    return edge_index, node_features, edge_attr, pseudo, zero_adj
    