import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, BatchNorm2d as BN2d
from torch.nn import Dropout
from torch_geometric.nn import PointConv, fps, radius,SplineConv, TAGConv, TopKPooling, ChebConv,ARMAConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class SplineConvNet(torch.nn.Module):
    def __init__(self, node_features, classes_num, K=3):
        super(SplineConvNet, self).__init__()
        self.conv1 = SplineConv(node_features, 128, 3, K)
        self.bn1 = BN(128)
        self.conv2 = SplineConv(128, 128, 3, K)
        self.bn2 = BN(128)
        self.lin1 = MLP([256, 256])
        self.final = torch.nn.Sequential(
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.5),            
            torch.nn.Linear(128, classes_num), 
        )
        
    def forward(self, data):
        x, edge_index, pseudo,  batch = data.x.float(), data.edge_index, data.pseudo, data.batch
        x1 = F.relu(self.conv1(x, edge_index, pseudo))
        x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, edge_index, pseudo))
        x2 = self.bn2(x2)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = gmp(out, batch)
        out = self.final(out)
        return F.log_softmax(out, dim=-1)
        
class TAGConvNet(torch.nn.Module):
    def __init__(self, node_features, classes_num, K=3, num_filters=128, dropout=0.5):
        super(TAGConvNet, self).__init__()
        self.conv1 = TAGConv(node_features, 128, 3, K)
        self.bn1 = BN(128)
        self.conv2 = TAGConv(128, 128, 3, K)
        self.bn2 = BN(128)
        self.lin1 = MLP([256, 256])
        self.final = torch.nn.Sequential(
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.5),
            
            torch.nn.Linear(128, classes_num), 
        )
        self.num_filters = num_filters
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x.float(), data.edge_index, data.batch, data.edge_attr.float()    
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x2 = self.bn2(x2)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = gmp(out, batch)
        out = self.final(out)
        return F.log_softmax(out, dim=-1)
         
        
class ARMAConvNet(torch.nn.Module):
    def __init__(self, node_features, classes_num, K=3):
        super(ARMAConvNet, self).__init__()
        self.conv1 = ARMAConv(node_features, 128, 3, K)
        self.bn1 = BN(128)
        self.conv2 = ARMAConv(128, 128, 3, K)
        self.bn2 = BN(128)
        self.lin1 = MLP([256, 256])
        self.final = torch.nn.Sequential(
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.5),
            
            torch.nn.Linear(128, classes_num), 
        )   
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x.float(), data.edge_index, data.batch, data.edge_attr.float()    
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x2 = self.bn2(x2)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = gmp(out, batch)
        out = self.final(out)
        return F.log_softmax(out, dim=-1)