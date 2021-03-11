import torch
import numpy as np
from numpy import linalg as LA
import os.path as osp
from optparse import OptionParser

from graph_dataset import *
from torch_geometric.data import DataLoader
from gcnn_models import *
from plot_confusion_matrix import plot_confusion_matrix 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("--gcnn_name", dest="gcnn_name")

(options, args) = parser.parse_args()
num_node_features = 4
num_classes = 10
gcnn_name = options.gcnn_name
dataset_name = 'ModelNet10'
class_names = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']
path = osp.join('../dataset', dataset_name)
test_dataset = Graph_Dataset(path, '10', False)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=4)
device = torch.device('cuda:0')
host = torch.device('cpu')

if gcnn_name == 'spline':
    model = SplineConvNet(num_node_features, num_classes)
    checkpoint = torch.load("../checkpoints/ModelNet10SplineConvNet_light_checkpoint.pth")
elif gcnn_name == 'tag':
   model = TAGConvNet(num_node_features, num_classes)
   checkpoint = torch.load("../checkpoints/ModelNet10TAGConvNet_light_checkpoint.pth")
elif gcnn_name == 'arma':
    model = ARMAConvNet(num_node_features, num_classes)
    checkpoint = torch.load("../checkpoints/ModelNet10ARMAConvNet_light_checkpoint.pth")
else:
    raise NameError("Wrong gcnn name.")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005, weight_decay=0.001)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters: {:d}'.format(total_params))
model.eval()
test_metrics = {"acc": []}
y_true = torch.ones([0]).int()
y_pred = torch.ones([0]).int()
for batch_i, data in enumerate(test_loader):
    data = data.to(device)
    with torch.no_grad():
        predictions = model(data)
    acc = 100 * (predictions.detach().argmax(1) == data.y).cpu().numpy().mean()    
    test_metrics["acc"].append(acc)
    pred = predictions.argmax(1).to(host)
    y = data.y.to(host)
    y_true = torch.cat((y_true.int(), y.int()))
    y_pred = torch.cat((y_pred, pred.int()))
print('Accuracy: {:f}'.format(np.mean(test_metrics["acc"])))   
cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
cm = np.array(cm)
print('The confusion matrix:')
print(cm)
plot_confusion_matrix(cm, class_names, normalize=True)
