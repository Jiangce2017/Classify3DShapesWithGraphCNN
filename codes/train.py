import torch
import numpy as np
import os.path as osp
from optparse import OptionParser

from graph_dataset import Graph_Dataset
from torch_geometric.data import DataLoader
from utils import Logger
from gcnn_models import *
from warmup_scheduler import GradualWarmupScheduler

parser = OptionParser()
parser.add_option("--gcnn_name", dest="gcnn_name")

(options, args) = parser.parse_args()
num_node_features = 4
num_classes = 10
num_epochs = 400
gcnn_name = options.gcnn_name
dataset_name = 'ModelNet10'
path = osp.join('../dataset', dataset_name)

exp_name = dataset_name + gcnn_name + 'test'

train_dataset = Graph_Dataset(path, '10', True)
print(len(train_dataset))
test_dataset = Graph_Dataset(path, '10', False)
print(len(test_dataset))
print('Dataset loaded.')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True,
                              num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, drop_last=True,
                             num_workers=6)

device = torch.device('cuda:0')

if gcnn_name == 'spline':
    model = SplineConvNet(num_node_features, num_classes)
elif gcnn_name == 'tag':
   model = TAGConvNet(num_node_features, num_classes)
elif gcnn_name == 'arma':
    model = ARMAConvNet(num_node_features, num_classes)
else:
    raise NameError("Wrong gcnn name.")

model.to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss().to(device)


scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)


print(exp_name)
result_path = osp.join('..', 'results')
train_logger = Logger(
    osp.join(result_path, exp_name+'_train.log'),
    ['epoch','num_epochs','loss','acc']
)
val_logger = Logger(
    osp.join(result_path, exp_name+'_val.log'), 
    ['epoch','num_epochs','acc']
)

best_acc = 0
def train():
    model.train()
    train_metrics = {"loss": [], "acc": []}
    for batch_i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        predictions = model(data)
        loss = F.nll_loss(predictions, data.y)
        loss.backward()
        optimizer.step()
        acc = 100 * (predictions.detach().argmax(1) == data.y).cpu().numpy().mean()
        train_metrics["loss"].append(loss.item())
        train_metrics["acc"].append(acc)
    return np.mean(train_metrics["acc"]), np.mean(train_metrics["loss"])
        
def test():
    model.eval()
    test_metrics = {"acc": []}
    correct = 0
    for batch_i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            predictions = model(data)
        acc = 100 * (predictions.detach().argmax(1) == data.y).cpu().numpy().mean()
        test_metrics["acc"].append(acc)
    return np.mean(test_metrics["acc"])


for epoch in range(num_epochs):
    scheduler_warmup.step() 
    train_acc, train_loss = train()
    test_acc= test()
    is_best = test_acc > best_acc
    best_acc = max(best_acc, test_acc)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc': best_acc
    }
    if is_best:
        torch.save(state, '%s/%s_checkpoint.pth' % (result_path, exp_name))

    train_logger.log({
        'epoch': epoch,
        'num_epochs': num_epochs,              
        'loss': train_loss,
        'acc': train_acc
    })  
    val_logger.log({
        'epoch': epoch,
        'num_epochs': num_epochs,
        'acc': test_acc
    })
    print(exp_name)
    log = 'Epoch: {:03d}, Train_Loss: {:.4f}, Train_Acc: {:.4f}, Test_Acc: {:.4f}'
    print(log.format(epoch, train_loss, train_acc, test_acc))
