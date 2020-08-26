import argparse
import sys
import torch
import time
import scipy.io as sio
import h5py
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def readfile(path):
    print('reading file ...')
    data = h5py.File(path)
    x_train = []
    x_label = []
    val_data = []
    val_label = []

    x_train = data['train_data']
    x_label = data['train_label']
    val_data = data['val_data']
    val_label = data['val_label']
    
    x_train = np.array(x_train, dtype=float)
    val_data = np.array(val_data, dtype=float)
    x_label = np.array(x_label, dtype=int)
    val_label = np.array(val_label, dtype=int)
    x_train = torch.FloatTensor(x_train)
    val_data = torch.FloatTensor(val_data)
    x_label = torch.LongTensor(x_label)
    val_label = torch.LongTensor(val_label)

    return x_train, x_label, val_data, val_label


class CNNnet(torch.nn.Module):
      def __init__(self, node_number, batch_size):
          super(CNNnet,self).__init__()
          self.node_number = node_number
          self.batch_size = batch_size
          self.aggregate_weight = torch.nn.Parameter(tensor([[[1],[1],[torch.linspace(1, node_numbe, node_number)]]]))
          self.conv1 = torch.nn.Sequential(
              torch.nn.Conv1d(in_channels=1,
                              out_channels=8,
                              kernel_size=9,
                              stride=1,
                              padding=4),
              torch.nn.BatchNorm1d(8),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              #torch.nn.AvgPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.conv2 = torch.nn.Sequential(
              torch.nn.Conv1d(8,16,9,1,4),
              torch.nn.BatchNorm1d(16),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              #torch.nn.AvgPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.mlp1 = torch.nn.Sequential(
              torch.nn.Linear(1024*16,50),
              torch.nn.Dropout(0.5),
          )
          self.mlp2 = torch.nn.Linear(50,2)
      def forward(self, x):
          x = torch.matmul(self.aggregate_weight, x)
          x = self.conv1(x)
          x = self.conv2(x)
          x = self.mlp1(x.view(x.size(0),-1))
          x = self.mlp2(x)
          return x

def main():

    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="dataset/dataset.mat", help='path of the dataset (default: data/data.mat)')
    parser.add_argument('--node_number', type=int, default=256, help='node number of graph (default: 256)')
    parser.add_argument('--batch_size', type=int, default=32, help='number of input size (default: 128)')

    args = parser.parse_args()

    x_train, x_label, val_data, val_label = readfile(args.dataset)   
    x_label = torch.squeeze(x_label, dim=1).long()
    x_label = torch.squeeze(x_label, dim=0).long()

    val_label = torch.squeeze(val_label, dim=1).long()    
    val_label = torch.squeeze(val_label, dim=0).long()    

    train_set = TensorDataset(x_train, x_label)
    val_set = TensorDataset(val_data, val_label)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CNNnet(args.node_number, args.batch_size)
    
    model  
    loss = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)   # optimize all model parameters
    loss_func = torch.nn.CrossEntropyLoss()
    best_acc = 0.0

    num_epoch = 100
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            train_pred = model(data[0]) 
            batch_loss = loss(train_pred, data[1])  
            batch_loss.backward()  
            optimizer.step()  

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

            #progress = ('#' * int(float(i) / len(train_loader) * 40)).ljust(40)
            #print('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, \
            #                                          (time.time() - epoch_start_time), progress), end='\r', flush=True)

        model.eval()
        
        
        val_TP = 1.0
        val_TN = 1.0
        val_FN = 1.0
        val_FP = 1.0
        predict_total = []
        label_total = []
    
        for i, data in enumerate(val_loader):
            val_pred = model(data[0])
            batch_loss = loss(val_pred, data[1])
    
            predict_val = np.argmax(val_pred.cpu().data.numpy(), axis=1)
            predict_total = np.append(predict_total, predict_val)
            label_val = data[1].numpy()
            label_total = np.append(label_total, label_val)
            
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
            
            
            #progress = ('#' * int(float(i) / len(val_loader) * 40)).ljust(40)
            #print('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, \
            #                                           (time.time() - epoch_start_time), progress), end='\r', flush=True)
            
            #test_pred = val_pred.max(1, keepdim=True)[1]
            #test_labels_view = data[1].cuda().view_as(test_pred)
            
            
        val_TP = ((predict_total == 1) & (label_total == 1)).sum().item()
        val_TN = ((predict_total == 0) & (label_total == 0)).sum().item()
        val_FN = ((predict_total == 0) & (label_total == 1)).sum().item()
        val_FP = ((predict_total == 1) & (label_total == 0)).sum().item()
    
        val_spe = val_TN/(val_FP + val_TN + 0.0001)
        val_rec = val_TP/(val_TP + val_FN + 0.0001)
        test_acc = (val_TP+val_TN)/(val_FP + val_TN + val_TP + val_FN + 0.0001)
            
        val_acc = val_acc / val_set.__len__()
        print('%3.5f  %3.5f  %3.5f  %3.5f' % (train_acc / train_set.__len__(), train_loss, val_acc, val_loss))

        if (val_acc > best_acc):
            with open('save/DEF_V4097_5.txt', 'w') as f:
                f.write(str(epoch) + '\t' + str(val_acc) + '\t' + str(val_spe) + '\t' + str(val_rec) + '\n')
            torch.save(model.state_dict(), 'save/model.pth')
            best_acc = val_acc
            #print('Model Saved!')

    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(param[0])

if __name__ == '__main__':
