import argparse
import sys
import torch
import time
import scipy.io as sio
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def readfile(path):
    print('reading file ...')
    data = sio.loadmat(path)
    x_train = []
    x_label = []
    val_data = []
    val_label = []

    x_train = data['train_data']
    x_label = data['train_label']
    val_data = data['test_data']
    val_label = data['test_label']
    
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
      def __init__(self, node_number, batch_size, k_hop):
          super(CNNnet,self).__init__()
          self.node_number = node_number
          self.batch_size = batch_size
          self.k_hop = k_hop
          self.aggregate_weight = torch.nn.Parameter(torch.rand(1, 1, node_number))
          self.conv1 = torch.nn.Sequential(
              torch.nn.Conv1d(in_channels=1,
                              out_channels=8,
                              kernel_size=3,
                              stride=1,
                              padding=1),
              torch.nn.BatchNorm1d(8),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              #torch.nn.AvgPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.conv2 = torch.nn.Sequential(
              torch.nn.Conv1d(8,16,3,1,1),
              torch.nn.BatchNorm1d(16),
              torch.nn.ReLU(),
              torch.nn.MaxPool1d(kernel_size=2),
              #torch.nn.AvgPool1d(kernel_size=2),
              torch.nn.Dropout(0.2),
          )
          self.mlp1 = torch.nn.Sequential(
              torch.nn.Linear(64*16,50),
              torch.nn.Dropout(0.5),
          )
          self.mlp2 = torch.nn.Linear(50,2)
      def forward(self, x):
          tmp_x = x
          for _ in range(self.k_hop):
              tmp_x = torch.matmul(tmp_x, x)
          x = torch.matmul(self.aggregate_weight, tmp_x)
          x = self.conv1(x)
          x = self.conv2(x)
          x = self.mlp1(x.view(x.size(0),-1))
          x = self.mlp2(x)
          return x

def main():

    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="dataset/AEF_V_0.mat", help='path of the dataset (default: data/data.mat)')
    parser.add_argument('--node_number', type=int, default=256, help='node number of graph (default: 256)')
    parser.add_argument('--batch_size', type=int, default=32, help='number of input size (default: 128)')
    parser.add_argument('--k_hop', type=int, default=4, help='times of aggregate (default: 1)')

    args = parser.parse_args()

    x_train, x_label, val_data, val_label = readfile(args.dataset)   # 'train.csv'
    x_train = x_train.permute(2, 0, 1)
    x_label = torch.squeeze(x_label, dim=1).long()

    val_data = val_data.permute(2, 0, 1)
    val_label = torch.squeeze(val_label, dim=1).long()    

    train_set = TensorDataset(x_train, x_label)
    val_set = TensorDataset(val_data, val_label)

    #batch_size = 128
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = CNNnet(args.node_number, args.batch_size, args.k_hop)
    #print(model) 
    model  
    loss = torch.nn.CrossEntropyLoss()
    #para = list(model.parameters())
    #print(para)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # optimize all cnn parameters
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
            #print(train_pred.size())
            #print(data[1].size()) 
            batch_loss = loss(train_pred, data[1])  
            batch_loss.backward()  
            optimizer.step()  

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

            

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
            
            
            
        val_TP = ((predict_total == 1) & (label_total == 1)).sum().item()
        val_TN = ((predict_total == 0) & (label_total == 0)).sum().item()
        val_FN = ((predict_total == 0) & (label_total == 1)).sum().item()
        val_FP = ((predict_total == 1) & (label_total == 0)).sum().item()
    
        val_spe = val_TN/(val_FP + val_TN + 0.001)
        val_rec = val_TP/(val_TP + val_FN + 0.001)
        test_acc = (val_TP+val_TN)/(val_FP + val_TN + val_TP + val_FN + 0.001)
                           
        val_acc = val_acc / val_set.__len__()
        print('%3.6f   %3.6f   %3.6f   %3.6f' % (train_acc / train_set.__len__(), train_loss, val_acc, val_loss))

        if (val_acc > best_acc):
            with open('save/AET_V_0.txt', 'w') as f:
                f.write(str(epoch) + '\t' + str(val_acc) + '\t' + str(val_spe) + '\t' + str(val_rec) + '\n')
            torch.save(model.state_dict(), 'save/model.pth')
            best_acc = val_acc
            
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(param[0])

if __name__ == '__main__':
    main()
