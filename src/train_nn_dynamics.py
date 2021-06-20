import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import sys, os
from dynamics_dataset import GymDynamicsDataset
from dynamics_model import FC
sys.path.append('../')
import matplotlib.pyplot as plt
from utils import convert


def train(dataset, model, prefix, epochs, lr, test_only=False, load_file=None):

    if not os.path.exists('../model/'+prefix):
        os.makedirs('../model/'+prefix)
        os.makedirs('../nnet/'+prefix)

    # define train set and validation set
    
    indices = torch.randperm(len(dataset)).tolist()
    train_set_size = int(0.9 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, indices[:train_set_size])
    test_set = torch.utils.data.Subset(dataset, indices[train_set_size:])
    
    # define data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_his = []

    if load_file:
        model.load_state_dict(torch.load(load_file))
        print("model loaded")
        avg_loss = test(model, test_loader, criterion)
        print("avg_loss: ", avg_loss)
    
    if test_only:
        avg_loss = test(model, test_loader, criterion)
        print('Average L2 loss: %f' % (avg_loss))
        
    save_idx = -1
    save_path_prefix = '../model/'+prefix+'/training_'

    # training

    # plt.figure()

    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        PATH = '../model/'+prefix+'/epoch_'+str(epoch)+'.pth'
        if epoch % 100 == 0:
            torch.save(model.state_dict(), PATH)
        print('epoch %d' % (epoch + 1))
        print('train loss: %f' % (train_loss))
        loss_his.append(train_loss)

        if epoch % 100 == 0:
            test_loss = test(model, test_loader, criterion)
            print('test  loss: %f' % (test_loss))
            # plt.plot(loss_his, color='b')
            # plt.pause(0.01)
    print("minimum loss epoch:", np.argmin(loss_his))
    print("minimum loss: ", np.min(loss_his))
    print('Finished Training')
    # plt.show()

def train_epoch(model, train_loader, optimizer, criterion):
    #train for one epoch
    running_loss = 0.0
    train_set_size = 0
    saved_cnt = 0

    for i, data in enumerate(train_loader, 0):
        train_set_size += 1

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # outputs = model.forward(inputs.float()).float()
        outputs = model.forward(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    running_loss /= train_set_size
    return running_loss

def test(model, test_loader, criterion):
    loss = 0
    tot = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model.forward(inputs)
            loss += criterion(outputs, labels)
            tot += 1

    avg_loss = loss / tot

    return avg_loss


def train_and_save_networks(args):

    dataset = GymDynamicsDataset(args["env_name"])
    
    data, label = dataset[0]
    
    model = FC(args["num_layer"], len(data), args["hidden_dim"], len(label))

    train(dataset, model, args["prefix"], args["epochs"], args["lr"], load_file=args["load_path"])
    
    convert(args["prefix"], "epoch_1000", args["num_layer"], len(data), args["hidden_dim"], len(label))

    # convert(args["prefix"], "epoch_400", args["num_layer"], len(data), args["hidden_dim"], len(label))

if __name__ == "__main__":

    # ant_args = {
    #     "env_name": "Free-Ant-v0",
    #     "num_layer": 2,
    #     "hidden_dim": 500,
    #     "epochs": 210,
    #     "lr": 0.01,
    #     "prefix": 'ant-FC2-100',
    # }

    # unicycle_args_3 = {
    #     "env_name": "Unicycle-v0",
    #     "num_layer": 3,
    #     "hidden_dim": 100,
    #     "epochs": 1005,
    #     "lr": 0.001,
    #     "prefix": 'unicycle-FC3-100-rk4-so',
    #     "load_path": "../model/unicycle-FC3-100-rk4/epoch_1000.pth"
    # }

    unicycle_args_5 = {
        "env_name": "Unicycle-v0",
        "num_layer": 5,
        "hidden_dim": 100,
        "epochs": 1005,
        "lr": 0.001,
        "prefix": 'unicycle-FC5-100-rk4-so',
        "load_path": None
    }

    unicycle_args_4 = {
        "env_name": "Unicycle-v0",
        "num_layer": 4,
        "hidden_dim": 50,
        "epochs": 1005,
        "lr": 0.001,
        "prefix": 'unicycle-FC4-50-rk4-so',
        "load_path": None
    }

    unicycle_args_35 = {
        "env_name": "Unicycle-v0",
        "num_layer": 3,
        "hidden_dim": 50,
        "epochs": 1005,
        "lr": 0.001,
        "prefix": 'unicycle-FC3-50-rk4-so',
        "load_path": None
    }
    train_and_save_networks(unicycle_args_35)