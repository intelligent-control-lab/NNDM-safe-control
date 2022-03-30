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
from train_nn_dynamics import train_and_save_networks
from scipy.stats import norm
import matplotlib.pyplot as plt

def test_ensemble(models, test_loader, criterion):
    loss = 0
    tot = 0
    results = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = torch.zeros((len(models), *labels.shape))
            for i in range(len(models)):
                outputs[i] = models[i].forward(inputs)
                # loss += criterion(outputs, labels)
            
            var, mean = torch.var_mean(outputs, axis=0)

            var = torch.mean(var, axis=1) # mean of dot_x dims.
            var = torch.sqrt(var)
            err = torch.norm(mean - labels, dim=1)
            # print(err.shape)
            # print(np.vstack((var, err)).shape)
            results.append(np.vstack((var, err)))
    results = np.array(results)
    results = np.hstack(results)
    print(np.shape(results))
    
    plt.figure()
    plt.scatter(results[0,:], results[1,:])
    plt.xlabel("std")
    plt.ylabel("error")
    plt.title("num ensemble: " +str(len(models)))
    plt.show()

def load_test_ensemble(env_name, n_ensemble, num_layer, hidden_dim, prefix):
    dataset = GymDynamicsDataset(env_name)
    data, label = dataset[0]
    models = [FC(num_layer, len(data), hidden_dim, len(label)) for i in range(n_ensemble)]
    for i in range(n_ensemble):
        load_path = "../model/"+prefix+str(i)+"/epoch_500.pth"
        models[i].load_state_dict(torch.load(load_path))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)
    criterion = nn.MSELoss()
    test_ensemble(models, test_loader, criterion)

def train_model_ensemble():
    
    n_ensemble = 10
    layer = 3
    hidden = 100
    for i in range(n_ensemble):
        unicycle_args = {
            "env_name": "Unicycle-v0",
            "n_ensemble": 10,
            "num_layer": layer,
            "hidden_dim": hidden,
            "epochs": 1005,
            "lr": 0.001,
            "prefix": 'unicycle-ensemble-FC'+str(layer)+'-'+str(hidden)+'-'+str(i),
            "load_path": None
        }

        train_and_save_networks(unicycle_args)
    
if __name__ == "__main__":

    # load_test_ensemble("Unicycle-v0", 9, 3, 100, 'unicycle-ensemble-FC3-100-')
    load_test_ensemble("Unicycle-v0", 9, 3, 100, 'unicycle-ensemble-FC3-100-')