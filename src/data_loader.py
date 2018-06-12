import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
csvpath = 'data/diabetes.csv'


class getData(Dataset):
    def __init__(self, transform=None):

        data = np.loadtxt(csvpath, delimiter=',', dtype=np.float, skiprows=1)
        self.len = data.shape[0]
        self.x = torch.tensor(data[:, 0:-1], dtype=torch.float) 
        self.y = torch.tensor(data[:, [-1]], dtype=torch.float)  

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
