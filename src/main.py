from torch import nn
import torch.optim as opt
from torch.nn import functional as F
from data_loader import getData
from torch.utils.data import DataLoader
import torch
from model import Model

data = getData()
""" geting batch data """

train_data = DataLoader(data, batch_size=6,
                        shuffle=True, num_workers=2)



model = Model()
lossFun = nn.MSELoss(size_average=True)
optimizer = opt.Adagrad(model.parameters(), lr=.01)

for i in range(2):
    for i_batch, sample_batched in enumerate(train_data):
        optimizer.zero_grad()
        input, labels = sample_batched
        print("****************** batch =  ", i_batch,
              "*****************************")
        y_predic = model(input=input)
        loss = lossFun(y_predic, labels)
        print("loss= ", loss.item())
        loss.backward()
        optimizer.step()


torch.save(model, 'model/model.pt')
