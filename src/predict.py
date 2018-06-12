import torch
from torch import nn
from model import Model

model = torch.load('model/model.pt')

input = torch.tensor([[
    0, 200, 78, 12, 11, 30, .07, 22
]], dtype = torch.float)


prediction = model(input)
 
print(prediction[0])