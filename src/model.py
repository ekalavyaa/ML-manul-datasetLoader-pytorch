
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 4)
        self.l2 = nn.Linear(4, 2)
        self.l3 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out1 = self.sigmoid(self.l1(input))
        out2 = self.sigmoid(self.l2(out1))
        out3 = self.sigmoid(self.l3(out2))
        return out3

