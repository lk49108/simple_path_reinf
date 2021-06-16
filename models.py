import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class DenseNet(nn.Module):
    def __init__(self, widths):
        super().__init__()

        num_layers = len(widths)
        layers = [[nn.Linear(widths[i], widths[i+1]), nn.ReLU()] for i in range(num_layers-2)]
        self.layers = [nn.Flatten(1, -1),
                      *list(itertools.chain(*layers)),
                      nn.Linear(widths[-2], widths[-1])]

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        prob = self.net(x)
        return prob

if __name__=='__main__':
    net=DenseNet([2,3,1])
    print(net.state_dict())
    d=torch.tensor([1,10]).float()
    f=nn.Softmax(dim=0)
    print(f(d))
