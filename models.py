import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from concurrent.futures import ThreadPoolExecutor

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
    def wait_on_future():
        f = executor.submit(pow, 5, 2)
        # This will never complete because there is only one worker thread and
        # it is executing this function.
        print(f.result())


    executor = ThreadPoolExecutor(max_workers=2)
    executor.submit(wait_on_future)
