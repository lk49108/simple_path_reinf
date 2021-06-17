import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange, tqdm

def train_network(model, loss_fn, optimizer, train_loader,
                  num_epochs=20, pbar_update_interval=200, print_logs=False):

    def make_train_step(model, loss_fn, optimizer):
        def train_step(x, y):
            model.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

        return train_step

    train_step = make_train_step(model, loss_fn, optimizer)

    pbar = trange(num_epochs) if print_logs else range(num_epochs)

    for i in pbar:
        for k, batch_data in enumerate(train_loader):
            batch_x = batch_data[:, :-1]
            batch_y = batch_data[:, -1].long()

            train_step(batch_x, batch_y)

            if print_logs and k % pbar_update_interval == 0:
                pass
                # acc = (y_pred.round() == batch_y).sum().float()/(len(batch_y))
                # pbar.set_postfix(loss=loss.item(), acc=acc.item())

    if print_logs:
        pbar.close()
