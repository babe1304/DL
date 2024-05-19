import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import matplotlib.pyplot as plt
from itertools import product
import time

class PTLogreg(nn.Module):
  def __init__(self, D, C):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """
    super(PTLogreg, self).__init__()
    self.W = nn.Parameter(torch.randn(D, C), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(C), requires_grad=True)

  def forward(self, X):
    s1 = torch.mm(X, self.W) + self.b
    return torch.softmax(s1, dim=1)

  def get_loss(self, X, Yoh_):
    probs = self.forward(X)
    return -torch.sum(Yoh_ * torch.log(probs + 1e-20), dim=1).mean()


def train(model, X, Yoh_, param_niter=1e5, param_delta=0.1, param_lambda=0.1):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
        - param_lambda: L2 regularization parameter
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    for i in range(int(param_niter) + 1):
        loss = model.get_loss(X, Yoh_) + param_lambda * torch.norm(model.W)
        loss.backward()
        if i % 100 == 0:
            print(f'Iteration {i}: loss {loss}, gradient W: {model.W.grad}, gradient b: {model.b.grad}')
        optimizer.step()
        optimizer.zero_grad()
    return

def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    X = torch.Tensor(X)
    return model.forward(X).detach().numpy()

def pt_logreg_decfun(model):
    return lambda X: np.argmax(eval(model, X), axis=1)

if __name__ == "__main__":
    np.random.seed(100)
    hiper_params = {'niter': [10000], 'delta': [0.05], 'lambda': [0]}

    X, Y = data.sample_gauss_2d(3, 100)
    Yoh_ = data.class_to_onehot(Y)
    X, Yoh_ = torch.Tensor(X), torch.Tensor(Yoh_)

    accs = {}

    for niter, delta, lambd in product(hiper_params['niter'], hiper_params['delta'], hiper_params['lambda']):
        ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])
        print(f'Hyperparameters: niter: {niter}, delta: {delta}, lambda: {lambd}')
        t1 = time.time()
        train(ptlr, X, Yoh_, niter, delta, lambd)
        t2 = time.time()
        probs = eval(ptlr, X)

        # ispiši performansu (preciznost i odziv po razredima)
        Y_ = np.argmax(probs, axis=1)
        acc, _, _, f1_macro = data.eval_perf_multi(Y_, Y)
        accs[(niter, delta, lambd)] = (acc, f1_macro, t2-t1)
        print(f'Accuracy: {acc}, F1 score: {f1_macro}\n====================================')

        # iscrtaj rezultate, decizijsku plohu
        X_ = X.detach().numpy()
        decfun = pt_logreg_decfun(ptlr)
        bbox=(np.min(X_, axis=0), np.max(X_, axis=0))
        data.graph_surface(decfun, bbox, offset=0.5)
        data.graph_data(X_, Y_, Y)
        plt.show()

    # Plotting accuracy from accs
    for key, val in accs.items():
        print(f'Hyperparameters: {key}, Accuracy: {val[0]}, F1 macro: {val[1]}, Training time: {val[2]}s')