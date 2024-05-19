import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import matplotlib.pyplot as plt
from itertools import product
import time

class PTDeep(nn.Module):
  def __init__(self, dims=[2, 3, 2], activation=nn.ReLU()):
    """Arguments:
       - dims: dimensions of the network
    """
    super(PTDeep, self).__init__()

    self.activation = activation
    self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

  def forward(self, X):
    for layer in self.layers[:-1]:
        X = self.activation(layer(X))
    return torch.softmax(self.layers[-1](X), dim=1)

  def get_loss(self, X, Yoh_):
    probs = self.forward(X)
    return -torch.sum(Yoh_ * torch.log(probs + 1e-20), dim=1).mean()
  
  def count_params(self):
    sum_params = 0
    for p in self.named_parameters():
      print(f'Layer: {p[0]}, dims: {p[1].shape}')
      sum_params += p[1].numel()
    print(f'Total number of parameters: {sum_params}')

class PTDeep2(nn.Module):
    def __init__(self, dims=[2, 3, 2], activation=nn.ReLU()):
        """
        Arguments:
        - dims: dimensions of the network
        """
        super(PTDeep2, self).__init__()

        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.gammas = nn.ParameterList([nn.Parameter(torch.ones(dims[i]), requires_grad=True) for i in range(len(dims))])
        self.betas = nn.ParameterList([nn.Parameter(torch.zeros(dims[i]), requires_grad=True) for i in range(len(dims))])

    def forward(self, X):
        if self.train:
            X = (X - X.mean(dim=0)) / X.std(dim=0)
        X = self.gammas[0] * X + self.betas[0]

        for i, layer in enumerate(self.layers[:-1]):
            X = self.activation(layer(X))

            if self.train:
                X = (X - X.mean(dim=0)) / X.std(dim=0)
            X = self.gammas[i+1] * X + self.betas[i+1]
        return torch.softmax(self.layers[-1](X), dim=1)

    def get_loss(self, X, Yoh_):
        probs = self.forward(X)
        return -torch.sum(Yoh_ * torch.log(probs + 1e-20), dim=1).mean()
    
    def count_params(self):
        sum_params = 0
        for p in self.named_parameters():
            print(f'Layer: {p[0]}, dims: {p[1].shape}')
            sum_params += p[1].numel()
            print(f'Total number of parameters: {sum_params}')

def train(model, train_dataloader, param_niter=1e5, param_delta=0.1, param_lambda=0.5):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
        - param_lambda: L2 regularization parameter
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

    for i in range(int(param_niter) + 1):
        for X, Yoh_ in train_dataloader:
            loss = model.get_loss(X, Yoh_)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if i % 100 == 0:
            print(f'Iteration {i}: loss {loss}')
    return

def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    model.eval()
    X = torch.Tensor(X)
    return model.forward(X).detach().numpy()

def pt_deep_decfun(model):
    return lambda X: np.argmax(eval(model, X), axis=1)

if __name__ == "__main__":
    np.random.seed(100)
    hiper_params = {'niter': [5e3], 'delta': [0.1], 'lambda': [1e-4]}

    # X, Y = data.sample_gauss_2d(3, 100)
    X, Y = data.sample_gmm_2d(6, 2, 10)
    Yoh_ = data.class_to_onehot(Y)
    X, Yoh_ = torch.Tensor(X), torch.Tensor(Yoh_)
    train_data = torch.utils.data.TensorDataset(X, Yoh_)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32) 
    accs = {}

    for niter, delta, lambd in product(hiper_params['niter'], hiper_params['delta'], hiper_params['lambda']):
        ptdeep = PTDeep2([X.shape[1], 10, 10, Yoh_.shape[1]], activation=nn.Tanh())
        ptdeep.count_params()
        print(f'Hyperparameters: niter: {niter}, delta: {delta}, lambda: {lambd}')
        t1 = time.time()
        train(ptdeep, train_dataloader, niter, delta, lambd)
        t2 = time.time()
        probs = eval(ptdeep, X)

        # ispiši performansu (preciznost i odziv po razredima)
        Y_ = np.argmax(probs, axis=1)
        acc, pr, M, f1_macro = data.eval_perf_multi(Y_, Y)
        accs[(niter, delta, lambd)] = (acc, f1_macro, t2-t1)
        print(f'Accuracy: {acc}, F1 score: {f1_macro}, Precision: {pr}, Confusion matrix: {M}\n====================================')

        # iscrtaj rezultate, decizijsku plohu
        X_ = X.detach().numpy()
        decfun = pt_deep_decfun(ptdeep)
        bbox=(np.min(X_, axis=0), np.max(X_, axis=0))
        data.graph_surface(decfun, bbox, offset=0.5)
        data.graph_data(X_, Y_, Y)
        plt.show()

    # Plotting accuracy from accs
    for key, val in accs.items():
        print(f'Hyperparameters: {key}, Accuracy: {val[0]}, F1 macro: {val[1]}, Training time: {val[2]}s')