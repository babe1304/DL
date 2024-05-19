import torch, torchvision, torch.nn as nn, torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import data

def batch(X, Y, batch_size=32):
    batches = []
    batch = []
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                batches.append([X[batch], Y[batch]])
                batch = []
    if batch:
        batches.append([X[batch], Y[batch]])
    return batches

def train(model, train_dataset, val_dataset, param_niter=100, param_delta=1e-3, param_lambda=0.1):
    """
    Arguments:
        - dataloader: type: torch.utils.data.DataLoader
        - param_niter: number of training iterations
        - param_delta: learning rate
        - param_lambda: L2 regularization parameter
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)
    train_loss = []
    X, Y = train_dataset.tensors
    X_val, Y_val = val_dataset.tensors
    val_loss = []

    for _ in range(int(param_niter) + 1):
        batches = batch(X, Y, batch_size=32)
        with tqdm(batches, unit='batch') as t:
            for X_batch, Y_batch in t:
                loss = F.cross_entropy(model(X_batch), Y_batch)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss=loss.item())
        scheduler.step()
        loss = F.cross_entropy(model(X_val), Y_val)
        val_loss.append(loss.item())
    return train_loss, val_loss

def eval(model, X_test, Y_test):
    Y_ = model(X_test)
    loss = F.cross_entropy(Y_, Y_test)
    acc, pr, M, f1 = data.eval_perf_multi(np.argmax(Y_.detach().numpy(), axis=1), Y_test)
    return loss.item(), acc, pr, M, f1
  

def create_FCModel(dims=[2, 3, 2], activation=nn.ReLU()):
    return nn.Sequential(
        nn.Flatten(),
        *[nn.Sequential(nn.Linear(dims[i], dims[i+1]), activation) for i in range(len(dims)-2)],
        nn.Linear(dims[-2], dims[-1]),
        nn.Softmax(dim=1)
    )

if __name__ == '__main__':
    dataset_root = '/tmp/mnist'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True, transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True, transform=torchvision.transforms.ToTensor())

    n = len(mnist_train)
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [int(n * 0.8), n - int(n * 0.8)])

    x_train, y_train = mnist_train.dataset.train_data, mnist_train.dataset.train_labels
    x_test, y_test = mnist_test.test_data, mnist_test.test_labels
    x_val, y_val = mnist_val.dataset.train_data, mnist_val.dataset.train_labels
    x_train, x_test, x_val = x_train.float().div_(255.0), x_test.float().div_(255.0), x_val.float().div_(255.0)

    D = x_train.shape[1] * x_train.shape[2]
    C = y_train.max().item() + 1

    model = create_FCModel([D, C], activation=nn.ReLU())
    train_data = torch.utils.data.TensorDataset(x_train.view(-1, D), y_train)
    val_data = torch.utils.data.TensorDataset(x_val.view(-1, D), y_val)

    train_loss, val_loss = train(model, train_data, val_data, param_niter=15, param_delta=1e-4, param_lambda=1e-5)

    weights = model[1].weight.detach().numpy()
    for i in range(10):
        plt.imshow(weights[i].reshape(28, 28))
        plt.show()

    eval_loss, eval_acc, eval_pr, eval_M, eval_f1 = eval(model, x_test, y_test)
    print(f'Accuracy: {eval_acc}\nEvaluation loss: {eval_loss},\nPrecision:\n{eval_pr},\nConfusion Matrix:\n{eval_M},\nF1: {eval_f1}')

    plt.plot(train_loss, color='blue', label='Training loss')
    plt.plot([i*len(batch(x_train, y_train)) for i in range(16)], val_loss, color='red', label='Validation loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.show()

    random_indices = np.random.choice(len(x_test), 10)
    x_sample = x_test[random_indices]
    y_sample = y_test[random_indices]
    y_pred = model(x_sample)
    y_pred = torch.argmax(y_pred, dim=1)
    for i in range(10):
        print(f'Predicted: {y_pred[i]}, Actual: {y_sample[i]}')
        plt.imshow(x_sample[i].detach().numpy(), cmap='gray')
        plt.show()

