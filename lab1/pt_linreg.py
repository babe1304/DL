import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def generate_data(a=1, b=1, n=100, sigma=0.3):
    X = torch.randn(n)
    Y = a * X + b + sigma * torch.randn(n)
    return X, Y

def pt_linreg(X, Y, lr=0.1, niter=100):
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    optimizer = optim.SGD([a, b], lr=lr)

    for i in range(niter + 1):
        Y_ = a*X + b

        diff = (Y-Y_)
        loss = torch.mean(diff**2)
        loss.backward()

        if i % 10 == 0:
            grad_a = torch.mean(-2 * diff * X)
            grad_b = torch.mean(-2 * diff)
            print(f'Analitički grad_a: {grad_a}, grad_b: {grad_b}')
            print(f'PyTorch grad_a: {a.grad}, grad_b: {b.grad}')
            print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')

        optimizer.step()
        optimizer.zero_grad()

    return a.detach().numpy(), b.detach().numpy()

def plot_data(X, Y, a, b):
    plt.plot(X, Y, 'o', label='$(x^{(i)},y^{(i)})$')
    plt.plot(X, a*X+b, label='$\\hat{y} = ax + b$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, Y = generate_data(a=-3, b=3, n=100, sigma=0.5)
    a, b = pt_linreg(X, Y, lr=0.1, niter=100)
    print(f'a: {a}, b: {b}')
    plot_data(X.detach().numpy(), Y.detach().numpy(), a, b)