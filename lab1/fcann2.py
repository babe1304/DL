import numpy as np
import data
import matplotlib.pyplot as plt
import time

def RELU(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)

class FCANN2:
    def __init__(self, output_dim=2, input_dim=2, hidden_dim=3):
        self.layer_1 = np.random.normal(scale=1/np.mean([input_dim, hidden_dim]), size=(input_dim, hidden_dim))
        self.bias_1 = np.zeros(hidden_dim)
        self.layer_2 = np.random.normal(scale=1/np.mean([hidden_dim, output_dim]), size=(hidden_dim, output_dim))
        self.bias_2 = np.zeros(output_dim)

    def forward(self, X):
        self.z1 = np.dot(X, self.layer_1) + self.bias_1
        self.a1 = RELU(self.z1)
        self.z2 = np.dot(self.a1, self.layer_2) + self.bias_2
        self.a2 = softmax(self.z2)
        return self.a2

def fcann2_train(X, Y_, hidden_dim=3, param_niter=1000, param_delta=0.5):
    N, D = X.shape
    c = max(Y_) + 1
    Y_ = data.class_to_onehot(Y_)
    model = FCANN2(output_dim=c, input_dim=D, hidden_dim=hidden_dim)

    for i in range(int(param_niter)):
        probs = model.forward(X)
        loss = - np.sum(Y_ * np.log(probs + 1e-20)) / N
        if i % 100 == 0:
            print("Iteration {}: loss {}".format(i, loss))

        dl_ds = probs - Y_
        dl_dw2 = np.dot(model.a1.T, dl_ds) / N
        dl_db2 = np.sum(dl_ds, axis=0) / N

        dl_da1 = np.dot(dl_ds, model.layer_2.T)
        dl_dz1 = dl_da1 * (model.z1 > 0)
        dl_dw1 = np.dot(X.T, dl_dz1) / N
        dl_db1 = np.sum(dl_dz1, axis=0) / N

        model.layer_1 += -param_delta * dl_dw1
        model.bias_1 += -param_delta * dl_db1
        model.layer_2 += -param_delta * dl_dw2
        model.bias_2 += -param_delta * dl_db2
    return model

def fcann2_classify(X, model):
    return np.argmax(model.forward(X), axis=1)

def fcann2_decfun(model):
    return lambda X: fcann2_classify(X, model)

if __name__=="__main__":
    np.random.seed(100)
    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    t1 = time.time()
    model = fcann2_train(X, Y_, param_niter=10000, param_delta=0.05, hidden_dim=5)
    t2 = time.time()
    print(f'Training time: {t2-t1}')
    Y = fcann2_classify(X, model)
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    print(f'Accuracy: {accuracy}\nPrecision: {pr}\nConfusion matrix:\n{M}')

    decfun = fcann2_decfun(model)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.show()