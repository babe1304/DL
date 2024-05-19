import numpy as np
import data
import matplotlib.pyplot as plt
import time

def logreg_train(X, Y_):
    '''
    Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
      w, b: parametri logističke regresije
    '''
    c = max(Y_) + 1
    Y_ = data.class_to_onehot(Y_)
    N, D = X.shape
    
    W = np.random.randn(D, c)
    b = 0
    param_niter = 1000
    param_delta = 0.75
    
    for i in range(param_niter):
        scores = np.dot(X, W) + b
        probs = softmax(scores)
        loss = - np.sum(Y_ * np.log(probs + 1e-20)) / N

        if i % 10 == 0:
            print("Iteration {}: loss {}".format(i, loss))
        
        dl_ds = probs - Y_
        dl_dw = np.dot(X.T, dl_ds) / N
        dl_db = np.sum(dl_ds, axis=0) / N

        W += -param_delta * dl_dw
        b += -param_delta * dl_db
    return W, b

def logreg_classify(X, W, b):
    '''
    Argumenti
      X:  podatci, np.array NxD
      W, b: parametri logističke regresije

    Povratne vrijednosti
      probs: vjerojatnosti razreda c1, np.array Nx1
    '''
    scores = np.dot(X, W) + b
    probs = softmax(scores)
    return probs

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)

def logreg_decfun(w, b):
    return lambda X: np.argmax(logreg_classify(X, w, b), axis=1)

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(4, 100)

    # train the model
    t1 = time.time()
    W, b = logreg_train(X, Y_)
    t2 = time.time()
    print(f'Training time: {t2 - t1}s')

    # evaluate the model
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    print(f'Accuracy: {accuracy}\nPrecision: {pr}\nConfusion matrix:\n{M}')

    # graph the decision surface
    decfun = logreg_decfun(W, b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.show()