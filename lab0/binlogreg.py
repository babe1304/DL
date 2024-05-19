import numpy as np
import data
import matplotlib.pyplot as plt
import time

def binlogreg_train(X, Y_):
    '''
    Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
      w, b: parametri logističke regresije
    '''
    N, D = X.shape
    W = np.random.randn(D, 1)
    b = 0
    param_niter = 1000
    param_delta = 0.1

    for i in range(param_niter):
        scores = np.dot(X, W) + b
        probs = (np.exp(scores) / (1 + np.exp(scores))).reshape(-1)
        loss = - np.sum(Y_ * np.log(probs + 1e-20) + (1 - Y_) * np.log(1 - probs + 1e-20)) / N

        if i % 10 == 0:
            print("Iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - Y_
        dL_dW = np.dot(X.T, dL_dscores).reshape(-1, 1) / N
        dL_db = np.sum(dL_dscores) / N

        W += -param_delta * dL_dW
        b += -param_delta * dL_db

    return W, b

def binlogreg_decfun(w, b):
    return lambda X: binlogreg_classify(X, w, b)

def binlogreg_classify(X, W, b):
    '''
    Argumenti
      X:  podatci, np.array NxD
      W, b: parametri logističke regresije

    Povratne vrijednosti
      probs: vjerojatnosti razreda c1, np.array Nx1
    '''
    scores = np.dot(X, W) + b
    probs = np.exp(scores) / (1 + np.exp(scores))
    return probs.reshape(-1)

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    t1 = time.time()
    w,b = binlogreg_train(X, Y_)
    t2 = time.time()

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    Y = probs > 0.5

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(f'Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nAP: {AP}')
    print(f'Time: {t2 - t1}s')

    # graph the decision surface
    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()