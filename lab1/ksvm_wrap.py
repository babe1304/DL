import sklearn.svm as svm
import data
import numpy as np
import matplotlib.pyplot as plt

class KSVMWrap:
    '''
    Metode:
    __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre

    predict(self, X)
        Predviđa i vraća indekse razreda podataka X

    get_scores(self, X):
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.

    support
        Indeksi podataka koji su odabrani za potporne vektore
    '''
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.model = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.model.fit(X, Y_)
        self.support = self.model.support_

    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self, X):
        return self.model.decision_function(X)
    
    def get_support(self):
        return self.support
    
if __name__=="__main__":
    np.random.seed(100)
    
    # get the training dataset
    # X,Y_ = data.sample_gauss_2d(4, 100)
    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    
    # train the model
    model = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')

    acc, pr, M, f1 = data.eval_perf_multi(model.predict(X), Y_)
    print(f'Accuracy: {acc}, Precision: {pr}, Confusion matrix: {M}, F1 score: {f1}')
    
    # graph the decision surface
    decfun = model.get_scores
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0)
    data.graph_data(X, Y_, model.predict(X), special=model.get_support())
    plt.show()