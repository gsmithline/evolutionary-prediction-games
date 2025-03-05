import numpy as np
import cvxpy as cp

__all__ = [
    'cp_hinge',
    'cp_logloss',
    'ConvexLinearClassifier',
    'TwoDimensionalEnumerationLinearClassifier',
    'OneDimensionalHardSVM',
]

cp_hinge = lambda y_hat, y: cp.pos(1 - cp.multiply(y, y_hat))
cp_logloss = lambda y_hat, y: cp.logistic(-cp.multiply(y, y_hat))
inf = 1e6


class LinearClassifier:
    def fit(self, X, y, sample_weights=1, random_state=None):
        raise NotImplementedError

    def decision_function(self, X):
        return X@self.w - self.b

    def predict(self, X):
        return (np.sign(self.decision_function(X))>0).astype(int).ravel()

    def score(self, X, y, sample_weights=1):
        n,d = X.shape
        sample_weights_vec = np.ones(n)*sample_weights
        sample_weights_vec /= sample_weights_vec.sum()
        return (y==self.predict(X))@sample_weights_vec

    def fit_p(self, X, P):
        n,d = X.shape
        return self.fit(
            X=np.vstack([X,X]),
            y=np.hstack([np.zeros(n),np.ones(n)]),
            sample_weights=np.hstack([P[0],P[1]]),
        )

    def score_p(self, X, P):
        n,d = X.shape
        return self.score(
            X=np.vstack([X,X]),
            y=np.hstack([np.zeros(n),np.ones(n)]),
            sample_weights=np.hstack([P[0],P[1]]),
        )

    @property
    def coef_(self):
        return np.array([self.w])

    @property
    def intercept_(self):
        return self.b


class ConvexLinearClassifier(LinearClassifier):
    LOSS_F = {
        'hinge': cp_hinge,
        'logistic': cp_logloss,
    }
    
    PENALTY_P = {
        'l1': 1,
        'l2': 2,
        'l_inf': 'inf',
    }

    SOLVER_SETTINGS = {
        'hinge': {
            'solver': cp.CLARABEL,
        },
        'logistic': {
            'solver': cp.CLARABEL,
            'max_iter': 10000,
        },
    }
    
    def __init__(self, loss, penalty, alpha, fit_intercept):
        self.loss_f = self.LOSS_F[loss]
        self.solver_settings = self.SOLVER_SETTINGS.get(loss,dict())
        self.regularization_p = self.PENALTY_P[penalty]
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.w = None
        self.b = None

    def fit(self, X, y, sample_weights=1):
        n,d = X.shape
        y = np.atleast_2d(y).T*2-1
        assert np.isin(y.ravel(),[-1,1]).all()
        w = cp.Variable((d,1))
        b = cp.Variable() if self.fit_intercept else 0
        y_hat = X@w - b
        sample_weights_vec = np.ones(n)*sample_weights
        sample_weights_vec /= sample_weights_vec.sum()
        loss = sample_weights_vec@self.loss_f(y_hat,y)
        reg = cp.norm(w, p=self.regularization_p)
        obj = loss + self.alpha*reg
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve(**self.solver_settings)
        self.prob = prob
        self.w = w.value
        self.b = b.value if type(b) is not int else b
        return self



class TwoDimensionalEnumerationLinearClassifier(LinearClassifier):
    def __init__(self, grid_n):
        self.grid_n = grid_n
        self.theta_grid = np.linspace(0,2*np.pi,grid_n)
        self.w_grid = np.vstack([np.cos(self.theta_grid), np.sin(self.theta_grid)])
        self.b = 0

    def fit(self, X, y, sample_weights=1):
        n = len(y)
        sample_weights_vec = np.ones(n)*sample_weights
        sample_weights_vec /= sample_weights_vec.sum()
        y_hat = np.sign(X@self.w_grid)==1
        y_tile = np.tile(y,(len(self.theta_grid),1)).T
        self.acc_grid = sample_weights_vec@(y_hat==y_tile)
        self.i = self.acc_grid.argmax()
        self.w = self.w_grid[:,self.i]
        self.theta = self.theta_grid[self.i]
        return self


class OneDimensionalHardSVM:
    def fit(self, X, y):
        if (y==0).all():
            self.margin=inf
        elif (y==1).all():
            self.margin=-inf
        else:
            xneg = X[y==0].max()
            xpos = X[y==1].min()
            assert xneg<=xpos
            self.margin = (xneg+xpos)/2

    def predict(self, X):
        return (X>=self.margin).ravel()

    def score(self,X,y,w=None):
        if w is None:
            w = np.ones_like(y)
            w /= w.sum()
        return (self.predict(X)==y)@w