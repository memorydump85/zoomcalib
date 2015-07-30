import numpy as np
from numpy.linalg import solve, slogdet
from scipy import optimize

import pyximport; pyximport.install()
from gram_matrix import *



#--------------------------------------
class GaussianProcess(object):
#--------------------------------------
    
    def __init__(self, train_x, train_t, covf):
        self.train_x = train_x
        self.train_t = train_t
        self.covf = covf
        self.C = None
        self.Cinvt = None

        
    def ensure_gram_matrix(self):
        if self.C is not None:
            return
        
        self.C = self.covf.compute_gram_matrix(self.train_x)
        self.Cinvt = solve(self.C, self.train_t)


    def predict(self, query, cov=False):
        return self.predict_(query) if cov else self.predict_mean_(query)
    
    
    def predict_mean_(self, query):
        N = len(self.train_x)
        M = len(query)
        
        data = np.concatenate((self.train_x, query))

        # TODO: compute only relevant parts of A
        A = self.covf.compute_gram_matrix(data)
        Kt = A[N:,:N]
        
        self.ensure_gram_matrix()
        q_mean = Kt.dot(self.Cinvt) 
        
        return q_mean

    
    def predict_(self, query):
        N = len(self.train_x)
        M = len(query)
        
        data = np.concatenate((self.train_x, query))

        # TODO: compute only relevant parts of A
        A = self.covf.compute_gram_matrix(data)
        Kt = A[N:,:N]
        Cq = A[N:,N:]
           
        self.ensure_gram_matrix()
        q_mean = Kt.dot(self.Cinvt)
        q_covf = Cq - Kt.dot(solve(self.C, Kt.T))
        
        return (q_mean, q_covf)
    
    
    def model_evidence(self):
        self.ensure_gram_matrix()
        t = self.train_t

        datafit = t.T.dot(self.Cinvt)
        s, logdet = slogdet(self.C)
        complexity = s*logdet
        nomalization = len(t)*np.log(np.pi*2)
        
        return -0.5 * (datafit + complexity + nomalization)


    @classmethod
    def fit(cls, x, t, covf, theta0):
        evidence = lambda theta: \
            -cls(x, t, covf(theta)).model_evidence()
        theta_opt = optimize.fmin_powell(func=evidence, x0=theta0, xtol=0.001, ftol=0.001, disp=False)
        
        if isinstance(theta_opt.tolist(), float):
            theta_opt = [theta_opt.tolist()]
        return cls(x, t, covf(theta_opt))


#--------------------------------------
class sqexp1D_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta
    
    def compute_gram_matrix(self, data):
        return gram_matrix_sq_exp_1D(data, *self.theta)


#--------------------------------------
class sqexp2D_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta
    
    def compute_gram_matrix(self, data):
        return gram_matrix_sq_exp_2D(data, *self.theta)
