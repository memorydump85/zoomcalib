import numpy as np
from numpy.linalg import solve, slogdet
from scipy import optimize



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
        
        N = len(self.train_x)
        data = self.train_x
        self.C = np.array([
                    self.covf(data[i], data[j], colocated=(i==j))
                    for i in xrange(0, N) for j in xrange (0, N)]
                         ).reshape(N,N)
        self.Cinvt = solve(self.C, self.train_t)


    def predict(self, query, cov=False):
        return self.predict_(query) if cov else self.predict_mean_(query)
    
    
    def predict_mean_(self, query):
        N = len(self.train_x)
        M = len(query)
               
        indices = [(i,j) for i in xrange(N, N+M) for j in xrange(0, N)]
        
        data = np.concatenate((self.train_x, query))
        covf_at = lambda i, j: self.covf(data[i], data[j], colocated=(i==j))

        # bottom rows of gram_matrix of `data`
        A = np.zeros((M, N))
        for i, j in indices:
            A[((i-N), j)] = covf_at(i, j)
            
        Kt = A[0:M,0:N]
            
        self.ensure_gram_matrix()
        q_mean = Kt.dot(self.Cinvt) 
        
        return q_mean

    
    def predict_(self, query):
        N = len(self.train_x)
        M = len(query)
               
        indices = [(i,j) for i in xrange(N, N+M) for j in xrange(0, N+M)]
        
        data = np.concatenate((self.train_x, query))
        covf_at = lambda i, j: self.covf(data[i], data[j], colocated=(i==j))

        # bottom rows of gram_matrix of `data`
        A = np.zeros((M, N+M))
        for i, j in indices:
            A[((i-N), j)] = covf_at(i, j)
            
        Kt = A[0:M,0:N]
        Cq = A[0:M,N:N+M] 
            
        self.ensure_gram_matrix()        
        q_mean = Kt.dot(self.Cinvt) 
        q_covf = Cq - Kt.dot(solve(self.C, Kt.T))
        
        return (q_mean, q_covf)
    
    
    def model_evidence(self):
        self.ensure_gram_matrix()
        t = self.train_t

        datafit = t.T.dot(solve(self.C, t))
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
        self.fsig, self.sig, self.noise_prec = theta
    
    def __call__(self, a, b, colocated):    
        z = a - b
        v = (self.fsig*self.fsig)* np.exp(-0.5*(z*z)/(self.sig*self.sig))
        return v + 1./(self.noise_prec*self.noise_prec) if colocated else v 


#--------------------------------------
def poly1D_covariancef(degree):
#--------------------------------------
    class poly1D_covf_impl(object):

        def __init__(self, theta):
            self.degree = degree
            self.var0 = theta[0]**2
        
        def _eval(self, a, b):
            return np.dot(a, b + self.var0)**self.degree

        def __call__(self, a, b, colocated):
            Kab = self._eval(a, b)
            Ka = np.sqrt(self._eval(a, a))
            Kb = np.sqrt(self._eval(b, b))
            return Kab / (Ka*Kb)

    return poly1D_covf_impl


#--------------------------------------
class sqexp2D_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta
        self.fsig, sig00, sig11, var10, self.noise_prec = theta
        Sigma = np.array([[sig00**2, var10], [var10, sig11**2]])
        self.Sigma_inv = np.linalg.inv(Sigma).flatten()
    
    def __call__(self, a, b, colocated):
        # chi2 = zT * Sigma_inv * z
        g, h = a - b
        p, q, r, s = self.Sigma_inv
        chi2 = g*(p*g+q*h) + h*(r*g+s*h)
    
        nu = self.fsig
        beta = self.noise_prec

        v = (nu*nu)*np.exp(-0.5*chi2)
        return v + 1./(beta*beta) if colocated else v
