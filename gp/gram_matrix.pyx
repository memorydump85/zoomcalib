import numpy as np
import cython

cimport numpy as np
from libc.math cimport exp


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gram_matrix_sq_exp_2D(
    np.ndarray[np.float_t, ndim=2] data,
    float sigma_f,
    float sigma_xx,
    float sigma_yy,
    float corr_xy,
    float sigma_inv_noise ):

    cdef np.ndarray[np.float_t, ndim=2] Sigma, Sigma_inv
    Sigma = np.array([ [ sigma_xx**2,  corr_xy    ],
                       [  corr_xy,    sigma_yy**2 ] ])
    Sigma_inv = np.linalg.inv(Sigma)

    cdef float p, q, r, s
    p = Sigma_inv[0,0]
    q = Sigma_inv[0,1]
    r = Sigma_inv[1,0]
    s = Sigma_inv[1,1]

    cdef int N
    N = data.shape[0]

    cdef float noise_variance
    noise_variance = 1./(sigma_inv_noise*sigma_inv_noise)

    cdef int i, j
    cdef float g, h, chi2, v

    cdef np.ndarray[np.float_t, ndim=2] K
    K = np.empty((N, N))

    for i in xrange(N):
        for j in xrange(i+1):
            g = data[i,0] - data[j,0]
            h = data[i,1] - data[j,1]

            chi2 = g*(p*g+q*h) + h*(r*g+s*h)
            v = (sigma_f*sigma_f)*exp(-0.5*chi2)
            if i==j: v += noise_variance

            K[i,j] = K[j,i] = v
            

    return K