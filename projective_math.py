import numpy as np
from collections import namedtuple



_Correspondence = namedtuple('_Correspondence', ['source', 'target'])


class UnitWeightingFunction(object):
    def __call__(self, p, q):
        return 1


class SqExpWeightingFunction(object):
    def __init__(self, bandwidth, magnitude=1.):
        self._tau = bandwidth
        self._nu = magnitude

    def __call__(self, p, q):
        z = np.subtract(p, q) / self._tau
        return self._nu*self._nu * np.exp(-(z*z).sum())


def _normalization_transform(points):
    muX, muY = np.mean(points, axis=0)
    stdX, stdY = np.std(points, axis=0)
    scaleX, scaleY = 1./stdX, 1./stdY

    X = np.array([
            [ scaleX,     0, -muX*scaleX ],
            [ 0,     scaleY, -muY*scaleY ],
            [ 0,          0,           1 ]])

    Xinv = np.array([
            [ stdX,       0,         muX ],
            [ 0,       stdY,         muY ],
            [ 0,          0,           1 ]])

    return X, Xinv


def _homogeneous_coords(p):
    assert len(p)==2 or len(p)==3
    return p if len(p)==3 else np.array([p[0], p[1], 1.])


#--------------------------------------
class WeightedLocalHomography(object):
#--------------------------------------
    def __init__(self, wfunc=UnitWeightingFunction()):
        self._corrs = []
        self.regularization_lambda = 0
        self._weighting_func = wfunc
        self._precompute_done = False


    def add_correspondence(self, source_xy, target_xy):
        assert len(source_xy) == 2
        assert len(target_xy) == 2
        self._corrs.append(_Correspondence(source_xy, target_xy))


    def _precompute(self):
        """ precompute normalization transforms and constraint
        matrix. These remain the same for every homography query """
        if self._precompute_done == True:
            return

        self._srcX, _             = _normalization_transform([ c.source for c in self._corrs ])
        self._tgtX, self._tgtXinv = _normalization_transform([ c.target for c in self._corrs ])

        constraints = []
        for c in self._corrs:
            x, y, _ = self._srcX.dot(_homogeneous_coords(c.source))
            i, j, _ = self._tgtX.dot(_homogeneous_coords(c.target))

            constraints.append( [-x, -y, -1, 0, 0, 0, i*x, i*y, i] )
            constraints.append( [0, 0, 0, -x, -y, -1, j*x, j*y, j] )

        self.constraint_matrix = np.array(constraints, dtype=np.float)
        self._precompute_done = True


    def get_homography_at(self, src_pt):
        self._precompute()

        A = self.constraint_matrix

        # The weighting is a diagonal matrix that encodes how similar
        # `src_pt` is to each source point in `self._corrs`
        w_diag = [ self._weighting_func(src_pt, c.source) for c in self._corrs ]

        # Each correspondence produces 2 constraints. So weights also
        # need to be repeated
        w_diag = np.sqrt(np.repeat(w_diag, 2))
        assert A.shape[0] == w_diag.shape[0]

        lambda2I = (self.regularization_lambda*self.regularization_lambda) * np.eye(A.shape[0])
        W = np.diag(w_diag)
        U, s, V = np.linalg.svd((W + lambda2I).dot(A))

        # Homography is the total least squares solution: The eigen-vector
        # corresponding to the smallest eigen-value
        H = V.T[:,-1].reshape((3,3))

        return reduce(np.dot, [self._tgtXinv, H, self._srcX])


    def map(self, src_pt):
        """ Map `src_pt` to the target plane using the local
        homography at `src_pt` """
        m = self.get_homography_at(src_pt).dot(_homogeneous_coords(src_pt))
        m /= m[2]
        return m


def estimate_intrinsics(homographies):
    """
    Estimate camera intrinsics `K` from a list of `homographies`. The
    method uses constraints imposed by the image of the absolute conic,
    as described in:
        Z. Zhang, "A flexible new technique for camera calibration",
        Section 3.1,
        IEEE Transactions on Pattern Analysis and Machine Intelligence

    If `homographies` contains only one homography, the center of the
    camera `cxy` in pixel coordinates must be known and supplied as an
    input parameter.

    if 'homographies' contains less than 3 homographies, the skew (or
    aspect ratio) of the estimated intrinsics matrix `K` is assumed to
    be equal to zero.
    """
    assert len(homographies) > 0

    if len(homographies) == 1:
        raise Exception('Need atleast 2 homographies to estimate intrinsics.\n' +
            'Use `estimate_intrinsics_assume_cxy_noskew` if you need to estimate' +
            'intrinsics from just one homography')

    constraints = []
    for H in homographies:

        def c(i, j):
            a0, a1, a2 = H[:,i]
            b0, b1, b2 = H[:,j]
            return np.array([
                a0*b0, a0*b1 + a1*b0, a1*b1, a2*b0 + a0*b2, a2*b1 + a1*b2, a2*b2 ])

        constraints.append(c(0,1))
        constraints.append(c(0,0) - c(1,1))

    if len(homographies) == 2: # skew = 0
        constraints.append(np.array([0, 1, 0, 0, 0, 0]))

    C = np.vstack(constraints)
    U, s, V = np.linalg.svd(C)

    # 1-based indices. Yuck! But code is consistent with the
    # formulas in the paper.
    b11, b12, b22, b13, b23, b33 = V.T[:,-1]

    v0 = (b12*b13 - b11*b23) / (b11*b22 - b12*b12)
    lambda_ = b33 - (b13*b13 + v0*(b12*b13 - b11*b23)) / b11
    alpha = np.sqrt(lambda_ / b11)
    beta = np.sqrt(lambda_*b11 / (b11*b22 - b12*b12))
    gamma = -b12*alpha*alpha*beta / lambda_
    u0 = gamma*v0 / beta - b13*alpha*alpha / lambda_

    return np.array([[ alpha, gamma,   u0 ],
                     [     0,  beta,   v0 ],
                     [     0,     0,    1 ]])


def estimate_intrinsics_assume_cxy_noskew(homographies, cxy):
    """
    Similar to `estimate_intrinsics`, but assumes that the camera
    center `cxy` in known, and the skew of the camera gamma is zero.
    These assumptions allow this method to estimate camera intrinsics
    from just one input homography.
    """
    assert len(homographies) > 0

    u0, v0 = cxy

    constraints = []
    for H in homographies:
        h00, h10, h20 = H[:,0]
        h01, h11, h21 = H[:,1]

        constraints.append([
            h00*h01 - u0*h00*h21 - u0*h01*h20 + u0*u0*h20*h21,
            h10*h11 - v0*h10*h21 - v0*h11*h20 + v0*v0*h20*h21,
            h20*h21
            ])
        constraints.append([
            h00*h00 - h01*h01 - 2*u0*h00*h20 + 2*u0*h01*h21 + u0*u0*h20*h20 - u0*u0*h21*h21,
            h10*h10 - h11*h11 - 2*v0*h10*h20 + 2*v0*h11*h21 + v0*v0*h20*h20 - v0*v0*h21*h21,
            h20*h20 - h21*h21
            ])

    C = np.vstack(constraints)
    U, s, V = np.linalg.svd(C)
    alpha, beta, _ = np.sqrt(1./V.T[:,-1])

    return np.array([[ alpha,    0.,   u0 ],
                     [     0,  beta,   v0 ],
                     [     0,     0,    1 ]])


def get_extrinsics_from_homography(H, intrinsics):
    M = np.linalg.inv(intrinsics).dot(H)
    M0 = M[:,0]
    M1 = M[:,1]

    # Columns should be unit vectors
    M0_scale = np.linalg.norm(M0)
    M1_scale = np.linalg.norm(M1)
    scale = np.sqrt(M0_scale) * np.sqrt(M1_scale)

    M /= scale
    # Recover sign of scale factor by noting that observations
    # must be in front of the camera, that is: z < 0
    if M[1,2] > 0: M *= -1

    # Assemble extrinsics matrix from the columns of M
    E = np.eye(4)
    E[:3,0] = M0
    E[:3,1] = M1
    E[:3,2] = np.cross(M0, M1)
    E[:3,3] = M[:,2]

    # Ensure that the rotation part of `E` is ortho-normal
    # For this we use the polar decomposition to find the
    # closest ortho-normal matrix to `E[:3,:3]`
    U, s, Vt = np.linalg.svd(E[:3,:3])
    E[:3,:3] = U.dot(Vt)

    return E
