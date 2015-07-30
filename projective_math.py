import numpy as np
from collections import namedtuple



Correspondence_ = namedtuple('Correspondence_', ['source', 'target'])


class UnitWeightingFunction(object):
    def __call__(self, p, q):
        return 1


class SqExpWeightingFunction(object):
    def __init__(self, bandwidth, magnitude=1.):
        self.tau_ = bandwidth
        self.lambda_ = magnitude

    def __call__(self, p, q):
        z = np.subtract(p, q) / self.tau_
        return self.lambda_**2 * np.exp(-(z**2).sum())


def normalization_transform_(points):
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


def homogeneous_coords_(p):
    assert len(p)==2 or len(p)==3
    return p if len(p)==3 else np.array([p[0], p[1], 1.])


#--------------------------------------
class WeightedLocalHomography(object):
#--------------------------------------
    def __init__(self, wfunc=UnitWeightingFunction()):
        self.corrs_ = []
        self.regularization_lambda = 0
        self.weighting_func = wfunc
        self.precompute_done_ = False


    def add_correspondence(self, source_xy, target_xy):
        assert len(source_xy) == 2
        assert len(target_xy) == 2
        self.corrs_.append(Correspondence_(source_xy, target_xy))


    def precompute_(self):
        """ precompute normalization transforms and constraint
        matrix. These remain the same for every homography query """
        if self.precompute_done_ == True:
            return

        self.srcX_, _             = normalization_transform_([ c.source for c in self.corrs_ ])
        self.tgtX_, self.tgtXinv_ = normalization_transform_([ c.target for c in self.corrs_ ])

        constraints = []
        for c in self.corrs_:
            x, y, _ = self.srcX_.dot(homogeneous_coords_(c.source))
            i, j, _ = self.tgtX_.dot(homogeneous_coords_(c.target))

            constraints.append( [-x, -y, -1, 0, 0, 0, i*x, i*y, i] )
            constraints.append( [0, 0, 0, -x, -y, -1, j*x, j*y, j] )

        self.constraint_matrix = np.array(constraints, dtype=np.float)
        self.precompute_done_ = True


    def get_homography_at(self, src_pt):
        self.precompute_()

        A = self.constraint_matrix
        
        # The weighting is a diagonal matrix that encodes how similar
        # `src_pt` is to each source point in `self.corrs_`
        w_diag = [ self.weighting_func(src_pt, c.source) for c in self.corrs_ ]

        # Each correspondence produces 2 constraints. So weights also
        # need to be repeated
        w_diag = np.sqrt(np.repeat(w_diag, 2))
        assert A.shape[0] == w_diag.shape[0]

        lambdaI = self.regularization_lambda * np.eye(A.shape[0])
        W = np.diag(w_diag)
        U, s, V = np.linalg.svd((W + lambdaI).dot(A))

        # Homography is the total least squares solution: The eigen-vector
        # corresponding to the smallest eigen-value
        H = V.T[:,-1].reshape((3,3))

        return reduce(np.dot, [self.tgtXinv_, H, self.srcX_])


    def map(self, src_pt):
        """ Map `src_pt` to the target plane using the local
        homography at `src_pt` """
        m = self.get_homography_at(src_pt).dot(homogeneous_coords_(src_pt))
        m /= m[2]
        return m