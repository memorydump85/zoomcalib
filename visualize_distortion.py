#! /usr/bin/python

import sys
import os.path
import numpy as np
import cPickle as pickle
from math import sqrt
from glob import iglob
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import scharr

from gp import GaussianProcess, sqexp2D_covariancef



#--------------------------------------
class HomographyModel(object):
#--------------------------------------
    """
    Encapsulation of the data stored in a '.lh0+' file
    and methods acting on that data
    """
    def __init__(self):
        self.filestem = None
        self.etag = None
        self.itag = None
        self.LH0 = None
        self.corrs = None


    @classmethod
    def load_from_file(class_, filename):
        # parse the filename to get intrinsic/extrinsic tags
        filestem = os.path.splitext(filename)[0]
        etag, itag = filestem.split('/')[-2:]

        with open(filename) as f:
            K, E = pickle.load(f)
            LH0 = K.dot(E)

        with open(filestem + '.corrs') as f:
            corrs = pickle.load(f)

        # create and populate instance
        instance = class_()
        instance.filestem = filestem
        instance.etag = etag
        instance.itag = itag
        instance.LH0 = LH0
        instance.corrs = corrs
        return instance


#--------------------------------------
class GPModel(object):
#--------------------------------------
    def __init__(self, points_i, values):
        assert len(points_i) == len(values)

        X = points_i
        S = np.cov(X.T)

        meanV = np.mean(values, axis=0)
        V = values - np.tile(meanV, (len(values), 1))

        self._meanV = meanV
        self._gp_x = GPModel._fit_gp(X, S, V[:,0])
        self._gp_y = GPModel._fit_gp(X, S, V[:,1])


    @staticmethod
    def _fit_gp(X, covX, t):
        xx, xy, yy = covX[0,0], covX[0,1], covX[1,1]

        # Perform hyper-parameter optimization with different
        # initial points and choose the GP with best model evidence
        theta0 = np.array(( t.std(), sqrt(xx), sqrt(yy), xy, 10. ))
        best_gp = GaussianProcess.fit(X, t, sqexp2D_covariancef, theta0)

        for tau in xrange(50, 800, 100):
            theta0 = np.array(( t.std(), tau, tau, 0, 10. ))
            gp = GaussianProcess.fit(X, t, sqexp2D_covariancef, theta0)
            if gp.model_evidence() > best_gp.model_evidence():
                best_gp = gp

        return best_gp


    def predict(self, X):
        V = np.vstack([ self._gp_x.predict(X), self._gp_y.predict(X) ]).T
        return V + np.tile(self._meanV, (len(X), 1))


def save_plot(hmodel):
    LH0 = hmodel.LH0
    det_w = np.array([ c.source for c in hmodel.corrs ])
    det_i = np.array([ c.target for c in hmodel.corrs ])

    #
    # Obtain distortion estimate
    #       detected + undistortion = mapped
    #  (or) undistortion = mapped - detected
    #
    def homogeneous_coords(arr):
        N = len(arr)
        return np.hstack([ arr, np.zeros((N, 1)), np.ones((N, 1)) ])

    mapped_i = LH0.dot(homogeneous_coords(det_w).T).T
    mapped_i = np.array([ p / p[2] for p in mapped_i ])
    mapped_i = mapped_i[:,:2]

    undistortion = mapped_i - det_i # image + undistortion = mapped
    max_distortion = np.max([np.linalg.norm(u) for u in undistortion])

    #
    # Save the undistortion observations to a file
    #
    with open(hmodel.filestem + '.uv', 'w') as f:
        pickle.dump(np.hstack([det_i, undistortion]), f)

    #
    # Fit non-parametric model to the observations
    #
    model = GPModel(det_i, undistortion)

    print '\nGP Hyper-parameters'
    print '---------------------'
    print '  x: ', model._gp_x._covf.theta
    print '        log-likelihood: %.4f' % model._gp_x.model_evidence()
    print '  y: ', model._gp_y._covf.theta
    print '        log-likelihood: %.4f' % model._gp_y.model_evidence()
    print ''
    print '  Optimization detail:'
    print '  [ x ]'
    print '  ' + str(model._gp_x.fit_result).replace('\n', '\n      ')
    print '  [ y ]'
    print '  ' + str(model._gp_y.fit_result).replace('\n', '\n      ')

    #
    # Visualization
    #
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    plt.figure(figsize=(16,10))

    #__1__
    plt.subplot(221)
    plt.title(hmodel.itag)

    im = imread(hmodel.filestem + '.png')
    im = rgb2gray(im)
    im = (im * 255.).astype(np.uint8)

    plt.imshow(im, cmap='gray')
    plt.plot(det_i[:,0], det_i[:,1], 'o')
    plt.grid()
    plt.axis('equal')

    X, Y = det_i[:,0], det_i[:,1]
    U, V = undistortion[:,0], undistortion[:,1]

    #__2__
    plt.subplot(222)
    plt.title('Exact Estimates')
    plt.quiver(X, Y, U, -V, angles='xy', scale_units='xy', scale=1) # plot exact
    plt.gca().invert_yaxis()
    plt.axis('equal')

    #__3__
    plt.subplot(223)
    plt.title('Scaled Estimates')
    plt.quiver(X, Y, U, -V, units='dots')
    plt.text( 0.5, 0, 'max observed distortion: %.2f px' % max_distortion, color='#CF4457', fontsize=10,
        horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.gca().invert_yaxis()
    plt.axis('equal')

    #__4__
    plt.subplot(224)
    plt.title('Scaled Undistortion')
    H, W = im.shape
    grid = np.array([[x, y] for y in xrange(0, H, 80) for x in xrange(0, W, 80)])
    predicted = model.predict(grid)
    X, Y = grid[:,0], grid[:,1]
    U, V = predicted[:,0], predicted[:,1]
    plt.quiver(X, Y, U, -V, units='dots')
    plt.gca().invert_yaxis()
    plt.axis('equal')

    plt.tight_layout()
    plt.gcf().patch.set_facecolor('#dddddd')
    plt.savefig(hmodel.filestem + '.svg', bbox_inches='tight')


def main():
    np.set_printoptions(precision=4, suppress=True)

    for filename in sys.argv[1:]:
        hmodel = HomographyModel.load_from_file(filename)
        save_plot(hmodel)

if __name__ == '__main__':
    main()