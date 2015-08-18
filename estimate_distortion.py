#! /usr/bin/python

import numpy as np
from math import sqrt
from itertools import chain
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.optimize import minimize

from apriltag import AprilTagDetector
from tag36h11_mosaic import TagMosaic
from projective_math import WeightedLocalHomography, SqExpWeightingFunction
from refine_homography import estimate_intrinsics_assume_cxy_noskew
from refine_homography import get_extrinsics_from_homography
from gp import GaussianProcess, sqexp2D_covariancef



def create_local_homography_object(bandwidth, magnitude, lambda_):
    """
    Helper function for creating WeightedLocalHomography objects
    """
    H = WeightedLocalHomography(SqExpWeightingFunction(bandwidth, magnitude))
    H.regularization_lambda = lambda_
    return H


def local_homography_error_impl(theta, t_src, t_tgt, v_src, v_tgt):
    """
    This is the objective function used for optimizing parameters of
    the `SqExpWeightingFunction` used for local homography fitting

    Parameters:
    -----------
       `theta` = [ `bandwidth`, `magnitude`, `lambda_` ]:
            parameters of the `SqExpWeightingFunction`

    Arguments:
    -----------
        `t_src`: list of training source points
        `t_tgt`: list of training target points
        `v_src`: list of validation source points
        `v_tgt`: list of validation target points
    """
    H = create_local_homography_object(*theta)
    for s, t in zip(t_src, t_tgt):
        H.add_correspondence(s, t)

    v_mapped = np.array([ H.map(s)[:2] for s in v_src ])
    return ((v_mapped - v_tgt)**2).sum(axis=1).mean()


def local_homography_error(theta, args):
    src, tgt = args
    return local_homography_error_impl(theta, src[:9], tgt[:9], src[9:25], tgt[9:25])


def matrix_to_xyzrph(M):
    tx = M[0,3]
    ty = M[1,3]
    tz = M[2,3]
    rx = np.arctan2(M[2,1], M[2,2])
    ry = np.arctan2(-M[2,0], sqrt(M[0,0]*M[0,0] + M[1,0]*M[1,0]))
    rz = np.arctan2(M[1,0], M[0,0])
    return np.array([tx, ty, tz, rx, ry, rz])


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


def process(filename):
    #
    # Conventions:
    # a_i, b_i
    #    are variables in image space, units are pixels
    # a_w, b_w
    #    are variables in world space, units are meters
    #
    print '\n========================================'
    print '  File: ' + filename
    print '========================================\n'

    im = imread(filename)
    im = rgb2gray(im)
    im = (im * 255.).astype(np.uint8)

    tag_mosaic = TagMosaic(0.0254)
    detections = AprilTagDetector().detect(im)
    print '  %d tags detected.' % len(detections)

    #
    # Sort detections by distance to center
    #
    c_i = np.array([im.shape[1], im.shape[0]]) / 2.
    dist = lambda p_i: np.linalg.norm(p_i - c_i)
    closer_to_center = lambda d1, d2: int(dist(d1.c) - dist(d2.c))
    detections.sort(cmp=closer_to_center)

    mosaic_pos = lambda det: tag_mosaic.get_position_meters(det.id)

    det_i = np.array([ d.c for d in detections ])
    det_w = np.array([ mosaic_pos(d) for d in detections ])

    #
    # To learn a weighted local homography, we find the weighting
    # function parameters that minimize reprojection error across
    # validation folds of the data.
    #
    def learn_homography_i2w():
        result = minimize( local_homography_error,
                    x0=[ 50, 1, 1e-3 ],
                    args=[ det_i, det_w ],
                    method='Powell',
                    options={'ftol': 1e-3} )

        print '\nHomography: i->w'
        print '------------------'
        print '  params:', result.x
        print '    rmse: %.6f' % sqrt(result.fun)
        print '\n  Optimization detail:'
        print '  ' + str(result).replace('\n', '\n      ')

        H = create_local_homography_object(*result.x)
        for i, w in zip(det_i, det_w):
            H.add_correspondence(i, w)

        return H

    def learn_homography_w2i():
        result = minimize( local_homography_error,
                    x0=[ 0.0254, 1, 1e-3 ],
                    method='Powell',
                    args=[ det_w, det_i ],
                    options={'ftol': 1e-3} )

        print '\nHomography: w->i'
        print '------------------'
        print '  params:', result.x
        print '    rmse: %.6f' % sqrt(result.fun)
        print '\n  Optimization detail:'
        print '  ' + str(result).replace('\n', '\n      ')

        H = create_local_homography_object(*result.x)
        for w, i in zip(det_w, det_i):
            H.add_correspondence(w, i)

        return H

    #
    # We assume that the distortion is zero at the center of
    # the image and we are interesting in the word to image
    # homography at the center of the image. However, we don't
    # know the center of the image in world coordinates.
    # So we follow the sequence:
    #
    # First, we learn a homography from image to world
    # Next, find where the image center `c_i` maps to in world coordinates (`c_w`)
    # Finally, find the local homography `LH0` from world to image at `c_w`
    #
    H_iw = learn_homography_i2w()
    c_i = np.array([im.shape[1], im.shape[0]]) / 2.
    c_w = H_iw.map(c_i)[:2]
    H_wi = learn_homography_w2i()
    LH0 = H_wi.get_homography_at(c_w)

    print '\nHomography at center'
    print '----------------------'
    print '      c_w =', c_w
    print '      c_i =', c_i
    print 'LH0 * c_w =', H_wi.map(c_w)

    #
    # Obtain distortion estimate
    #       detected + undistortion = mapped
    #  (or) undistortion = mapped - detected
    #
    def homogeneous_coords(arr):
        return np.hstack([ arr, np.ones((len(arr), 1)) ])

    mapped_i = LH0.dot(homogeneous_coords(det_w).T).T
    mapped_i = np.array([ p / p[2] for p in mapped_i ])
    mapped_i = mapped_i[:,:2]

    undistortion = mapped_i - det_i # image + undistortion = mapped
    max_distortion = np.max([np.linalg.norm(u) for u in undistortion])
    print '\nMaximum distortion is %.2f pixels' % max_distortion

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
    from skimage.filters import scharr
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    plt.figure(figsize=(16,10))

    #__1__
    plt.subplot(221)
    plt.title(filename.split('/')[-1])
    plt.imshow(im, cmap='gray')
    plt.plot(det_i[:,0], det_i[:,1], 'o')
    for i, d in enumerate(detections):
        plt.text(d.c[0], d.c[1], str(i),
            fontsize=8, color='white', bbox=dict(facecolor='maroon', alpha=0.75))
    plt.grid()
    plt.axis('equal')

    #__2__
    plt.subplot(222)
    plt.title('Non-linear Homography')

    if False:
        plt.imshow(1.-scharr(im), cmap='bone', interpolation='gaussian')

        from collections import defaultdict
        row_groups = defaultdict(list)
        col_groups = defaultdict(list)

        for d in detections:
            row, col = tag_mosaic.get_row(d.id), tag_mosaic.get_col(d.id)
            row_groups[row] += [ d.id ]
            col_groups[col] += [ d.id ]

        for k, v in row_groups.iteritems():
            a = tag_mosaic.get_position_meters(np.min(v))
            b = tag_mosaic.get_position_meters(np.max(v))
            x_coords = np.linspace(a[0], b[0], 100)
            points = np.array([ H_wi.map([x, a[1]]) for x in x_coords ])
            plt.plot(points[:,0], points[:,1], '-',color='#CF4457', linewidth=2)

        for k, v in col_groups.iteritems():
            a = tag_mosaic.get_position_meters(np.min(v))
            b = tag_mosaic.get_position_meters(np.max(v))
            y_coords = np.linspace(a[1], b[1], 100)
            points = np.array([ H_wi.map([a[0], y]) for y in y_coords ])
            plt.plot(points[:,0], points[:,1], '-',color='#CF4457', linewidth=2)

        plt.plot(det_i[:,0], det_i[:,1], 'kx')
        plt.grid()
        plt.axis('equal')

    #__3__
    plt.subplot(223)
    plt.title('Qualitative Estimates')

    for i, d in enumerate(detections):
        plt.text(d.c[0], d.c[1], str(i), fontsize=8, color='#999999')

    X, Y = det_i[:,0], det_i[:,1]
    U, V = undistortion[:,0], undistortion[:,1]
    plt.quiver(X, Y, U, -V, units='dots')
    # plt.quiver(X, Y, U, -V, angles='xy', scale_units='xy', scale=1) # plot exact
    plt.text( 0.5, 0, 'max observed distortion: %.2f px' % max_distortion, color='#CF4457', fontsize=10,
        horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.gca().invert_yaxis()
    plt.axis('equal')

    #__4__
    plt.subplot(224)
    plt.title('Qualitative Undistortion')
    H, W = im.shape
    grid = np.array([[x, y] for y in xrange(0, H, 80) for x in xrange(0, W, 80)])
    predicted = model.predict(grid)
    X, Y = grid[:,0], grid[:,1]
    U, V = predicted[:,0], predicted[:,1]
    plt.quiver(X, Y, U, -V, units='dots')
    #plt.quiver(X, Y, U, -V, angles='xy', scale_units='xy', scale=1)) # plot exact
    plt.gca().invert_yaxis()
    plt.axis('equal')

    plt.tight_layout()
    plt.gcf().patch.set_facecolor('#dddddd')
    #plt.show()
    plt.savefig(filename + '.svg', bbox_inches='tight')


def main():
    import sys

    np.set_printoptions(precision=4, suppress=True)

    if len(sys.argv)==1:
        imfiles = ["/var/tmp/capture/070.png"]
    else:
        imfiles = sys.argv[1:]

    for filename in imfiles:
        process(filename)

if __name__ == '__main__':
    main()