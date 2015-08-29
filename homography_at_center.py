#! /usr/bin/python

import numpy as np
from math import sqrt
import os.path
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.optimize import minimize
from sklearn.cross_validation import LeaveOneOut

from apriltag import AprilTagDetector
from tag36h11_mosaic import TagMosaic
from projective_math import WeightedLocalHomography, SqExpWeightingFunction
from tupletypes import Correspondence, WorldImageHomographyInfo



def create_local_homography_object(bandwidth, magnitude, lambda_):
    """
    Helper function for creating WeightedLocalHomography objects
    """
    H = WeightedLocalHomography(SqExpWeightingFunction(bandwidth, magnitude))
    H.regularization_lambda = lambda_
    return H


def local_homography_error(theta, t_src, t_tgt, v_src, v_tgt):
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


def get_homography_model(filename):
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
    # leave-one-out validation folds of the data. Since the
    # homography is local at the center, we only use 9 detections
    # nearest to the center
    #
    det_i9 = det_i[:9]
    det_w9 = det_w[:9]

    def local_homography_loocv_error(theta, args):
        src, tgt = args
        errs = [ local_homography_error(theta, src[t_ix], tgt[t_ix], src[v_ix], tgt[v_ix])
                    for t_ix, v_ix in LeaveOneOut(len(src)) ]
        return np.mean(errs)

    def learn_homography_i2w():
        result = minimize( local_homography_loocv_error,
                    x0=[ 50, 1, 1e-3 ],
                    args=[ det_i9, det_w9 ],
                    method='Powell',
                    options={'ftol': 1e-3} )

        print '\nHomography: i->w'
        print '------------------'
        print '  params:', result.x
        print '    rmse: %.6f' % sqrt(result.fun)
        print '\n  Optimization detail:'
        print '  ' + str(result).replace('\n', '\n      ')

        H = create_local_homography_object(*result.x)
        for i, w in zip(det_i9, det_w9):
            H.add_correspondence(i, w)

        return H

    def learn_homography_w2i():
        result = minimize( local_homography_loocv_error,
                    x0=[ 0.0254, 1, 1e-3 ],
                    method='Powell',
                    args=[ det_w9, det_i9 ],
                    options={'ftol': 1e-3} )

        print '\nHomography: w->i'
        print '------------------'
        print '  params:', result.x
        print '    rmse: %.6f' % sqrt(result.fun)
        print '\n  Optimization detail:'
        print '  ' + str(result).replace('\n', '\n      ')

        H = create_local_homography_object(*result.x)
        for w, i in zip(det_w9, det_i9):
            H.add_correspondence(w, i)

        return H

    #
    # We assume that the distortion is zero at the center of
    # the image and we are interesting in the word to image
    # homography at the center of the image. However, we don't
    # know the center of the image in world coordinates.
    # So we follow a procedure as explained below:
    #
    # First, we learn a homography from image to world
    # Next, we find where the image center `c_i` maps to in
    # world coordinates (`c_w`). Finally, we find the local
    # homography `LH0` from world to image at `c_w`
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
    print ''

    corrs = [ Correspondence(w, i) for w, i in zip(det_w, det_i) ]
    return corrs, WorldImageHomographyInfo(H_wi, c_w, c_i)


def main():
    np.set_printoptions(precision=4, suppress=True)
    import sys
    import cPickle as pickle

    for filename in sys.argv[1:]:
        corrs, model = get_homography_model(filename)
        filestem = os.path.splitext(filename)[0]
        with open(filestem + '.lh0', 'w') as f:
            pickle.dump(model, f)
        with open(filestem + '.corrs', 'w') as f:
            pickle.dump(corrs, f)

if __name__ == '__main__':
    main()
