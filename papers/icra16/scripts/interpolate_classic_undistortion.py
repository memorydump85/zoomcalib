#! /usr/bin/python

import sys
from glob import iglob
import numpy as np
import cPickle as pickle
from itertools import groupby
from skimage.io import imread
from skimage.color import rgb2gray

from apriltag import AprilTagDetector
from tag36h11_mosaic import TagMosaic
from visualize_distortion import GPModel
from homography_at_center import get_tag_detections



#--------------------------------------
class TagDetectionEx(object):
#--------------------------------------
    def __init__(self, apriltagdet):
        self.id = apriltagdet.id
        self.c0 = apriltagdet.c
        self.c1 = None


def main():
    model_folder = sys.argv[1]
    test_image = sys.argv[2]

    zoom_levels = [ float(f.split('/')[-2]) for f in iglob(model_folder + '/*/') ]
    zoom_levels = np.array(sorted(zoom_levels))
    test_zoom = float(test_image.split('/')[-2])

    test_zoom_pos = np.searchsorted(zoom_levels, test_zoom)
    zoom0 = zoom_levels[test_zoom_pos-1]
    zoom1 = zoom_levels[test_zoom_pos]

    with open('%s/%03d/classic.poly' % (model_folder,zoom0)) as f:
        model0 = pickle.load(f)

    with open('%s/%03d/classic.poly' % (model_folder,zoom1)) as f:
        model1 = pickle.load(f)

    def interpolate_predict(query_points):
        predict0 = np.array([ model0.undistort(q) for q in query_points ])
        predict1 = np.array([ model1.undistort(q) for q in query_points ])

        t = (zoom1-test_zoom) / (zoom1-zoom0)
        return t*predict0 + (1.-t)*predict1


    im = imread(test_image)
    im = rgb2gray(im)
    detections = [ TagDetectionEx(d) for d in get_tag_detections(im) ]

    # Rectify tag detections and store in `c1` attribute
    det_i = np.array([ d.c0 for d in detections ])
    undist = interpolate_predict(det_i)

    for d, r in zip(detections, undist):
        d.c1 = np.add(d.c0, r)

    def line_fit_sqerr(points):
        """ smallest eigen value of covariance """
        cov = np.cov(points)
        return np.linalg.eig(cov)[0].min()

    mosaic = TagMosaic(0.0254)

    # Row-wise average residual
    tag_row = lambda d: mosaic.get_row(d.id)
    detections.sort(key=tag_row)

    sq_errs = []
    for _, group in groupby(detections, key=tag_row):
        rectified = np.array([ d.c1 for d in group ])
        if len(rectified) < 3: continue
        sq_errs.append( line_fit_sqerr(rectified.T) )

    sq_errs = np.array(sq_errs)
    row_rmse, row_rmaxse = np.sqrt(sq_errs.mean()), np.sqrt(sq_errs.max())

    # Col-wise average residual
    tag_col = lambda d: mosaic.get_col(d.id)
    detections.sort(key=tag_col)

    sq_errs = []
    for _, group in groupby(detections, key=tag_col):
        rectified = np.array([ d.c1 for d in group ])
        if len(rectified) < 3: continue
        sq_errs.append( line_fit_sqerr(rectified.T) )

    sq_errs = np.array(sq_errs)
    col_rmse, col_rmaxse = np.sqrt(sq_errs.mean()), np.sqrt(sq_errs.max())

    print '( %d, %.2f, %.2f ),' % (
        int(test_image.split('/')[-2]),
        np.sqrt(row_rmse**2 + col_rmse**2),
        max(row_rmaxse, col_rmaxse))

if __name__ == '__main__':
    main()