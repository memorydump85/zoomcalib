#! /usr/bin/python

import sys
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

    with open(model_folder + '/pose0.gp') as f:
        gpmodel = pickle.load(f)

    mosaic = TagMosaic(0.0254)

    for imfile in sys.argv[2:]:
        im = imread(imfile)
        im = rgb2gray(im)
        detections = [ TagDetectionEx(d) for d in get_tag_detections(im) ]

        # Rectify tag detections and store in `c1` attribute
        det_i = np.array([ d.c0 for d in detections ])
        undist = gpmodel.predict(det_i)

        for d, r in zip(detections, undist):
            d.c1 = np.add(d.c0, r)

        def line_fit_sqerr(points):
            """ smallest eigen value of covariance """
            cov = np.cov(points)
            return np.linalg.eig(cov)[0].min()

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

        print '[ %d, %.2f, %.2f ],' % (
            int(imfile.split('/')[-2]),
            np.sqrt(row_rmse**2 + col_rmse**2),
            max(row_rmaxse, col_rmaxse))


if __name__ == '__main__':
    main()