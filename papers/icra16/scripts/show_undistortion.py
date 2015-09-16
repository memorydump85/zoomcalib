#! /usr/bin/python

import sys
import numpy as np
import cPickle as pickle
from itertools import groupby, cycle
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib
from matplotlib import pyplot as plt

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
    matplotlib.rcParams.update({'font.size': 8})

    model_folder = sys.argv[1]

    with open(model_folder + '/pose0.gp') as f:
        gpmodel = pickle.load(f)

    mosaic = TagMosaic(0.0254)

    for num, imfile in enumerate(sys.argv[2:14]):
        im = imread(imfile)
        im = rgb2gray(im)
        detections = [ TagDetectionEx(d) for d in get_tag_detections(im) ]
        print imfile

        # Rectify tag detections and store in `c1` attribute
        det_i = np.array([ d.c0 for d in detections ])
        undist = gpmodel.predict(det_i)
        rectified = det_i + undist

        for d, r in zip(detections, undist):
            d.c1 = np.add(d.c0, r)

        plt.subplot(3, 4, num+1)
        plt.axis('off')

        plt.plot(det_i[:,0], det_i[:,1], 'o', markersize=3, markeredgecolor='b', markerfacecolor='c')
        plt.plot(rectified[:,0], rectified[:,1], 'ko', markersize=3)

        def line_fit_sqerr(points):
            """ smallest eigen value of covariance """
            cov = np.cov(points)
            return np.linalg.eig(cov)[0].min()

        # Plot rows
        tag_row = lambda d: mosaic.get_row(d.id)
        detections.sort(key=tag_row)

        sq_errs = []
        colors = cycle(['#922428', '#CC2529'])
        for _, group in groupby(detections, key=tag_row):
            rectified = np.array([d.c1 for d in group])
            plt.plot(rectified[:,0], rectified[:,1], '-', color=colors.next())
            if len(rectified) < 3: continue
            sq_errs.append( line_fit_sqerr(rectified.T) )

        sq_errs = np.array(sq_errs)
        row_rmse = np.sqrt(sq_errs.max())

        # Plot cols
        tag_col = lambda d: mosaic.get_col(d.id)
        detections.sort(key=tag_col)

        sq_errs = []
        colors = cycle(['#6B4C9A', '#396AB1'])
        for _, group in groupby(detections, key=tag_col):
            rectified = np.array([d.c1 for d in group])
            plt.plot(rectified[:,0], rectified[:,1], '-', color=colors.next())
            if len(rectified) < 3: continue
            sq_errs.append( line_fit_sqerr(rectified.T) )

        sq_errs = np.array(sq_errs)
        col_rmse = np.sqrt(sq_errs.max())

        plt.title( '%s (%.1f, %.1f)' % (imfile.split('/')[-1], row_rmse, col_rmse) )

    plt.show()




if __name__ == '__main__':
    main()