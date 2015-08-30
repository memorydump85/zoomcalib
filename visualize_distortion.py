#! /usr/bin/python

import sys
import os.path
import numpy as np
import cPickle as pickle
from glob import iglob

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import scharr



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
            LH0 = pickle.load(f)

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
    # for i, d in enumerate(detections):
    #     plt.text(d.c[0], d.c[1], str(i),
    #         fontsize=8, color='white', bbox=dict(facecolor='maroon', alpha=0.75))
    plt.grid()
    plt.axis('equal')

    X, Y = det_i[:,0], det_i[:,1]
    U, V = undistortion[:,0], undistortion[:,1]

    #__2__
    plt.subplot(223)
    plt.title('Scaled Estimates')
    plt.quiver(X, Y, U, -V, units='dots')
    plt.text( 0.5, 0, 'max observed distortion: %.2f px' % max_distortion, color='#CF4457', fontsize=10,
        horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.gca().invert_yaxis()
    plt.axis('equal')

    #__3__
    plt.subplot(224)
    plt.title('Exact Estimates')
    plt.quiver(X, Y, U, -V, angles='xy', scale_units='xy', scale=1) # plot exact
    plt.gca().invert_yaxis()
    plt.axis('equal')

    plt.savefig(hmodel.filestem + '.svg', bbox_inches='tight')


def main():
    np.set_printoptions(precision=4, suppress=True)

    for filename in sys.argv[1:]:
        hmodel = HomographyModel.load_from_file(filename)
        save_plot(hmodel)

if __name__ == '__main__':
    main()