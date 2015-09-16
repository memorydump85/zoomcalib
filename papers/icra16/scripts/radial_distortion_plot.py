#! /usr/bin/python

import os.path
import numpy as np
import cPickle as pickle

from camera_math import matrix_to_xyzrph, matrix_to_intrinsics



#--------------------------------------
class CameraCalibration(object):
#--------------------------------------
    """
    Encapsulation of the data stored in a '.lh0+' file
    and the corresponding undistortion in a '.uv' file
    """
    def __init__(self):
        self.etag = None
        self.itag = None
        self.intrinsics = None
        self.extrinsics = None
        self.undistortion = None


    @classmethod
    def load_from_file(class_, filename):
        # parse the filename to get intrinsic/extrinsic tags
        filestem = os.path.splitext(filename)[0]
        itag, etag = filestem.split('/')[-2:]

        with open(filestem + '.lh0+') as f:
            K, E = pickle.load(f)

        with open(filestem + '.uv') as f:
            undistortion = pickle.load(f)

        # create and populate instance
        instance = class_()
        instance.etag = etag
        instance.itag = itag
        instance.intrinsics = matrix_to_intrinsics(K)
        instance.extrinsics = matrix_to_xyzrph(E)
        instance.undistortion = undistortion
        return instance


    def undistortion_data(self):
        """
        undistortion augmented with zoom data, such that
        the data has the columns [ X, Y, Zoom, U, V ]
        """
        N = len(self.undistortion)
        zoom = np.repeat(float(self.itag), N)
        x, y, u, v = self.undistortion.T
        return np.vstack([ x, y, zoom, u, v ]).T


def main():
    import sys

    from matplotlib import pyplot as plt
    plt.figure(figsize=(4*1.5,7*1.5))

    for num, filename in enumerate(sys.argv[1:]):
        calibration = CameraCalibration.load_from_file(filename)
        x, y, _, u, v = calibration.undistortion_data().T
        fx, fy, cx, cy = calibration.intrinsics

        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        d = np.sqrt(u**2 + v**2)

        def polyfit(degree):
            SCALE = 1600. # for stability
            phi = lambda v : np.vstack([ v**n for n in xrange(2, degree+1) ])
            coeff, _, _, _ = np.linalg.lstsq(phi(r/SCALE).T, d/SCALE)
            print coeff
            poly_x = np.arange(0, 1.25, .001)
            poly_y = phi(poly_x).T.dot(coeff)
            return poly_x*SCALE, poly_y*SCALE

        #
        # Visualization
        #
        ax = plt.subplot(3, 1, num+1)
        plt.plot(r, d, 'ko', markersize=2)

        plt.plot(*polyfit(2), linestyle='-', color='#922428', linewidth=2, label='degree 2')
        plt.plot(*polyfit(4), linestyle='-', color='#DA7C30', linewidth=2, label='degree 4')
        plt.plot(*polyfit(5), linestyle='-', color='#396AB1', linewidth=2, label='degree 5')
        plt.xticks([0, 400, 800, 1200, 1600])
        plt.gca().set_ylim([0, 120])
        plt.gca().set_xlim([0, 2000])
        if num == 2:
            plt.xlabel('radius (pixels)')
        if num == 1:
            plt.ylabel('distortion (pixels)')
        if num != 2:
            plt.setp( ax.get_xticklabels(), visible=False)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(b=True, which='major', color='#ededed', linestyle='-')
        plt.gca().set_axisbelow(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()