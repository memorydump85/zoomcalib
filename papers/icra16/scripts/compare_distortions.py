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


def main():
    import sys
    from glob import iglob

    np.set_printoptions(precision=4, suppress=True)

    #
    # Visualization
    #
    from matplotlib import pyplot as plt
    colors = ['red', 'blue', 'black']

    for c, filename in zip(colors, sys.argv[1:]):
        calibration = CameraCalibration.load_from_file(filename)
        X, Y, U, V = calibration.undistortion.T
        plt.quiver(X, Y, U, -V, units='dots', color=c)

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()