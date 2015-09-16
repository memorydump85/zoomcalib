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


    def undistortion_magnitudes(self):
        x, y, u, v = self.undistortion.T
        return np.sqrt(u**2+v**2)


def main():
    import sys
    from glob import iglob

    np.set_printoptions(precision=4, suppress=True)

    calibrations = [ CameraCalibration.load_from_file(f) for f in sys.argv[1:] ]
    undistortion_data = np.vstack([ cc.undistortion_data() for cc in calibrations ])

    #
    # Visualization
    #
    from matplotlib import pyplot as plt

    #plt.style.use('ggplot')
    plt.figure(figsize=(8,5))

    zoom_stops = [ float(cc.itag) for cc in calibrations ]
    intrinsics = np.array([ cc.intrinsics for cc in calibrations ])
    mean_distortions =  np.array([ cc.undistortion_magnitudes().mean() for cc in calibrations ])
    max_distortions =  np.array([ cc.undistortion_magnitudes().max() for cc in calibrations ])

    plt.plot(zoom_stops, max_distortions, 'o-', color='#CC2529', label='max', markersize=5, linewidth=2)
    plt.plot(zoom_stops, mean_distortions, 'D-', color='#DA7C30', label='mean', markersize=4, linewidth=2)
    plt.legend()
    plt.xticks([18, 24, 28, 31, 35, 40, 44, 50, 55, 60, 65, 70])
    plt.xlim( (15, 75) )
    plt.xlabel('lens zoom setting')
    plt.ylabel('pixels')
    plt.grid(b=True, which='major', color='#ededed', linestyle='-')
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()