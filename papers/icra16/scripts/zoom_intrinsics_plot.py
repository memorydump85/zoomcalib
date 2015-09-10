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


    def distortion_magnitudes(self):
        uv = self.undistortion[:,2:]
        return np.linalg.norm(uv, axis=1)


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
    plt.figure(figsize=(8,8))

    zoom_stops = [ float(cc.itag) for cc in calibrations ]
    intrinsics = np.array([ cc.intrinsics for cc in calibrations ])
    max_distortion = [ cc.distortion_magnitudes().max() for cc in calibrations ]
    mean_distortion = [ cc.distortion_magnitudes().mean() for cc in calibrations ]

    plt.subplot(221)
    plt.plot(zoom_stops, intrinsics[:,0], 'o-', color='#7570b3', label='fx', linewidth=2)
    plt.plot(zoom_stops, intrinsics[:,1], 'o-', color='#d95f02', label='fy', linewidth=2)
    plt.ylabel('focal length (pixels)')
    plt.legend(loc='upper left')
    plt.setp( plt.gca().get_xticklabels(), visible=False)
    plt.grid()

    plt.subplot(222)
    plt.ylabel('distortion (pixels)')
    plt.plot(zoom_stops, mean_distortion, 'o-', color='#7570b3', linewidth=2)
    plt.setp( plt.gca().get_xticklabels(), visible=False)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.grid()

    plt.subplot(223)
    plt.xlabel('zoom stops')
    plt.ylabel('cx (pixels)')
    plt.plot(zoom_stops, intrinsics[:,2], 'o-', color='#7570b3', linewidth=2)
    plt.grid()

    plt.subplot(224)
    plt.xlabel('zoom stops')
    plt.ylabel('cy (pixels)')
    plt.plot(zoom_stops, intrinsics[:,3], 'o-', color='#7570b3', linewidth=2)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()