#! /usr/bin/python

import sys
import os
import glob
import numpy as np
import cPickle as pickle
from matplotlib import pyplot as plt

from camera_math import matrix_to_intrinsics



def main():
    with open(sys.argv[1] + '/intrinsics.model') as f:
        data, focus_model, cx_model, cy_model = pickle.load(f)

    zoom, fx, fy, cx, cy = data.T

    calibration_data = []
    for filename in glob.iglob(sys.argv[1] + '/*/pose0.lh0+'):
        filestem = os.path.splitext(filename)[0]
        itag, etag = filestem.split('/')[-2:]

        with open(filename) as f:
            K, E = pickle.load(f)

        k = matrix_to_intrinsics(K)
        calibration_data.append([float(itag), k[0], k[1], k[2], k[3] ])

    calibration_data = np.vstack(calibration_data)


    plt.subplot(221)
    plt.title('fx')
    plt.plot(zoom, fx, '.', alpha=0.2, color='#d95f02', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,1], 'kx')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(focus_model, r)
    plt.plot(r, p, '-', alpha=0.9, color='#7570b3')
    plt.ylabel('pixels')
    plt.xlim( (10, 80) )
    plt.ylim( (0, 16000) )
    plt.setp( plt.gca().get_xticklabels(), visible=False )
    plt.grid()

    plt.subplot(222)
    plt.title('fy')
    plt.plot(zoom, fy, '.', alpha=0.2, color='#d95f02', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,2], 'kx')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(focus_model, r)
    plt.plot(r, p, '-', alpha=0.9, color='#7570b3')
    plt.xlim( (10, 80) )
    plt.ylim( (0, 16000) )
    plt.setp( plt.gca().get_yticklabels(), visible=False )
    plt.setp( plt.gca().get_xticklabels(), visible=False )
    plt.grid()

    plt.subplot(223)
    plt.title('cx')
    plt.plot(zoom, cx, '.', alpha=0.2, color='#d95f02', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,3], 'kx')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(cx_model, r)
    plt.plot(r, p, '-', alpha=0.9, color='#7570b3')
    plt.xlim( (10, 80) )
    plt.ylim( (800, 1600) )
    plt.xlabel('lens zoom setting')
    plt.ylabel('pixels')
    plt.grid()

    plt.subplot(224)
    plt.title('cy')
    plt.plot(zoom, cy, '.', alpha=0.2, color='#d95f02', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,4], 'kx')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(cy_model, r)
    plt.plot(r, p, '-', alpha=0.9, color='#7570b3')
    plt.xlim( (10, 80) )
    plt.ylim( (800, 1600) )
    plt.xlabel('lens zoom setting')
    plt.setp( plt.gca().get_yticklabels(), visible=False )
    plt.grid()

    plt.show()

if __name__ == '__main__':
    main()