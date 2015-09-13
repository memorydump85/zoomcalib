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
    plt.plot(calibration_data[:,0], calibration_data[:,1], 'k+')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(focus_model, r)
    plt.plot(r, p, '-', alpha=0.7, color='#7570b3', linewidth=2)
    plt.ylabel('pixels')
    plt.xticks([18, 20, 24, 28, 31, 35, 40, 44, 50, 55, 60, 65, 70])
    plt.xlim( (10, 80) )
    plt.ylim( (0, 10000) )
    plt.setp( plt.gca().get_xticklabels(), visible=False )
    plt.grid(b=True, which='major', color='#ededed', linestyle='-')

    plt.subplot(222)
    plt.title('fy')
    plt.plot(zoom, fy, '.', alpha=0.2, color='#d95f02', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,2], 'k+')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(focus_model, r)
    plt.plot(r, p, '-', alpha=0.7, color='#7570b3', linewidth=2)
    plt.xticks([18, 20, 24, 28, 31, 35, 40, 44, 50, 55, 60, 65, 70])
    plt.xlim( (10, 80) )
    plt.ylim( (0, 10000) )
    plt.setp( plt.gca().get_yticklabels(), visible=False )
    plt.setp( plt.gca().get_xticklabels(), visible=False )
    plt.grid(b=True, which='major', color='#ededed', linestyle='-')

    plt.subplot(223)
    plt.title('cx')
    plt.plot(zoom, cx, '.', alpha=0.2, color='#d95f02', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,3], 'k+')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(cx_model, r)
    plt.plot(r, p, '-', alpha=0.7, color='#7570b3', linewidth=2)
    plt.xticks([18, 20, 24, 28, 31, 35, 40, 44, 50, 55, 60, 65, 70])
    plt.xlim( (10, 80) )
    plt.ylim( (800, 1700) )
    plt.xlabel('lens zoom setting')
    plt.ylabel('pixels')
    plt.grid(b=True, which='major', color='#ededed', linestyle='-')

    plt.subplot(224)
    plt.title('cy')
    plt.plot(zoom, cy, '.', alpha=0.2, color='#d95f02', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,4], 'k+')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(cy_model, r)
    plt.plot(r, p, '-', alpha=0.7, color='#7570b3', linewidth=2)
    plt.xticks([18, 20, 24, 28, 31, 35, 40, 44, 50, 55, 60, 65, 70])
    plt.xlim( (10, 80) )
    plt.ylim( (800, 1700) )
    plt.xlabel('lens zoom setting')
    plt.setp( plt.gca().get_yticklabels(), visible=False )
    plt.grid(b=True, which='major', color='#ededed', linestyle='-')

    plt.show()

if __name__ == '__main__':
    main()