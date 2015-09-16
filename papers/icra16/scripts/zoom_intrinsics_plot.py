#! /usr/bin/python

import sys
import os
import glob
import numpy as np
import cPickle as pickle
import matplotlib
from matplotlib import pyplot as plt

from camera_math import matrix_to_intrinsics



def main():
    matplotlib.rcParams.update({'font.size': 10})

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

    plt.figure(figsize=(6,8))

    plt.subplot(211)
    plt.title('focal lengths')
    plt.grid(b=True, which='major', color='#ededed', linestyle='-')
    plt.plot(zoom, fy, '.', color='#DA7C30', markersize=3)
    plt.plot(zoom, fx, '.', color='#CC2529', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,1], 'k+')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(focus_model, r)
    plt.plot(r, p, '-', color='#396AB1', linewidth=2)
    plt.ylabel('pixels')
    plt.xticks([18, 24, 28, 31, 35, 40, 44, 50, 55, 60, 65, 70])
    plt.xlim( (15, 75) )
    plt.ylim( (0, 10000) )
    plt.setp( plt.gca().get_xticklabels(), visible=False )
    plt.gca().set_axisbelow(True)

    # plt.subplot(312)
    # plt.title('fy')
    # plt.grid(b=True, which='major', color='#ededed', linestyle='-')
    # plt.plot(zoom, fy, '.', color='#DA7C30', markersize=3)
    # plt.plot(calibration_data[:,0], calibration_data[:,2], 'k+')
    # r = np.arange(18, 70, 0.1)
    # p = np.polyval(focus_model, r)
    # plt.plot(r, p, '-', color='#396AB1', linewidth=2)
    # plt.xticks([18, 20, 24, 28, 31, 35, 40, 44, 50, 55, 60, 65, 70])
    # plt.xlim( (10, 80) )
    # plt.ylim( (0, 10000) )
    # plt.setp( plt.gca().get_yticklabels(), visible=False )
    # plt.setp( plt.gca().get_xticklabels(), visible=False )

    plt.subplot(212)
    plt.title('principal point')
    plt.grid(b=True, which='major', color='#ededed', linestyle='-')
    plt.plot(zoom, cx, '.', color='#CC2529', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,3], 'k+')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(cx_model, r)
    plt.plot(r, p, '-', color='#396AB1', linewidth=2)
    plt.text(18, 1350, 'cx', fontdict={'size': 10})
    plt.xticks([18, 24, 28, 31, 35, 40, 44, 50, 55, 60, 65, 70])
    plt.xlim( (15, 75) )
    plt.ylim( (800, 1700) )
    plt.xlabel('lens zoom setting')
    plt.ylabel('pixels')

    plt.subplot(212)
    plt.grid(b=True, which='major', color='#ededed', linestyle='-')
    plt.plot(zoom, cy, '.', color='#DA7C30', markersize=3)
    plt.plot(calibration_data[:,0], calibration_data[:,4], 'k+')
    r = np.arange(18, 70, 0.1)
    p = np.polyval(cy_model, r)
    plt.plot(r, p, '-', color='#396AB1', linewidth=2)
    plt.text(18, 1125, 'cy', fontdict={'size': 10})
    plt.xticks([18, 24, 28, 31, 35, 40, 44, 50, 55, 60, 65, 70])
    plt.xlim( (15, 75) )
    plt.ylim( (800, 1700) )
    plt.xlabel('lens zoom setting')
    plt.gca().set_axisbelow(True)
    #plt.setp( plt.gca().get_yticklabels(), visible=False )

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()