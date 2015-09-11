#! /usr/bin/python

import sys
import numpy as np
import cPickle as pickle
from matplotlib import pyplot as plt



def main():
    with open(sys.argv[1] + '/intrinsics.model') as f:
        data, focus_model, cx_model, cy_model = pickle.load(f)

    print focus_model
    print cx_model
    print cy_model
    zoom, fx, fy, cx, cy = data.T

    plt.subplot(221)
    plt.title('fx')
    plt.plot(zoom, fx, '.', alpha=0.2, color='#7570b3', markersize=3)
    r = np.arange(18, 70, 0.1)
    p = np.polyval(focus_model, r)
    plt.plot(r, p, '-', alpha=0.9, color='#d95f02')
    plt.ylabel('pixels')
    plt.xlim( (10, 80) )
    plt.ylim( (0, 16000) )
    plt.setp( plt.gca().get_xticklabels(), visible=False )
    plt.grid()

    plt.subplot(222)
    plt.title('fy')
    plt.plot(zoom, fy, '.', alpha=0.2, color='#7570b3', markersize=3)
    r = np.arange(18, 70, 0.1)
    p = np.polyval(focus_model, r)
    plt.plot(r, p, '-', alpha=0.9, color='#d95f02')
    plt.xlim( (10, 80) )
    plt.ylim( (0, 16000) )
    plt.setp( plt.gca().get_yticklabels(), visible=False )
    plt.setp( plt.gca().get_xticklabels(), visible=False )
    plt.grid()

    plt.subplot(223)
    plt.title('cx')
    plt.plot(zoom, cx, '.', alpha=0.2, color='#7570b3', markersize=3)
    r = np.arange(18, 70, 0.1)
    p = np.polyval(cx_model, r)
    plt.plot(r, p, '-', alpha=0.9, color='#d95f02')
    plt.xlim( (10, 80) )
    plt.ylim( (800, 1600) )
    plt.xlabel('lens zoom setting')
    plt.ylabel('pixels')
    plt.grid()

    plt.subplot(224)
    plt.title('cy')
    plt.plot(zoom, cy, '.', alpha=0.2, color='#7570b3', markersize=3)
    r = np.arange(18, 70, 0.1)
    p = np.polyval(cy_model, r)
    plt.plot(r, p, '-', alpha=0.9, color='#d95f02')
    plt.xlim( (10, 80) )
    plt.ylim( (800, 1600) )
    plt.xlabel('lens zoom setting')
    plt.setp( plt.gca().get_yticklabels(), visible=False )
    plt.grid()

    plt.show()

if __name__ == '__main__':
    main()