#! /usr/bin/python

import sys
import glob
import numpy as np
import cPickle as pickle
from scipy.optimize import minimize



def load_intrinsics_sample_data(filename):
    with open(filename) as f:
        samples = pickle.load(f)

    zoom_stop = int(filename.split('/')[-2])
    zoom_stop_array = np.reshape([zoom_stop]*len(samples), (-1, 1))
    return np.hstack(( zoom_stop_array, samples ))


def robust_regression(x, t, degree):

    def mean_squared_error(w, x, t):
        y = np.polyval(w, x)
        return np.mean((y - t)**2)

    def median_squared_error(w, x, t):
        y = np.polyval(w, x)
        return np.median((y - t)**2)

    def huber_loss(w, x, t, delta):
        y = np.polyval(w, x)
        abserr = np.abs(y-t)
        return np.where(abserr <= delta, abserr**2/2., delta*(abserr - delta/2.)).sum()

    def pseudo_huber_loss(w, x, t, delta):
        y = np.polyval(w, x)
        abserr = np.abs(y-t)
        return (delta**2*(np.sqrt(1 + (abserr/delta)**2) - 1)).sum()

    w0 = [0.] * (degree+1)
    poly_model = minimize(mean_squared_error, x0=w0, args=(x, t), method="Powell").x
    return poly_model


def main():
    folder = sys.argv[1]
    data = []
    for subfolder in glob.iglob(folder + '/*/intrinsics.samples'):
        data.append( load_intrinsics_sample_data(subfolder) )

    data = np.vstack(data)

    zoom, fx, fy, cx, cy = np.vstack(data).T
    zoomzoom = np.hstack(( zoom, zoom ))
    fxfy = np.hstack(( fx, fy ))

    focus_model = robust_regression(zoomzoom, fxfy, 4)
    cx_model = robust_regression(zoom, cx, 6)
    cy_model = robust_regression(zoom, cy, 6)

    print focus_model
    print cx_model
    print cy_model

    with open(folder + '/intrinsics.model', 'w') as f:
        pickle.dump((data, focus_model, cx_model, cy_model), f)

if __name__ == '__main__':
    main()
