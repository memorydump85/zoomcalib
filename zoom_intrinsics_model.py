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

    def median_squared_error(w, x, t):
        y = np.polyval(w, x)
        return np.median((y - t)**2)

    linear_model = minimize(median_squared_error, x0=[0., 0.], args=(x, t), method="Powell").x
    if degree == 1:
        return linear_model

    w0 = [0.] * (degree+1)
    w0[-1] = linear_model[0]
    w0[-2] = linear_model[1]
    poly_model = minimize(median_squared_error, x0=w0, args=(x, t), method="Powell").x
    return poly_model


def main():
    folder = sys.argv[1]
    data = []
    for subfolder in glob.iglob(folder + '/*/intrinsics.samples'):
        data.append( load_intrinsics_sample_data(subfolder) )

    data = np.vstack(data)

    zoom, fx, fy, cx, cy = np.vstack(data).T
    zoomzoom = np.vstack(( zoom, zoom ))
    fxfy = np.vstack(( fx, fy ))

    focus_model = robust_regression(zoomzoom, fxfy, 2)
    cx_model = robust_regression(zoom, cx, 1)
    cy_model = robust_regression(zoom, cy, 1)

    with open(folder + '/intrinsics.model', 'w') as f:
        pickle.dump((data, focus_model, cx_model, cy_model), f)

if __name__ == '__main__':
    main()
