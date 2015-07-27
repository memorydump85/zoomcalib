import numpy as np
from sklearn.preprocessing import StandardScaler

from gp import GaussianProcess, sqexp2D_covariancef


class WhiteningScaler(object):
    def fit(self, X):
        self.mu_ = X.mean(axis=0)
        U, s, V = np.linalg.svd(X.T.dot(X))
        self.xform_ = np.diag(s).dot(V.T)
        self.inv_xform_ = V.dot(np.diag(1./s))

    def transform(self, X):
        centered = X - np.tile(self.mu_, (len(X), 1))
        return self.xform_.dot(centered.T).T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        scaled = self.inv_xform_.dot(X.T).T
        return scaled + np.tile(self.mu_, (len(X), 1))


def main():
    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.util import img_as_ubyte

    im = imread('/var/tmp/datasets/tamron-2.2/im000.png')
    im = rgb2gray(im)
    im = img_as_ubyte(im)

    image_points = np.load('image_points.npy')
    undistortion = np.load('undistortion.npy')

    mean_u = undistortion.mean(axis=0)
    u = undistortion - np.tile(mean_u, (len(undistortion), 1))

    cov = np.cov(image_points.T)
    theta0_ux = [u[:,0].std(), np.sqrt(cov[0,0]), np.sqrt(cov[1,1]), cov[1,0], 10 ]
    gp_ux = GaussianProcess.fit(image_points, u[:,0], sqexp2D_covariancef, theta0_ux)
    print gp_ux.covf.theta
    theta0_uy = [u[:,1].std(), np.sqrt(cov[0,0]), np.sqrt(cov[1,1]), cov[1,0], 10 ]
    gp_uy = GaussianProcess.fit(image_points, u[:,1], sqexp2D_covariancef, theta0_uy)
    print gp_ux.covf.theta

    H, W = im.shape
    grid = np.array([[x, y] for y in xrange(0, H, 20) for x in xrange(0, W, 20)])
    U = gp_ux.predict(grid) + mean_u[0]
    V = gp_uy.predict(grid) + mean_u[1]
    from matplotlib import pyplot as plt
    plt.quiver(grid[:,0], grid[:,1], U, V, units='xy', color='r', width=1)
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    main()