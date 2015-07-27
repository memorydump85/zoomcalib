import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler

from gp import GaussianProcess, sqexp2D_covariancef



#--------------------------------------
class GPDistortionModel(object):
#--------------------------------------
    @classmethod
    def fit(cls, im_points, distortions):
        assert len(im_points) == len(distortions)

        X = im_points
        S = np.cov(X.T)
                
        meanD = np.mean(distortions, axis=0)
        D = distortions - np.tile(meanD, (len(distortions), 1))

        theta0 = D[:,0].std(), sqrt(S[0,0]), sqrt(S[1,1]), S[1,0], 10.
        gp_dx = GaussianProcess.fit(X, D[:,0], sqexp2D_covariancef, theta0)

        theta0 = D[:,1].std(), sqrt(S[0,0]), sqrt(S[1,1]), S[1,0], 10.
        gp_dy = GaussianProcess.fit(X, D[:,1], sqexp2D_covariancef, theta0)

        this = GPDistortionModel()
        this.meanD_ = meanD
        this.gp_dx_ = gp_dx
        this.gp_dy_ = gp_dy

        return this


    def predict(self, X):
        D = np.vstack([ self.gp_dx_.predict(X), self.gp_dy_.predict(X) ]).T
        return D + np.tile(self.meanD_, (len(X), 1))



def main():
    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.util import img_as_ubyte

    im = imread('/var/tmp/datasets/tamron-2.2/im000.png')
    im = rgb2gray(im)
    im = img_as_ubyte(im)

    image_points = np.load('image_points.npy')
    distortion = -np.load('undistortion.npy')

    model = GPDistortionModel.fit(image_points, distortion)

    H, W = im.shape
    grid = np.array([[x, y] for y in xrange(0, H, 20) for x in xrange(0, W, 20)])
    predicted = model.predict(grid)
    U, V = predicted[:,0], predicted[:,1]

    from matplotlib import pyplot as plt
    plt.quiver(grid[:,0], grid[:,1], U, V, units='xy', color='r', width=1)
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    main()