import numpy as np
from numpy import sqrt
from numpy.linalg.linalg import LinAlgError
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.optimize import minimize

from apriltag import AprilTagDetector
from projective_math import Homography
from tag36h11_mosaic import TagMosaic
from gp import GaussianProcess, sqexp2D_covariancef



np.set_printoptions(precision=4, suppress=True)


#--------------------------------------
class WorldImageNonLinearMapping(object):
#--------------------------------------
    def __init__(self, X, values):
        assert len(X) == len(values)

        S = np.cov(X.T)

        meanV = np.mean(values, axis=0)
        V = values - np.tile(meanV, (len(values), 1))

        self._meanV = meanV
        self._gp_x = WorldImageNonLinearMapping._fit_gp(X, S, V[:,0])
        self._gp_y = WorldImageNonLinearMapping._fit_gp(X, S, V[:,1])


    @staticmethod
    def _fit_gp(X, covX, t):
        xx, xy, yy = covX[0,0], covX[0,1], covX[1,1]

        # Perform hyper-parameter optimization with different
        # initial points and choose the GP with best model evidence
        theta0 = ( t.std(), sqrt(xx), sqrt(yy), xy, 10. )
        best_gp = GaussianProcess.fit(X, t, sqexp2D_covariancef, theta0)
        print '          : %.4f' % best_gp.model_evidence()

        for tau in np.linspace(0, 3, 50) + 0.1:
            try:
                theta0 = ( t.std(), tau, tau, 0, 10. )
                gp = GaussianProcess.fit(X, t, sqexp2D_covariancef, theta0)
                if gp.model_evidence() > best_gp.model_evidence():
                    best_gp = gp
                    print '    %.4f: %.4f' % (tau, best_gp.model_evidence())
            except LinAlgError as err:
                print '    %.4f: %s' % (tau, str(err))


        return best_gp


    def predict(self, X):
        V = np.vstack([ self._gp_x.predict(X), self._gp_y.predict(X) ]).T
        return V + np.tile(self._meanV, (len(X), 1))


    def __call__(self, X):
        return self.predict(X)


def process(filename):
    #
    # Conventions:
    # a_i, b_i
    #    are variables in image space, units are pixels
    # a_w, b_w
    #    are variables in world space, units are meters
    #
    print '\n========================================'
    print '  File: ' + filename
    print '========================================\n'

    im = imread(filename)
    im = rgb2gray(im)
    im = (im * 255.).astype(np.uint8)

    tag_mosaic = TagMosaic(0.0254)
    detections = AprilTagDetector().detect(im)
    print '  %d tags detected.' % len(detections)

    mosaic_pos = lambda det: tag_mosaic.get_position_meters(det.id)

    #
    # Non-parametric mapping from world to image
    #
    det_w = np.array([ mosaic_pos(d) for d in detections ])
    det_i = np.array([ d.c for d in detections ])
    mapping_wi = WorldImageNonLinearMapping(det_w, det_i)

    #
    # Find the image center in world coordinates `c_w`.
    # we compute `c_w` as:
    #
    #    c_w = minimize dist( mapping_wi(p_w), c_i )
    #            p_w
    #
    c_i = np.array([im.shape[1], im.shape[0]]) / 2.
    mse = lambda p_w: ((c_i - mapping_wi([p_w])[0])**2).mean()
    result = minimize(mse, x0=[0., 0.], method='Powell')
    c_w = result.x
    rmse = np.sqrt(((mapping_wi(det_w) - det_i) ** 2).mean(axis=0))

    print '\nCenter calculation'
    print '--------------------'
    print '              c_w  =', c_w
    print '              c_i  =', c_i
    print '   mapping_wi(c_w) =', mapping_wi([c_w])[0]
    print '             rmse  =', rmse


    #
    # Compute `H0`, the homography at center
    #
    offsets = np.array([ [+1,+1], [+1,-1], [-1,-1], [-1,+1] ]) * 0.01
    a_w = offsets + c_w
    a_i = mapping_wi(a_w)

    H0 = Homography()
    for w, i in zip(a_w, a_i): H0.add_correspondence(w, i)

    linear_mapped = np.array([ H0.map(w)[:2] for w in det_w ])
    nonlinear_mapped = mapping_wi(det_w)
    undistortion = linear_mapped - det_i
    max_distortion = np.max([np.linalg.norm(u) for u in undistortion])
    print '\nMaximum distortion is %.2f pixels' % max_distortion


    #
    # Visualization
    #
    from skimage.filters import scharr
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    plt.figure(figsize=(16,10))

    #__1__
    plt.subplot(221)
    plt.title(filename.split('/')[-1])
    plt.imshow(im, cmap='gray')
    plt.plot(det_i[:,0], det_i[:,1], 'o')
    # for i, d in enumerate(detections):
    #     plt.text(d.c[0], d.c[1], str(d.id),
    #         fontsize=8, color='white', bbox=dict(facecolor='maroon', alpha=0.75))
    plt.grid()
    plt.axis('equal')

    #__2__
    plt.subplot(222)
    plt.title('Non-linear Homography')

    plt.imshow(1.-scharr(im), cmap='bone', interpolation='gaussian')

    from collections import defaultdict
    row_groups = defaultdict(list)
    col_groups = defaultdict(list)

    for d in detections:
        row, col = tag_mosaic.get_row(d.id), tag_mosaic.get_col(d.id)
        row_groups[row] += [ d.id ]
        col_groups[col] += [ d.id ]

    for k, v in row_groups.iteritems():
        a = tag_mosaic.get_position_meters(np.min(v))
        b = tag_mosaic.get_position_meters(np.max(v))
        x_coords = np.linspace(a[0]-0.0254, b[0]+0.0254, 500)
        y_coords = np.repeat(a[1], len(x_coords))
        points =  mapping_wi(np.vstack((x_coords, y_coords)).T)
        plt.plot(points[:,0], points[:,1], '-',color='#CF4457', linewidth=1)

    for k, v in col_groups.iteritems():
        a = tag_mosaic.get_position_meters(np.min(v))
        b = tag_mosaic.get_position_meters(np.max(v))
        y_coords = np.linspace(a[1]+0.0254, b[1]-0.0254, 500)
        x_coords = np.repeat(a[0], len(y_coords))
        points =  mapping_wi(np.vstack((x_coords, y_coords)).T)
        plt.plot(points[:,0], points[:,1], '-',color='#CF4457', linewidth=1)

    plt.plot(det_i[:,0], det_i[:,1], 'kx')
    plt.grid()
    plt.text( 0.5, 0, 'rmse: [ %.2f %.2f ] px' % (rmse[0], rmse[1]), color='#CF4457', fontsize=10,
        horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.axis('equal')

    #__3__
    plt.subplot(223)
    plt.title('Qualitative Estimates')

    for i, d in enumerate(detections):
        plt.text(d.c[0], d.c[1], str(i), fontsize=8, color='#999999')

    X, Y = det_i[:,0], det_i[:,1]
    U, V = undistortion[:,0], undistortion[:,1]
    plt.quiver(X, Y, U, -V, units='dots')
    # plt.quiver(X, Y, U, -V, angles='xy', scale_units='xy', scale=1) # plot exact
    plt.text( 0.5, 0, 'max observed distortion: %.2f px' % max_distortion, color='#CF4457', fontsize=10,
        horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.gca().invert_yaxis()
    plt.axis('equal')

    plt.tight_layout()
    plt.gcf().patch.set_facecolor('#dddddd')
    plt.show()


def main():
    # 18, 20, 24, 26, 29, 35, 38, 44, 50, 56, 60, 70
    from glob import iglob
    for filename in iglob('/var/tmp/capture/44.png'):
        process(filename)

if __name__ == '__main__':
    main()
