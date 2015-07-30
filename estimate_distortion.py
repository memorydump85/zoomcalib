import numpy as np
from math import sqrt
from itertools import chain
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from scipy.optimize import minimize, minimize_scalar
from sklearn.preprocessing import StandardScaler

from apriltag import AprilTagDetector
from projective_math import WeightedLocalHomography, SqExpWeightingFunction
from tag36h11_mosaic import TagMosaic
from gp import GaussianProcess, sqexp2D_covariancef



np.set_printoptions(precision=4)


def main():
    #
    # Conventions:
    # a_i, b_i
    #    are variables in image space, units are pixels
    # a_w, b_w
    #    are variables in world space, units are meters
    #
    
    im = imread('/var/tmp/datasets/tamron-2.2/im001.png')
    im = rgb2gray(im)
    im = img_as_ubyte(im)
    
    c_i = np.array([im.shape[1], im.shape[0]]) / 2.

    tag_mosaic = TagMosaic(0.0254)
    detections = AprilTagDetector().detect(im)

    # Sort detections by distance to center
    dist = lambda p_i: np.linalg.norm(p_i - c_i)
    closer_to_center = lambda d1, d2: int(dist(d1.c) - dist(d2.c))
    detections.sort(cmp=closer_to_center)

    # Space out the validation samples equally in theta
    v_ixs = range(len(detections))
    theta = lambda index: np.arctan2(*(detections[index].c-c_i))
    v_ixs.sort(key=theta)
    v_ixs = v_ixs[11::7] + [0] # include detection closest to center
    detections_validate = [ detections[i] for i in v_ixs ]

    t_ixs = set(range(len(detections))) - set(v_ixs)
    detections_train = [ detections[i] for i in t_ixs ]

    #
    # In the following section of we learn a bunch of homographies.
    #   First, we learn a homography from image to world
    #   Next, find where the image center `c_i` projects to in world coordinates (`c_w`)
    #   Finally, find the local homography `LH0` from world to image at `c_w`
    #
    # To learn the homography, we find the weighting function parameter
    # that minimizes reprojection error on the validation points.
    #
    mosaic_pos = lambda det: tag_mosaic.get_position_meters(det.id)

    train_i = np.array([ d.c for d in detections_train ])
    train_w = np.array([ mosaic_pos(d) for d in detections_train ])
    validate_i = np.array([ d.c for d in detections_validate ])
    validate_w = np.array([ mosaic_pos(d) for d in detections_validate ])

    def create_local_homography_object(bandwidth):
        H = WeightedLocalHomography(SqExpWeightingFunction(bandwidth=bandwidth))
        H.regularization_lambda = 1e-8
        return H

    def local_homography_error(bandwidth, t_src, t_tgt, v_src, v_tgt):
        """ 
        This is the objective function used for optimizing the `bandwidth`
        parameter of the `SqExpWeightingFunction`

        Parameters:
        -----------
            `t_src`, `t_tgt`: lists of training source and target points
            `v_src`, `v_tgt`: lists of validation source and target points
        """
        H = create_local_homography_object(bandwidth)
        for s, t in zip(t_src, t_tgt):
            H.add_correspondence(s, t)

        sqerr = lambda a, b: np.linalg.norm(a-b)**2
        sse = 0.
        for s, t in zip(v_src, v_tgt):
            m = H.map(s)[:2]
            sse += sqerr(m, t)

        N = len(v_src)
        return np.sqrt(sse/N)

    def learn_homography_i2w():
        result = minimize_scalar( local_homography_error,
                    method='Bounded',
                    bounds=(1e-4, 1e4),
                    args=(train_i, train_w, validate_i, validate_w),
                    tol=1e-4 )
        
        print '\nHomography: i->w'
        print '------------------'
        print result

        H = create_local_homography_object(bandwidth=result.x)
        for i, w in zip(chain(train_i, validate_i), chain(train_w, validate_w)):
            H.add_correspondence(i, w)

        return H

    def learn_homography_w2i():
        result = minimize_scalar( local_homography_error,
                    method='Bounded',
                    bounds=(1e-4, 1e4),
                    args=(train_w, train_i, validate_w, validate_i),
                    tol=1e-4 )
        
        print '\nHomography: w->i'
        print '------------------'
        print result
        
        H = create_local_homography_object(bandwidth=result.x)
        for w, i in zip(chain(train_w, validate_w), chain(train_i, validate_i)):
            H.add_correspondence(w, i)

        return H

    # Map center `c_i` to world coords to get `c_w`.
    # Then compute homography at `c_w`
    H_iw = learn_homography_i2w()
    c_w = H_iw.map(c_i)[:2]
    H_wi = learn_homography_w2i()
    LH0 = H_wi.get_homography_at(c_w)

    def homogeneous_coords(p):
        assert len(p)==2 or len(p)==3
        return p if len(p)==3 else np.array([p[0], p[1], 1.])

    print '\nHomography at center'
    print '----------------------'
    print '      c_w =', c_w
    print '      c_i =', c_i
    print 'LH0 * c_w =', H_wi.map(c_w)

    world_points = np.array([ homogeneous_coords(mosaic_pos(d)) for d in detections ])
    mapped_points = LH0.dot(world_points.T).T
    mapped_points = np.array([ p / p[2] for p in mapped_points ])
    mapped_points = mapped_points[:,:2]

    image_points = np.array([ d.c for d in detections ])
    undistortion = mapped_points - image_points # image + undistortion = mapped
    
    #--------------------------------------
    class GPModel(object):
    #--------------------------------------
        def __init__(self, im_points, values):
            assert len(im_points) == len(values)

            X = im_points
            S = np.cov(X.T)
                    
            meanV = np.mean(values, axis=0)
            V = values - np.tile(meanV, (len(values), 1))

            theta0 = V[:,0].std(), sqrt(S[0,0]), sqrt(S[1,1]), S[1,0], 10.
            gp_x = GaussianProcess.fit(X, V[:,0], sqexp2D_covariancef, theta0)

            theta0 = V[:,1].std(), sqrt(S[0,0]), sqrt(S[1,1]), S[1,0], 10.
            gp_y = GaussianProcess.fit(X, V[:,1], sqexp2D_covariancef, theta0)

            self.meanV_ = meanV
            self.gp_x_ = gp_x
            self.gp_y_ = gp_y

        def predict(self, X):
            V = np.vstack([ self.gp_x_.predict(X), self.gp_y_.predict(X) ]).T
            return V + np.tile(self.meanV_, (len(X), 1))

    model = GPModel(image_points, undistortion)
    print '\nGP Hyper-parameters'
    print '---------------------'
    print '  x: ', model.gp_x_.covf.theta
    print '  y: ', model.gp_y_.covf.theta


    if True:
        from skimage.filters import scharr
        from matplotlib import pyplot as plt

        plt.subplot(221)
        plt.imshow(np.flipud(im), cmap='bone')

        plt.subplot(223)
        plt.title('Estimates')
        h1, = plt.plot(train_i[:,0], train_i[:,1], 'b+')
        h2, = plt.plot(validate_i[:,0], validate_i[:,1], 'bo')
        plt.legend([h1, h2], ['train', 'validate'], fontsize='xx-small')

        X, Y = image_points[:,0], image_points[:,1]
        U, V = undistortion[:,0], undistortion[:,1]
        plt.quiver(X, Y, U, V, units='xy', color='r', width=2)
        plt.axis('equal')

        plt.subplot(224)
        plt.title('Undistortion Model')
        H, W = im.shape
        grid = np.array([[x, y] for y in xrange(0, H, 20) for x in xrange(0, W, 20)])
        predicted = model.predict(grid)
        X, Y = grid[:,0], grid[:,1]
        U, V = predicted[:,0], predicted[:,1]
        plt.quiver(X, Y, U, V, units='xy', color='r', width=2)
        plt.axis('equal')        

        plt.show()


if __name__ == '__main__':
    main()