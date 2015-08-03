import numpy as np
from math import sqrt
from itertools import chain
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.optimize import minimize, minimize_scalar
from sklearn.preprocessing import StandardScaler

from apriltag import AprilTagDetector
from projective_math import WeightedLocalHomography, SqExpWeightingFunction
from tag36h11_mosaic import TagMosaic
from gp import GaussianProcess, sqexp2D_covariancef



np.set_printoptions(precision=4, suppress=True)


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
    
    c_i = np.array([im.shape[1], im.shape[0]]) / 2.

    tag_mosaic = TagMosaic(0.0254)
    detections = AprilTagDetector().detect(im)
    print '  %d tags detected.' % len(detections)

    # Sort detections by distance to center
    dist = lambda p_i: np.linalg.norm(p_i - c_i)
    closer_to_center = lambda d1, d2: int(dist(d1.c) - dist(d2.c))
    detections.sort(cmp=closer_to_center)

    # 1 in 10 detections is used for validation
    # We also want to keep the detection closest to the center for validation
    v_ixs = range(len(detections))
    v_ixs = np.random.choice(v_ixs[10:], len(detections)/10, replace=False).tolist()
    v_ixs += [0] # detection closest to center
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
        H.regularization_lambda = 1e-3
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
                    options={'xatol': 1e-4} )
        
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
                    options={'xatol': 1e-4} )
        
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


    max_distortion = np.max([np.linalg.norm(u) for u in undistortion])
    print '\nMaximum distortion is %.2f pixels' % max_distortion
    

    #--------------------------------------
    class GPModel(object):
    #--------------------------------------
        def __init__(self, im_points, values):
            assert len(im_points) == len(values)

            X = im_points
            S = np.cov(X.T)
                    
            meanV = np.mean(values, axis=0)
            V = values - np.tile(meanV, (len(values), 1))

            theta0 = np.array(( V[:,0].std(), sqrt(S[0,0]), sqrt(S[1,1]), S[1,0], 10. ))
            gp_x = GaussianProcess.fit(X, V[:,0], sqexp2D_covariancef, theta0)

            theta0 = np.array(( V[:,1].std(), sqrt(S[0,0]), sqrt(S[1,1]), S[1,0], 10. ))
            gp_y = GaussianProcess.fit(X, V[:,1], sqexp2D_covariancef, theta0)

            self.meanV_ = meanV
            self.gp_x_ = gp_x
            self.gp_y_ = gp_y

        def predict(self, X):
            V = np.vstack([ self.gp_x_.predict(X), self.gp_y_.predict(X) ]).T
            return V + np.tile(self.meanV_, (len(X), 1))

    
    model = GPModel(image_points, undistortion)
    
    import textwrap
    print '\nGP Hyper-parameters'
    print '---------------------'
    print '  x: ', model.gp_x_.covf.theta
    print '        log-likelihood: %.4f' % model.gp_x_.model_evidence()
    print '  y: ', model.gp_y_.covf.theta
    print '        log-likelihood: %.4f' % model.gp_y_.model_evidence()
    print ''
    print '  Optimization detail:'
    print '  [ x ]'
    print str(model.gp_x_.fit_result_).replace('\n', '\n      ')
    print '  [ y ]'
    print str(model.gp_y_.fit_result_).replace('\n', '\n      ')


    #   Visualization
    if True:
        from skimage.filters import scharr
        from matplotlib import pyplot as plt

        plt.style.use('ggplot')

        #__1__
        plt.subplot(221)
        plt.title(filename.split('/')[-1])
        plt.imshow(im, cmap='gray')
        plt.plot(image_points[:,0], image_points[:,1], 'o')
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
            x_coords = np.linspace(a[0], b[0], 100)
            points = np.array([ H_wi.map([x, a[1]]) for x in x_coords ])
            plt.plot(points[:,0], points[:,1], '-',color='#CF4457', linewidth=2)

        for k, v in col_groups.iteritems():
            a = tag_mosaic.get_position_meters(np.min(v))
            b = tag_mosaic.get_position_meters(np.max(v))
            y_coords = np.linspace(a[1], b[1], 100)
            points = np.array([ H_wi.map([a[0], y]) for y in y_coords ])
            plt.plot(points[:,0], points[:,1], '-',color='#CF4457', linewidth=2)
        
        plt.plot(image_points[:,0], image_points[:,1], 'kx')
        plt.grid()
        plt.axis('equal')

        #__3__
        plt.subplot(223)
        plt.title('Estimates')

        for i, d in enumerate(detections):
            plt.text(d.c[0], d.c[1], str(i), fontsize=8, color='#999999')

        h1, = plt.plot(train_i[:,0], train_i[:,1], '+')
        h2, = plt.plot(validate_i[:,0], validate_i[:,1], 'o')
        plt.legend([h1, h2], ['train', 'validate'], fontsize='xx-small')

        X, Y = image_points[:,0], image_points[:,1]
        U, V = undistortion[:,0], undistortion[:,1]
        plt.quiver(X, Y, U, -V, units='dots', width=1)
        plt.gca().invert_yaxis()
        plt.axis('equal')

        #__4__
        plt.subplot(224)
        plt.title('Undistortion Model')
        H, W = im.shape
        grid = np.array([[x, y] for y in xrange(0, H, 80) for x in xrange(0, W, 80)])
        predicted = model.predict(grid)
        X, Y = grid[:,0], grid[:,1]
        U, V = predicted[:,0], predicted[:,1]
        plt.quiver(X, Y, U, -V, units='dots', color='#CF4457', width=1)
        plt.gca().invert_yaxis()
        plt.axis('equal')        

        plt.tight_layout()
        plt.gcf().patch.set_facecolor('#dddddd')
        plt.show()


def main():
    from glob import iglob
    for filename in iglob('/var/tmp/capture/29.png'):
        process(filename)

if __name__ == '__main__':
    main()