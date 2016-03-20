#! /usr/bin/python

import numpy as np
from collections import namedtuple, OrderedDict
from joblib import Memory



def get_tag_detections(im, cache_tag):
    #
    # Because of a bug in the tag detector, it doesn't seem
    # to detect tags larger than a certain size. To work-around
    # this limitation, we detect tags on two different image
    # scales and use the one with more detections
    #
    assert len(im.shape) == 2

    from skimage.transform import rescale as imrescale
    im4 = imrescale(im, 1./4)

    from skimage.util import img_as_ubyte
    im = img_as_ubyte(im)
    im4 = img_as_ubyte(im4)

    from apriltag import AprilTagDetector, AprilTagDetection
    detections1 = AprilTagDetector().detect(im)
    detections4 = AprilTagDetector().detect(im4)
    for d in detections4:
        d.c[0] *= 4.
        d.c[1] *= 4.

    # note that everything other than the tag center is wrong
    # in detections4

    if len(detections4) > len(detections1):
        return detections4
    else:
        return detections1


HomographyInfo = namedtuple('HomographyInfo',
                    ['corrs', 'H', 'imshape'])

nvmem = Memory(cachedir='/tmp/classic_calibration', verbose=1)
@nvmem.cache
def get_homography_estimate(filename):
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

    from skimage.io import imread
    from skimage.color import rgb2gray
    im  = imread(filename)
    im  = rgb2gray(im)

    detections = get_tag_detections(im, cache_tag=filename)
    print '  %d tags detected.' % len(detections)

    #
    # Sort detections by distance to center
    #
    c_i = np.array([im.shape[1], im.shape[0]]) / 2.
    dist = lambda p_i: np.linalg.norm(p_i - c_i)
    closer_to_center = lambda d1, d2: int(dist(d1.c) - dist(d2.c))
    detections.sort(cmp=closer_to_center)

    from tag36h11_mosaic import TagMosaic
    tag_mosaic = TagMosaic(0.0254)
    mosaic_pos = lambda det: tag_mosaic.get_position_meters(det.id)

    det_i = np.array([ d.c for d in detections ])
    det_w = np.array([ mosaic_pos(d) for d in detections ])

    #
    # Improvisation on the classic calibration procedure. We
    # obtain our initial homography estimate from 9 detections
    # at the center. This is better because the distortion is
    # minimal at the center.
    #
    det_i9 = det_i[:9]
    det_w9 = det_w[:9]

    from projective_math import WeightedLocalHomography, UnitWeightingFunction
    H_estimator = WeightedLocalHomography(UnitWeightingFunction())
    for s, t in zip(det_w9, det_i9):
        H_estimator.add_correspondence(s, t)

    from tupletypes import Correspondence
    H = H_estimator.get_homography_at(det_w9[0])
    corrs = [ Correspondence(w, i) for w, i in zip(det_w, det_i) ]
    return HomographyInfo(corrs, H, im.shape)


#-------------------------------------
class IntrinsicsNode(object):
#-------------------------------------
    def __init__(self, tag, fx, fy, cx, cy, k1=0, k2=0, k3=0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.tag = tag # convenient identification

    def to_tuple(self):
        return ( self.fx, self.fy, self.cx, self.cy,
                 self.k1, self.k2, self.k3 )

    def set_value(self, *tupl):
        self.fx = tupl[0]
        self.fy = tupl[1]
        self.cx = tupl[2]
        self.cy = tupl[3]
        self.k1 = tupl[4]
        self.k2 = tupl[5]
        self.k2 = tupl[6]

    def __repr__(self):
        return repr(self.to_tuple())

    def to_matrix(self):
        from camera_math import intrinsics_to_matrix
        return intrinsics_to_matrix(self.fx, self.fy, self.cx, self.cy)

    def transform_to_camera_frame(self, points):
        cx, cy = self.cx, self.cy
        c = np.array([cx, cy]).T[:, None] # shape is (2, 1)
        return points - c # points is (2, N)

    def distort_radius(self, sq_radius):
        return self.k1*sq_radius + self.k2*(sq_radius**2) + self.k3*(sq_radius**3)


#-------------------------------------
class ExtrinsicsNode(object):
#-------------------------------------
    def __init__(self, tag, x, y, z, r, p, h):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.p = p
        self.h = h
        self.tag = tag # convenient identification

    def to_tuple(self):
        return self.x, self.y, self.z, self.r, self.p, self.h

    def set_value(self, *tupl):
        self.x = tupl[0]
        self.y = tupl[1]
        self.z = tupl[2]
        self.r = tupl[3]
        self.p = tupl[4]
        self.h = tupl[5]

    def __repr__(self):
        return repr(self.to_tuple())

    def to_matrix(self):
        from camera_math import xyzrph_to_matrix
        return xyzrph_to_matrix(*self.to_tuple())


#--------------------------------------
class HomographyConstraint(object):
#--------------------------------------
    def __init__(self, corrs, imshape, inode, enode):
        self.imshape = imshape

        # 3-D homogeneous form with z=0
        p_src = [ c.source for c in corrs ]
        N = len(p_src)
        p_src = np.hstack([ p_src, np.zeros((N,1)), np.ones((N,1)) ])

        self.p_src = p_src.T
        self.p_tgt = np.array([ c.target for c in corrs ]).T
        self.inode = inode
        self.enode = enode

    def reproject_world_points(self):
        """
        reproject world points `self.p_w` through the homography
        described by the composition of intrinsics and extrinsics.
        """
        K = self.inode.to_matrix()
        E = self.enode.to_matrix()
        H = K.dot(E)

        p_mapped = H.dot(self.p_src)[:3,:]

        # normalize homogeneous coordinates
        M = np.diag(1./p_mapped[2,:])
        p_mapped = p_mapped[:2,:].dot(M)

        return p_mapped


    def reproject_distort_world_points(self):
        """
        reproject world points `self.p_w` through the homography
        described by the composition of intrinsics and extrinsics.
        """
        p_mapped = self.reproject_world_points()

        # distort point according to radial model
        M = np.max(self.imshape)
        p_bar = self.inode.transform_to_camera_frame(p_mapped) / M
        rr = (p_bar**2).sum(axis=0) # sq. radius
        p_distorted = p_mapped + M * p_bar*self.inode.distort_radius(rr)

        return p_distorted


    def sq_errors(self):
        """
        compute the geometric reprojection error of the world
        points `self.p_w` through the homography described
        the composition of intrinsics and extrinsics.
        """
        p_distorted = self.reproject_distort_world_points()
        return ((p_distorted - self.p_tgt)**2).ravel()


#--------------------------------------
class ConstraintGraph(object):
#--------------------------------------
    def __init__(self):
        self.inodes = OrderedDict()
        self.enodes = OrderedDict()
        self.constraints = list()


    def constraint_errors(self):
        return np.hstack([ c.sq_errors() for c in self.constraints ])


    def sq_pixel_errors(self):
        homography_constraints = ( c for c in self.constraints if isinstance(c, HomographyConstraint) )
        return np.hstack([ c.sq_errors() for c in homography_constraints ])


    def _pack_into_vector(self):
        """ pack node states into a vector """
        istate = np.hstack([ i.to_tuple() for i in self.inodes.values() ])
        estate = np.hstack([ e.to_tuple() for e in self.enodes.values() ])
        return np.hstack(( istate, estate ))


    def _unpack_from_vector(self, v):
        """ Set node values from the vector `v` """
        N = len(self.inodes)
        istate = np.reshape(v[:7*N], (-1, 7))
        estate = np.reshape(v[7*N:], (-1, 6))

        for inode, ival in zip(self.inodes.values(), istate):
            inode.set_value(*ival)

        for enode, eval_ in zip(self.enodes.values(), estate):
            enode.set_value(*eval_)


    state = property(_pack_into_vector, _unpack_from_vector)


    def _pack_intrinsics_into_vector(self):
        """ pack intrinsic node states into a vector """
        return np.hstack([ i.to_tuple() for i in self.inodes.values() ])


    def _unpack_intrinsics_from_vector(self, v):
        """ Set intrinsic node values from the vector `v` """
        istate = np.reshape(v, (-1, 4))

        for inode, ival in zip(self.inodes.values(), istate):
            inode.set_value(*ival)


    istate = property(_pack_intrinsics_into_vector, _unpack_intrinsics_from_vector)


#---------------------------------------
class ClassicLensWarp(object):
#---------------------------------------
    def __init__(self, inode, imshape):
        self._inode = inode
        self._cxy = [inode.cx, inode.cy]
        self._imshape = imshape


    def _distort(self, q):
        M = np.max(self._imshape)
        p_bar = np.subtract(q, self._cxy) / M
        rr = (p_bar**2).sum(axis=0) # sq. radius
        return q + M * p_bar*self._inode.distort_radius(rr)


    def undistort(self, p):

        def objective(x):
            d = self._distort(x)
            return (d-p)**2 # sq.euclidean err

        from scipy.optimize import root
        return root(objective, x0=p).x


def main():
    np.set_printoptions(precision=4, suppress=True)

    import sys
    homography_info = [ get_homography_estimate(f) for f in sys.argv[1:] ]

    #
    # Estimate intrinsics
    #
    from camera_math import estimate_intrinsics_noskew
    from camera_math import get_extrinsics_from_homography
    from camera_math import matrix_to_xyzrph, matrix_to_intrinsics
    K = estimate_intrinsics_noskew([ hinf.H for hinf in homography_info])

    #
    # Setup factor graph
    #
    graph = ConstraintGraph()
    inode = IntrinsicsNode('in', *matrix_to_intrinsics(K))
    graph.inodes['in'] = inode

    for i, hinf in enumerate(homography_info):
        E = get_extrinsics_from_homography(hinf.H, K)
        enode = ExtrinsicsNode('pose%d'%i, *matrix_to_xyzrph(E))
        c = HomographyConstraint(hinf.corrs, hinf.imshape, inode, enode)
        graph.enodes['pose%d'%i] = enode
        graph.constraints.append(c)


    print 'Graph'
    print '-----'
    print '  %d intrinsic nodes' % len(graph.inodes)
    print '  %d extrinsic nodes' % len(graph.enodes)
    print '  %d constraints' % len(graph.constraints)
    print ''
    for constraint in (c for c in graph.constraints if isinstance(c, HomographyConstraint)):
        rmse = np.sqrt(constraint.sq_errors().mean())
        print '  %s rmse: %.2f' % (constraint.enode.tag, rmse)

    #
    # Optimize graph to reduce error in constraints
    #
    def print_graph_summary(title):
        print '\n' + title
        print '-'*len(title)
        print '    rmse: %.4f' % np.sqrt(graph.sq_pixel_errors().mean())
        print '  rmaxse: %.4f' % np.sqrt(graph.sq_pixel_errors().max())
        print ''
        for itag, inode in graph.inodes.iteritems():
            print '  intrinsics@ ' + itag + " =", np.array(inode.to_tuple())

    def objective(x):
        graph.state = x
        return graph.constraint_errors()

    def optimize_graph():
        x0 = graph.state
        print_graph_summary('Initial:')

        print '\nOptimizing graph ...'
        from scipy.optimize import root
        result = root(objective, x0, method='lm', options={'factor': 100, 'col_deriv': 1})
        print '  Success: ' + str(result.success)
        print '  %s' % result.message

        graph.state = result.x
        print_graph_summary('Final:')

    print '\n'
    print '====================='
    print '  Optimization'
    print '====================='

    optimize_graph()

    import os.path
    folder = os.path.dirname(sys.argv[1])

    import cPickle as pickle
    warp = ClassicLensWarp(inode, homography_info[0].imshape)
    with open(folder + '/classic.poly', 'w') as f:
        pickle.dump(warp, f)


if __name__ == '__main__':
    main()