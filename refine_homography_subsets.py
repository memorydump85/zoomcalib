#! /usr/bin/python

import os.path
import numpy as np
import cPickle as pickle
from itertools import groupby
from collections import OrderedDict
from scipy.optimize import root

from camera_math import estimate_intrinsics_noskew_assume_cxy
from camera_math import estimate_intrinsics_noskew
from camera_math import get_extrinsics_from_homography
from camera_math import matrix_to_xyzrph, matrix_to_intrinsics
from camera_math import xyzrph_to_matrix, intrinsics_to_matrix
from tupletypes import WorldImageHomographyInfo



#--------------------------------------
class HomographyModel(object):
#--------------------------------------
    """
    Encapsulation of the data in `WorldImageHomographyInfo`
    and methods that act on that data

    Members:
    --------
        `hinfo`: `WorldImageHomographyInfo` object
         `itag`: intrinsics tag
         `etag`: extrinsics tag
    """
    def __init__(self, hinfo):
        self.hinfo = hinfo
        self.etag = None
        self.itag = None
        self.H0 = None


    @classmethod
    def load_from_file(class_, filename):
        # parse the filename to get intrinsic/extrinsic tags
        itag, etag = filename.split('/')[-2:]
        etag = etag.split('.')[0]

        with open(filename) as f:
            hinfo = pickle.load(f)

        # create and populate instance
        instance = class_(hinfo)
        instance.etag = etag
        instance.itag = itag
        return instance


    def homography_at_center(self):
        if self.H0 is None:
            H_wi, c_w, _ = self.hinfo
            self.H0 = H_wi.get_homography_at(c_w)

        return self.H0


#-------------------------------------
class IntrinsicsNode(object):
#-------------------------------------
    def __init__(self, fx, fy, cx, cy, tag):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.tag = tag # convenient identification

    def to_tuple(self):
        return self.fx, self.fy, self.cx, self.cy

    def set_value(self, *tupl):
        self.fx = tupl[0]
        self.fy = tupl[1]
        self.cx = tupl[2]
        self.cy = tupl[3]

    def __repr__(self):
        return repr(self.to_tuple())

    def to_matrix(self):
        return intrinsics_to_matrix(*self.to_tuple())


#-------------------------------------
class ExtrinsicsNode(object):
#-------------------------------------
    def __init__(self, x, y, z, r, p, h, tag):
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
        return xyzrph_to_matrix(*self.to_tuple())


#--------------------------------------
class HomographyConstraint(object):
#--------------------------------------
    def __init__(self, hmodel, inode, enode):
        H_wi, c_w, _ = hmodel.hinfo
        weights = H_wi.get_correspondence_weights(c_w)

        # 3-D homogeneous form with z=0
        p_src = [ c.source for c in H_wi._corrs ]
        N = len(p_src)
        p_src = np.hstack([ p_src, np.zeros((N,1)), np.ones((N,1)) ])

        self.p_src = p_src.T
        self.p_tgt = np.array([ c.target for c in H_wi._corrs ]).T
        self.W = np.diag(weights)
        self.inode = inode
        self.enode = enode

    def sq_unweighted_reprojection_errors(self):
        """
        compute the geometric reprojection error of the world
        points `self.p_w` through the homography described
        the composition of intrinsics and extrinsics.
        """
        K = self.inode.to_matrix()
        E = self.enode.to_matrix()
        H = K.dot(E)

        p_mapped = H.dot(self.p_src)[:3,:]

        # normalize homogeneous coordinates
        M = np.diag(1./p_mapped[2,:])
        p_mapped = p_mapped[:2,:].dot(M)

        return ((p_mapped - self.p_tgt)**2)


    def sq_errors(self):
        sqerr = self.sq_unweighted_reprojection_errors()
        return sqerr.dot(self.W).ravel()


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
        return np.hstack([ c.sq_unweighted_reprojection_errors() for c in homography_constraints ])


    def _pack_into_vector(self):
        """ pack node states into a vector """
        istate = np.hstack([ i.to_tuple() for i in self.inodes.values() ])
        estate = np.hstack([ e.to_tuple() for e in self.enodes.values() ])
        return np.hstack(( istate, estate ))


    def _unpack_from_vector(self, v):
        """ Set node values from the vector `v` """
        N = len(self.inodes)
        istate = np.reshape(v[:4*N], (-1, 4))
        estate = np.reshape(v[4*N:], (-1, 6))

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


def refine_homography_subset(args):
    """ Load homographies pickled in `homography_files` and
    choose a subset of these homographies according to `bit_index`.
    Refine this subset of homographies """

    homography_files, bit_index = args

    def take_by_bitindex(list_, bits_as_int):
        indices = xrange(len(list_)-1, -1, -1)
        bool_bits = ( bool(bits_as_int & 2**i) for i in indices )
        return [ entry for entry, bit in zip(list_, bool_bits) if bit ]

    homography_files = take_by_bitindex(homography_files, bit_index)
    hmodels = [ HomographyModel.load_from_file(f) for f in homography_files ]

    #
    # Deconstruct information in the HomographyModels into
    # a graph of nodes and constraints
    #
    graph = ConstraintGraph()

    itag_getter = lambda e: e.itag
    hmodels.sort(key=itag_getter)

    for itag, group in groupby(hmodels, key=itag_getter):
        group = list(group)
        homographies = [ hm.homography_at_center() for hm in group ]

        K = estimate_intrinsics_noskew(homographies)
        inode = IntrinsicsNode(*matrix_to_intrinsics(K), tag=itag)

        # For each HomographyModel in `group` construct an ExtrinsicsNode
        enode_tags = [ '%s/%s' % (itag, hm.etag) for hm in group ]
        E_matrices = ( get_extrinsics_from_homography(H, K) for H in homographies )
        enodes = [ ExtrinsicsNode(*matrix_to_xyzrph(E), tag=tag) for E, tag in zip(E_matrices, enode_tags) ]

        # Instantiate constraints between each ExtrinsicsNode and the single IntrinsicsNode
        constraints = [ HomographyConstraint(hm, inode, enode) for hm, enode in zip(group, enodes) ]

        # Add nodes and constraints to graph
        graph.inodes[itag] = inode
        graph.enodes.update( (e.tag, e) for e in enodes )
        graph.constraints.extend( constraints )

    #
    # Optimize graph to reduce error in constraints
    #
    def objective(x):
        graph.state = x
        return graph.constraint_errors()

    x0 = graph.state
    result = root(objective, x0, method='lm', options={'factor': 0.1, 'col_deriv': 1})
    graph.state = result.x

    return graph.istate


def main():
    import sys
    import multiprocessing
    from glob import glob

    np.set_printoptions(precision=4, suppress=True)

    def random_combinations(n, m):
        n_choose_m = [ x for x in xrange(2**n) if bin(x).count('1') == m ]
        return np.random.choice(n_choose_m, 250, replace=False)

    for subfolder in sorted(glob(sys.argv[1] + '/*/')):
        print '  %s' % subfolder
        homography_files = glob(subfolder + '*.lh0')
        num_files = len(homography_files)

        pool = multiprocessing.Pool()
        args = [ (homography_files, bit_index) for bit_index in random_combinations(num_files, 5) ]
        samples = pool.map(refine_homography_subset, args)

        with open(subfolder + '/intrinsics.samples', 'w') as f:
            pickle.dump(samples, f)

if __name__ == '__main__':
    main()
