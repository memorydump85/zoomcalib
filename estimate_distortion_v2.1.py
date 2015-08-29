#! /usr/bin/python

import os.path
import numpy as np
import cPickle as pickle
from collections import OrderedDict
from scipy.optimize import root

from projective_math import SqExpWeightingFunction
from refine_homography import estimate_intrinsics
from refine_homography import estimate_intrinsics_noskew
from refine_homography import get_extrinsics_from_homography
from refine_homography import  _matrix_to_xyzrph, _matrix_to_intrinsics
from refine_homography import _xyzrph_to_matrix, _intrisics_to_matrix
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
        etag, itag = filename.split('/')[-2:]
        itag = itag.split('.')[0]

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
        return _intrisics_to_matrix(*self.to_tuple())


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
        return _xyzrph_to_matrix(*self.to_tuple())


#--------------------------------------
class HomographyConstraint(object):
#--------------------------------------
    def __init__(self, hmodel, inode, enode):
        H_wi, c_w, _ = hmodel.hinfo
        wfunc = H_wi.regularized_weighting_function

        w0 = wfunc(c_w, c_w)
        weights = np.array(H_wi.get_correspondence_weights(c_w)) / w0

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


    def sq_reprojection_errors(self):
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
        return np.hstack([ c.sq_reprojection_errors() for c in self.constraints ])


    def sq_pixel_errors(self):
        return np.hstack([ c.sq_unweighted_reprojection_errors() for c in self.constraints ])


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


def main():
    import sys
    from glob import iglob
    from itertools import groupby

    np.set_printoptions(precision=4, suppress=True)

    folder = sys.argv[1]
    saved_files = iglob(folder + '/pose?/*.lh0')
    hmodels = [ HomographyModel.load_from_file(f) for f in saved_files ]

    #
    # Deconstruct information in the HomographyModels into
    # a graph of nodes and constraints
    #
    graph = ConstraintGraph()

    #
    # Construct intrinsic nodes
    #
    itag_getter = lambda e: e.itag
    hmodels.sort(key=itag_getter)

    for itag, group in groupby(hmodels, key=itag_getter):
        homographies = [ hm.homography_at_center() for hm in group ]
        K = estimate_intrinsics_noskew(homographies)
        graph.inodes[itag] = IntrinsicsNode(*_matrix_to_intrinsics(K), tag=itag)

    #
    # Construct extrinsic nodes
    #
    etag_getter = lambda e: e.etag
    hmodels.sort(key=etag_getter)

    for etag, group in groupby(hmodels, key=etag_getter):
        estimates = []
        for hm in group:
            K = graph.inodes.get(hm.itag).to_matrix()[:,:3]
            E = get_extrinsics_from_homography(hm.homography_at_center(), K)
            estimates.append(_matrix_to_xyzrph(E))
        graph.enodes[etag] = ExtrinsicsNode(*np.mean(estimates, axis=0), tag=etag)

    print '%d intrinsic nodes' % len(graph.inodes)
    print '%d extrinsic nodes' % len(graph.enodes)

    #
    # Connect nodes by constraints
    #
    for hm in hmodels:
        inode = graph.inodes[hm.itag]
        enode = graph.enodes[hm.etag]
        constraint = HomographyConstraint(hm, inode, enode)
        graph.constraints.append(constraint)

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

    print_graph_summary('Initial:')

    def objective(x):
        graph.state = x
        return graph.constraint_errors()

    print '\nOptimizing graph ...'
    x0 = graph.state
    result = root(objective, x0, method='lm')
    print '  Success: ' + str(result.success)
    print '  %s' % result.message

    print_graph_summary('Final:')

    #
    # Write out the refined homographies
    #
    for constraint in graph.constraints:
        etag, itag = constraint.enode.tag, constraint.inode.tag
        K = constraint.inode.to_matrix()
        E = constraint.enode.to_matrix()
        H = K.dot(E)
        filename = '%s/%s/%s.lh0+' % (folder, etag, itag)
        with open(filename, 'w') as f:
            pickle.dump(H, f)


if __name__ == '__main__':
    main()
