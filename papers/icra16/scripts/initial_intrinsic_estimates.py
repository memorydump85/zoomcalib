#! /usr/bin/python

import os.path
import numpy as np
import cPickle as pickle
from collections import OrderedDict
from scipy.optimize import root

from camera_math import estimate_intrinsics_noskew_assume_cxy
from camera_math import robust_estimate_intrinsics_noskew
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


def main():
    import sys
    from glob import iglob
    from itertools import groupby

    np.set_printoptions(precision=4, suppress=True)

    folder = sys.argv[1]
    saved_files = iglob(folder + '/*.lh0')
    hmodels = [ HomographyModel.load_from_file(f) for f in saved_files ]
    print '%d hmodels\n' % len(hmodels)

    itag_getter = lambda e: e.itag
    hmodels.sort(key=itag_getter)

    def random_combinations(n, m):
        n_choose_m = [ x for x in xrange(2**n) if bin(x).count('1') == m ]
        return np.random.choice(n_choose_m, 200, replace=False)

    def take_by_bitindex(list_, bits_as_int):
        indices = xrange(len(list_)-1, -1, -1)
        bool_bits = ( bool(bits_as_int & 2**i) for i in indices )
        return [ entry for entry, bit in zip(list_, bool_bits) if bit ]

    for itag, group in groupby(hmodels, key=itag_getter):
        group = list(group)
        homographies = [ hm.homography_at_center() for hm in group ]

        # K = robust_estimate_intrinsics_noskew(homographies)
        # print np.array(matrix_to_intrinsics(K))

        K = estimate_intrinsics_noskew(homographies)
        print np.array(matrix_to_intrinsics(K))

        bootstrap_estimates = []
        for combination in random_combinations(len(homographies), 3):
            subset = take_by_bitindex(homographies, combination)
            try:
                K = estimate_intrinsics_noskew(subset)
                bootstrap_estimates.append(matrix_to_intrinsics(K))
            except:
                pass

        bootstrap_estimates = np.vstack(bootstrap_estimates)
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(bootstrap_estimates))
        medoid_ix = np.argmin(distances.mean(axis=0))
        print bootstrap_estimates[medoid_ix]
        print np.median(bootstrap_estimates, axis=0)

if __name__ == '__main__':
    main()