import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from apriltag import AprilTagDetector
from projective_math import WeightedLocalHomography, SqExpWeightingFunction
from tag36h11_mosaic import TagMosaic



def main():

    im = imread('/var/tmp/datasets/tamron-2.2/im000.png')
    im = rgb2gray(im)
    im = img_as_ubyte(im)

    tag_mosaic = TagMosaic(0.0254)
    detections = AprilTagDetector().detect(im)

    np.random.shuffle(detections)
    world = np.array([ tag_mosaic.get_position_meters(d.id) for d in detections ])
    image = np.array([ d.c for d in detections ])

    wlhe = WeightedLocalHomography(wfunc=SqExpWeightingFunction(.01, 45))
    wlhe.regularization_lambda = 1e-3

    # Use the first N-10 detections to fit model
    for w, i in zip(world[:-20], image[:-20]):
        wlhe.add_correspondence(w, i)

    # Validate on the last 10 detections
    sqerr = []
    for w, i in zip(world[-20:], image[-20:]):
        p = wlhe.map(w)
        p = p / p[2]
        sqerr.append( np.linalg.norm(p[:2] - i)**2 )

    print '    rmse = %.4f' % np.sqrt(np.mean(sqerr))
    print 'r_max_se = %.4f' % np.sqrt(np.max(sqerr))

    if True:
        from skimage.filters import scharr
        from matplotlib import pyplot as plt

        plt.imshow(1.-scharr(im), cmap='bone')

        INCH = 0.0254
        def get_mapped_points(y):
            mapped_points = []
            for x in np.arange(0, 9, 0.1):
                p = wlhe.map([x*INCH, -y])
                mapped_points.append(p)
            return np.array(mapped_points)

        for y in xrange(0, 8):
            pts = get_mapped_points(y*INCH)
            plt.plot(pts[:,0], pts[:,1], 'm-', linewidth=1.5)

        detection_xy = np.array([ d.c for d in detections ])
        plt.plot(detection_xy[:,0], detection_xy[:,1], 'b.')
        detection_xy = np.array([ d.c for d in detections ])
        plt.plot(detection_xy[-10:,0], detection_xy[-10:,1], 'k*')
        plt.show()

if __name__ == '__main__':
    main()