"""
Ad-hoc script to convert a tag-mosaic.png to svg
"""

from skimage.io import imread

im = imread('mosaic.png')[:,:,1]
H, W = im.shape

tag_size = 0.8
mosaic_size = 25. * tag_size
unit = tag_size / 10.;

def rect(x, y):
	print '    <rect x="%sin" y="%sin" width="%sin" height="%sin"/>' % (str(x*unit), str(y*unit), str(unit), str(unit))

print '<svg width="%sin" height="%sin">' % (str(mosaic_size), str(mosaic_size))
for y in xrange(H):
	for x in xrange(W):
		if im[y,x] == 0:
			rect(x, y)
print '</svg>'