"""
Ad-hoc script to convert a tag-mosaic.png to svg
"""

from skimage.io import imread

im = imread('/tmp/tags/mosaic.png')[:,:,1]
H, W = im.shape

def rect(x, y):
	print '    <rect x="%.1fin" y="%.1fin" width="0.1in" height="0.1in"/>' % (x, y)

print '<svg width="25in" height="25in">'
for y in xrange(H):
	for x in xrange(W):
		if im[y,x] == 0:
			rect(x*0.1, y*0.1)
print '</svg>'