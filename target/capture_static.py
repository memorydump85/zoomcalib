import sys
import os, os.path, shutil
from subprocess import check_call, check_output, STDOUT
from PIL.ExifTags import TAGS
from cStringIO import StringIO

def get_camera_image():
    sys.stdout.write('  Acquiring image ...\r')
    sys.stdout.flush()

    if os.path.exists('/tmp/cap.jpeg'):
        os.remove('/tmp/cap.jpeg')

    check_output(["gphoto2", "--filename=/tmp/cap.jpeg", "--capture-image-and-download"], stderr=STDOUT)

    if not os.path.exists('/tmp/cap.jpeg'):
        print 'Error capturing image!'
        return

    focal_length = None
    for line in StringIO(check_output(['identify', '-verbose', '/tmp/cap.jpeg'])):
        if line.strip().startswith("exif:FocalLength:"):
            focal_length = eval(line.strip().split()[1])

    if focal_length is None:
        print 'EXIF data did not contain Focal Length information!'
        return

    print '  Image acquired. Focal length =', focal_length

    if not os.path.exists('/var/tmp/capture'):
        os.mkdir('/var/tmp/capture')

    shutil.copyfile('/tmp/cap.jpeg', '/var/tmp/capture/%s.jpg' % focal_length)
    
while True:
	get_camera_image()
	raw_input("Press Enter to continue...")