import sys
import os, os.path, shutil
from subprocess import Popen, check_call, check_output, STDOUT
from cStringIO import StringIO

def get_camera_image(output_folder):
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

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print '  Converting image to .png'
    filename = '%s/%03d' % (output_folder, focal_length)
    shutil.copyfile('/tmp/cap.jpeg', filename + '.jpeg')
    Popen(['convert', filename + '.jpeg', filename + '.png'])


def main():
    if len(sys.argv) != 2:
        print '  USAGE: capture_static.py <output-folder>'
        sys.exit(-1)

    output_folder = sys.argv[1]
    while True:
        get_camera_image(output_folder)
        raw_input("Press Enter to continue...")

if __name__ == '__main__':
    main()