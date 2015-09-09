import sys
import glob
import os, os.path
from itertools import groupby
from subprocess import Popen, check_call, check_output
from cStringIO import StringIO



def get_focal_length_from_EXIF(filename):
    focal_length = None
    for line in StringIO(check_output(['identify', '-verbose', filename])):
        if line.strip().startswith("exif:FocalLength:"):
            focal_length = eval(line.strip().split()[1])

    if focal_length is None:
        raise Exception('EXIF data did not contain Focal Length information!')

    print '%s: zoom level %d' % (filename, focal_length)
    return focal_length


def main():
    folder = sys.argv[1]
    filenames = sorted(glob.glob(folder + '/*.JPG'))
    zoom_values = [ get_focal_length_from_EXIF(f) for f in filenames ]

    # create folders for zoom values
    for val in set(zoom_values):
        subfolder = folder + '/%03d' % val
        if not os.path.exists(subfolder):
            print '\ncreating', subfolder
            os.mkdir(subfolder)

    files_and_zooms = list(zip(filenames, zoom_values))
    zoom_getter = lambda e: e[1]
    files_and_zooms.sort(key=zoom_getter)

    print ''
    for zoom, file_group in groupby(files_and_zooms, key=zoom_getter):
        subfolder = '%03d' % zoom
        for i, (filename, _) in enumerate(file_group):
            command = [ "convert", filename, folder+'/'+subfolder+'/pose'+str(i)+'.png' ]
            print ' '.join(command)
            check_call(command)


if __name__ == '__main__':
    main()