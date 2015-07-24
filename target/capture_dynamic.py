import threading
import time
from gi.repository import Gtk, Gdk, WebKit, GObject



GObject.threads_init()

webview = WebKit.WebView()
webview.open("file:///home/rpradeep/studio/zoomcalib/target/calib_target.html")


#
# Move the svg element by executing javascript
# within the browser widget
#
svg_moved_event = threading.Event()

def webview_svg_set_pos(top, left):
    webview.execute_script("""
        var mosaic = document.getElementById("mosaic");
        mosaic.style.top = %s;
        mosaic.style.left = %s;
        """ % (top, left))

    global svg_moved_event
    svg_moved_event.set()


#
# Use gphoto2 to capture image
#
image_aquired_event = threading.Event()

import sys
import os, os.path, shutil
from subprocess import check_call, check_output, STDOUT
from PIL.ExifTags import TAGS
from cStringIO import StringIO

def get_camera_image(token):
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

    shutil.copyfile('/tmp/cap.jpeg', '/var/tmp/capture/%s_%s.jpeg' % (focal_length, token))
    
    global image_aquired_event
    image_aquired_event.set()


#
# Thread to translate the svg image on screen
#
class AnimateThread(threading.Thread):
    def __init__(self):
        super(AnimateThread, self).__init__()

    def run(self):
        global svg_moved_event
        global image_aquired_event

        time.sleep(2)
        positions = [ ('"0.00in"', '"0.00in"'),
                      ('"0.25in"', '"0.25in"'),
                      ('"0.25in"', '"0.75in"'),
                      ('"0.75in"', '"0.75in"'),
                      ('"0.75in"', '"0.25in"'),
                      ('"0.50in"', '"0.50in"') ]
        
        for i, (top, left) in enumerate(positions):
            svg_moved_event.clear()
            GObject.idle_add(webview_svg_set_pos, top, left)
            svg_moved_event.wait()

            time.sleep(0.2)
            
            image_aquired_event.clear()
            GObject.idle_add(get_camera_image, chr(65+i))
            image_aquired_event.wait()
      
        GObject.idle_add(Gtk.main_quit)


#
# Construct the main window and start it all up
#
print 'Detecting cameras ...'
print '----------------------------------------------------------'
check_call(['gphoto2', '--auto-detect'])
print ' '

scroll_area = Gtk.ScrolledWindow()
scroll_area.add(webview)

win = Gtk.Window()
win.fullscreen()
win.connect("destroy", Gtk.main_quit)
win.add(scroll_area)
win.show_all()

t = AnimateThread()
t.start()

Gtk.main()