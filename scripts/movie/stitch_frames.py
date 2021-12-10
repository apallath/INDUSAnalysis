"""
Merges images with frame descriptors (as list or linspace) into composite frames,
and stitches composite frames into a movie with ffmpeg.
"""
import argparse
import os

from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageFont, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("nframes", type=int, help="number of frames to write into movie")
parser.add_argument("-start", type=int, help="frame index to start at")
parser.add_argument("-shape", type=int, nargs=2, help="shape (x, y) of composite image")
parser.add_argument("-imgformats", nargs='+',
                    help="image format, or multiple image formats to merge into composite frame, with bracket-type format placeholder for image index")
parser.add_argument("-imglabels", nargs='+',
                    help="label for each image on the composite frame")
parser.add_argument("-compositeformat",
                    help="in case of more than one image per frame, image format for composite frames")
parser.add_argument("-framelabels_list", nargs='+', help="construct labels for each frame from list")
parser.add_argument("-framelabels_linspace", nargs=3, help="construct labels for each frame from linspace")
parser.add_argument("-framelabels_format", help="format string to format label")
parser.add_argument("-framelabels_size", type=int, default=48, help="font size of label [default=48]")

ffmpegargs = parser.add_argument_group('Arguments for movie generation with ffmpeg')
ffmpegargs.add_argument("-rate", type=int, help="frame rate for movie")
ffmpegargs.add_argument("-o", help="output movie")

a = parser.parse_args()

################################################################################
# Merge images
################################################################################\

# Checks
assert(a.shape[0] * a.shape[1] == len(a.imgformats))
assert(len(a.imgformats) == len(a.imglabels))

# Read frame labels
if a.framelabels_list is not None:
    framelabels = a.framelabels_list
elif a.framelabels_linspace is not None:
    framelabels = np.linspace(float(a.framelabels_linspace[0]), float(a.framelabels_linspace[1]), int(a.framelabels_linspace[0]))
else:
    framelabels = a.nframes * [""]

assert(a.nframes == len(framelabels))

# Load images and write composite images
for i, label in enumerate(framelabels):
    imgfiles = [imgformat.format(i + a.start) for imgformat in a.imgformats]

    images = [Image.open(x) for x in imgfiles]

    for imgidx in range(len(images)):
        draw = ImageDraw.Draw(images[imgidx])
        # find font
        font_prop = font_manager.FontProperties(family='sans-serif', weight='bold')
        font_file = font_manager.findfont(font_prop)
        font = ImageFont.truetype(font_file, 48)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((100, 5), a.imglabels[imgidx] + a.framelabels_format.format(label), (0, 0, 0), font=font)

    widths, heights = zip(*(im.size for im in images))

    new_width = a.shape[0] * max(widths)
    new_height = a.shape[1] * max(heights)

    x_offset = new_width // 2
    y_offset = new_height // 2

    new_image = Image.new('RGB', (new_width, new_height))

    # Paste and merge images
    """
    new_image.paste(images[0], (0, 0))
    new_image.paste(images[1], (0, y_offset))

    new_image.paste(images[2], (x_offset, 0))
    new_image.paste(images[3], (x_offset, y_offset))

    new_image.save(out_image_format.format(i + 1))
    """

################################################################################
# Make movie
################################################################################

# os.system("ffmpeg -r 5 -i {}_compile_%05d.jpg -vcodec mpeg4 -y -q:v 1 -vb 40M 00_{}_tpfp.mp4".format(TYPE, TYPE))
