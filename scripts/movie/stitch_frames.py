"""
Merges images with frame descriptors (as list or linspace) into composite frames,
and stitches composite frames into a movie with ffmpeg.
"""
import argparse
import os

from PIL import Image, ImageFont, ImageDraw
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("nframes", type=int, help="number of frames to write into movie")
parser.add_argument("-shape", type=int, nargs=2, help="shape (x, y) of composite image")
parser.add_argument("-imgformats", nargs='+',
                    help="image format, or multiple image formats to merge into composite frame, with bracket-type format placeholder for image index")
parser.add_argument("-imglabels", nargs='+',
                    help="label for each image on the composite frame")
parser.add_argument("-compositeformat",
                    help="in case of more than one image per frame, image format for composite frames")
parser.add_argument("-framelabels_list", nargs='+', help="construct labels for each frame from list")
parser.add_argument("-framelabels_linspace", nargs=3, help="construct labels for each frame from linspace")

ffmpegargs = parser.add_argument_group('Arguments for movie generation with ffmpeg')
ffmpegargs.add_argument("-rate", type=int, help="frame rate for movie")
ffmpegargs.add_argument("-o", help="output movie")

parser.parse_args()

################################################################################
# Merge images
################################################################################

################################################################################
# Make movie
################################################################################
