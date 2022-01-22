# File: src/chandra/observing.py

import io
import urllib
from astropy.io import fits
import aplpy
import matplotlib.pyplot as plt
import numpy as np
import sys


#### FINDER CHARTS #############
#### ADAPTED FROM PYSALT: https://github.com/saltastro/pysalt #######

def get_dss(imserver, ra, dec):
    url = "http://archive.stsci.edu/cgi-bin/dss_search?v=%s&r=%f&d=%f&e=J2000&h=5.0&w=5.0&f=fits&c=none" %\
                    (imserver, ra, dec)
    print(url)
    fitsData = io.BytesIO()
    data = urllib.request.urlopen(url).read()
    fitsData.write(data)
    fitsData.seek(0)
    return fits.open(fitsData)

def init_plot(hdu, title, ra, dec):
    servname = {}
    servname['none']=''
    servname['poss2ukstu_red'] = "POSS2/UKSTU Red"
    servname['poss2ukstu_blue'] = "POSS2/UKSTU Blue"
    servname['poss2ukstu_ir'] = "POSS2/UKSTU IR"
    servname['poss1_blue'] = "POSS1 Blue"
    servname['poss1_red'] = "POSS1 Red"
    
    imserver = 'poss2ukstu_red'

    out = sys.stdout
    sys.stdout = open("/dev/null", 'w')
    plot = aplpy.FITSFigure(hdu)
    plot.show_grayscale()
    plot.set_theme('publication')
    sys.stdout = out
    plot.add_label(0.5, 1.03,
                  title,
                  relative=True, style='italic', weight='bold', size='large')
    plot.add_label(-0.05, -0.05, "%s" % 'DSS Red', relative=True, style='italic', weight='bold')

    plot.add_grid()
    plot.grid.set_alpha(0.2)
    plot.grid.set_color('b')

    cart = 2.2

    plot.add_label(ra,
                  dec+cart/60.0,
                  "N",
                  style='italic',
                  weight='bold',
                  size='large',
                  color='k')
    plot.add_label(ra+cart/(np.abs(np.cos(dec*np.pi/180.0))*60),
                  dec,
                  "E",
                  style='italic',
                  weight='bold',
                  size='large',
                  horizontalalignment='right',
                  color='k')
    plot = draw_line(plot, 0, 8, ra, dec, color='tab:red', linewidth=0.5, alpha=1.0)
    plot = draw_line(plot, 90, 8, ra, dec, color='tab:red', linewidth=0.5, alpha=1.0)
    return plot

# draw a line centered at ra,dec of a given length at a given angle
def draw_line(plot, theta, length, ra, dec, color='b', linewidth=1, alpha=0.7):
    theta = theta*np.pi/180.0
    length = length/2.0
    dx = np.sin(theta)*length/(np.cos(dec*np.pi/180.0)*60.0)
    dy = np.cos(theta)*length/60.0
    coords = np.array([[ra+dx, ra-dx], [dec+dy, dec-dy]])
    plot.show_lines([coords], color=color, linewidth=linewidth, alpha=alpha)
    return plot

def finderchart(ra, dec, name):
    img = get_dss('poss2ukstu_red', ra, dec)
    plot = init_plot(img, name, ra, dec)
    plt.tight_layout()
    return plot