# File: src/chandra/plot.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy import units as u, constants as c

from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft

def plot_bin2d(x, y, z, statistic = np.median, bins = 50, range = None, 
              vmin = None, vmax = None, cmap = 'Spectral_r', norm = None):
    
    bs = stats.binned_statistic_2d(x, y, z, bins = bins, statistic = statistic, range = range)
    im = plt.pcolormesh(bs.x_edge, bs.y_edge, bs.statistic.T, vmin = vmin, vmax = vmax, cmap = cmap,
                       norm = norm)
    
    return im