# File: src/chandra/plot.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy import units as u, constants as c

from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft

def plot_bin2d(x, y, z, statistic = np.median, bins = 50, range = None, 
              vmin = None, vmax = None, cmap = 'Spectral_r', norm = None, bs_kw = {}, im_kw = {}):
    
    bs = stats.binned_statistic_2d(x, y, z, bins = bins, statistic = statistic, range = range, **bs_kw)
    im = plt.pcolormesh(bs.x_edge, bs.y_edge, bs.statistic.T, vmin = vmin, vmax = vmax, cmap = cmap,
                       norm = norm, **im_kw)
    
    return im

# def hist_norm(x, y, pkw = dict(), normx = False, normy = False, xlab = 'x', ylab = 'y', **hkw):

#     H, xedge, yedge = np.histogram2d(x, y, **hkw)

#     plt.xlabel(xlab)
#     plt.ylabel(ylab)
    
#     if normx:
#         Hs = np.nanmax(H, axis = 0)
#         Hs += (Hs == 0)
#         H = H * 1 / (Hs[None, :])
#         plt.xlabel(xlab + ' $\mid$ ' + ylab)
    
#     elif normy:
#         Hs = np.nanmax(H, axis = 1)
#         Hs += (Hs == 0)
#         H = H * 1 / (Hs[:, None])
#         plt.ylabel(ylab + ' $\mid$ ' + xlab)
    
#     ax = plt.pcolormesh(xedge, yedge, H.T, **pkw)

#     return ax

def hist_norm(x, y, pkw = dict(), normx = False, normy = False, 
              rangex = None, rangey = None, bins = [100, 100],
              margx = False, margy = False, margscale = 0.5, marglog = False, 
              margkw = dict(color = 'w', lw = 2.5), margbins = 100,
              xlab = 'x', ylab = 'y',
              **hkw,):
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
        
    if rangex is None:
        rangex = [np.nanmin(x), np.nanmax(x)]
    
    if rangey is None:
        rangey = [np.nanmin(y), np.nanmax(y)]
        
    if rangex is not None and rangey is not None:
        range = (rangex, rangey)
    
    H, xedge, yedge = np.histogram2d(x, y, range = range, bins = bins, **hkw)
    
    if normx:
        Hs = np.nanmax(H, axis = 0)
        Hs += (Hs == 0)
        H = H * 1 / (Hs[None, :])
        plt.xlabel(xlab + ' $\mid$ ' + ylab)
    
    elif normy:
        Hs = np.nanmax(H, axis = 1)
        Hs += (Hs == 0)
        H = H * 1 / (Hs[:, None])
        plt.ylabel(ylab + ' $\mid$ ' + xlab)
            
    if margx:
        xf, xg = np.histogram(x, bins = margbins, range = rangex)
        xg = xg[:-1] + np.diff(xg)[0]/2
        if marglog:
            xf = np.log10(xf)
        cl = np.isfinite(xf)
        xf = (xf - xf[cl].min()) / (xf[cl].max() - xf[cl].min())
        xf *= margscale
        xf += np.sign(rangey[0]) * np.abs(rangey[0])
        plt.plot(xg, xf, **margkw)
                
    if margy:
        yf, yg = np.histogram(y, bins = margbins, range = rangey)
        yg = yg[:-1] + np.diff(yg)[0]/2
        if marglog:
            yf = np.log10(yf)
        
        cl = np.isfinite(yf)
        yf = (yf - yf[cl].min()) / (yf[cl].max() - yf[cl].min())
        
        yf *= margscale
        yf += np.sign(rangex[0]) * np.abs(rangex[0])
        plt.plot(yf, yg, **margkw)
    
    
    im = plt.pcolormesh(xedge, yedge, H.T, **pkw)

    return im