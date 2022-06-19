import numpy as np 
import scipy
from scipy import stats

def binned_stat(x, y, bins, statistic = np.nanmedian, **kw):
    
    bs = stats.binned_statistic(x, y, bins = bins, statistic = statistic, **kw)
    mid = bs.bin_edges[:-1] + np.diff(bs.bin_edges)[0]/2
    
    return mid, bs.statistic

def wmean(x, e_x):
    mean = np.average(x, weights = 1 / e_x**2)
    sigma = np.sqrt(1 / np.sum(1 / e_x**2))
    return mean, sigma