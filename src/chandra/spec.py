# File: src/chandra/spec.py

from bisect import bisect_left
from numpy.polynomial.polynomial import polyfit,polyval
import numpy as np

def cont_norm(wl, fl, ivar, centre, width, edge):
    c1 = bisect_left(wl, centre - width)
    c2 = bisect_left(wl, centre + width)

    wl,fl,ivar = wl[c1:c2], fl[c1:c2],ivar[c1:c2]
    
    cmask = np.repeat(True, len(wl))
    cmask[edge:-edge] = False
    
    contp = polyfit(wl[cmask], fl[cmask], 1)
    cont = polyval(wl, contp)
    
    fl = fl/cont
    ivar = ivar * cont**2
    
    return wl, fl, ivar