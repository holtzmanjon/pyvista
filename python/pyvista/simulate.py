# routines for simulating images

import numpy as np
import pdb
from astropy.nddata import CCDData

def gauss2d(data, coords,fwhm=1,noise=False) :
    """ Add 2d gaussians to data given input coords, fwhm, noise"""
    if type(coords[0]) is not list : coords=[coords]
    pix = np.mgrid[0:data.shape[0],0:data.shape[1]]
    ypix = pix[0]
    xpix = pix[1]
    sig2 = (fwhm/2.354)**2
    for coord in coords: 
        amp = coord[0]/2./np.pi/sig2
        x = coord[1]
        y = coord[2]
        dist2 = (xpix-x)**2 + (ypix-y)**2
        gd = np.where(dist2 < 100*sig2)
        data[gd[0],gd[1]] += amp*np.exp(-dist2[gd[0],gd[1]]/(2.*sig2))

    return CCDData(data,unit='photon')
