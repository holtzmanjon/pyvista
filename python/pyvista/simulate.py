# routines for simulating images

import numpy as np
import pdb

def gauss2d(data, coords,fwhm=1,noise=False) :
    """ Add 2d gaussians to data given input coords, fwhm, noise"""
    pix = np.mgrid[0:data.shape[0],0:data.shape[1]]
    ypix = pix[0]
    xpix = pix[1]
    sig2 = (fwhm/2.354)**2
    for coord in coords: 
        amp = coord[0]/2./np.pi/sig2
        x = coord[1]
        y = coord[2]
        dist2 = (xpix-x)**2 + (ypix-y)**2
        gd = np.where(dist2 < 100*sig2)[0]
        data[gd] += amp*np.exp(-dist2[gd]/(2.*sig2))
        pdb.set_trace()
