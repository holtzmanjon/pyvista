# routines for simulating images

import numpy as np
import pdb
from astropy.nddata import CCDData

sig2fwhm = 2*np.sqrt(2*np.log(2))

def gauss2d(data, coords,fwhm=1,back=0,noise=False,rn=0) :
    """ Add 2d gaussians to data given input coords, fwhm, noise
    """

    # if we are only given one object, make it a list
    try : iter(coords[0])
    except TypeError : coords=[coords]

    # set arrays of pixel index values
    ypix,xpix = np.mgrid[0:data.shape[0],0:data.shape[1]]
    sig2 = (fwhm/sig2fwhm)**2

    # loop over input objects
    for coord in coords: 
        amp = coord[0]/2./np.pi/sig2
        x = coord[1]
        y = coord[2]
        dist2 = (xpix-x)**2 + (ypix-y)**2
        # select points within 10*sigma ~ 4 FWHM
        gd = np.where(dist2 < 100*sig2)
        data[gd[0],gd[1]] += amp*np.exp(-dist2[gd[0],gd[1]]/(2.*sig2))

    # add background
    data = data.astype(float) + back

    # add noise
    if noise or rn>0 :
        data = np.random.poisson(data)
        if rn>0 : data = data.astype(float) +  \
                         rn*np.random.normal(size=data.shape)

    return CCDData(data,unit='photon')
