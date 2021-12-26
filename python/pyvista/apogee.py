from astropy.io import fits
import astropy.units as u
import glob
import numpy as np
import pdb
import os
from pydl.pydlutils.yanny import yanny
from tools import plots, match
from ccdproc import CCDData
import matplotlib.pyplot as plt

def unzip(file,dark=None) :
    """ Read APOGEE .apz file, get CDS image
    """
    # open file and confirm checksums
    hd=fits.open(file, do_not_scale_image_data = True, uint = True, checksum = True)

    # file has initial header, avg_dcounts, then nreads
    nreads = len(hd)-2
    try:
        avg_dcounts=hd[1].data
    except:
        # fix header if there is a problem (e.g., MJD=55728, 01660046)
        hd[1].verify('fix')
        avg_dcounts=hd[1].data

    # first read is in extension 2
    ext = 2

    # loop over reads, processing into raw reads, and appending
    for read in range(1,nreads+1) :
        header = hd[ext].header
        try:
          raw = hd[ext].data
        except:
          hd[ext].verify('fix')
          raw = hd[ext].data
        if read == 1 :
          data = np.copy(raw)
          data3d=np.zeros([nreads,2048,2048],dtype=np.int16)
          data3d[0]=data[0:2048,0:2048]
        else :
          data = np.add(data,raw,dtype=np.int16)
          data = np.add(data,avg_dcounts,dtype=np.int16)
          data3d[read-1]=data[0:2048,0:2048]

        ext += 1

      # compute and add the cdsframe, subtract dark if we have one
    if dark is not None :
        # if we don't have enough reads in the dark, do nothing
        try :
            data3d -= dark[0:nreads]
        except:
            print('not halting: not enough reads in dark, skipping dark subtraction for mjdcube')
            pass

    return data3d

def cds(file,dark=None) :
    """ CDS extraction of a cube
    """
    header = fits.open(file)[1].header
    cube = unzip(file,dark=dark)
    out= (cube[-1,0:2048,0:2048] - cube[1,0:2048,0:2048] ).astype(np.float32)
    return CCDData(data=vert(out),header=header,unit=u.dimensionless_unscaled)

def vert(data) :
    """ Vertical bias subtraction from reference pixels
    """ 
    for i in range(4) :
        top = np.median(data[2044:2048,i*512:(i+1)*512])
        bottom = np.median(data[0:4,i*512:(i+1)*512])
        data[:,i*512:(i+1)*512]-=(top+bottom)/2.

    return data

