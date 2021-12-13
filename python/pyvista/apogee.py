from astropy.io import fits
import astropy.units as u
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

def config(cid,specid=2) :
    """ Get FIBERMAP structure from configuration file for specified instrument
    """
    if isinstance(cid,str):
        conf = yanny(cid)
    else :
        conf = yanny('/home/sdss5/software/sdsscore/main/apo/summary_files/{:04d}XX/confSummary-{:d}.par'.format(cid//100,cid))

    gd =np.where((conf['FIBERMAP']['spectrographId'] == specid) & (conf['FIBERMAP']['fiberId'] > 0) )[0]
    return conf['FIBERMAP'][gd]

def flux(im,inst='APOGEE') :
    """ Plots of median flux vs various
    """
    flux = np.median(im.data,axis=1)
    if inst == 'APOGEE' :
        print(im.header['CONFIGFL'])
        fil=os.path.basename(im.header['FILENAME'])
        cid=os.path.basename(im.header['CONFIGFL'])
        des=im.header['DESIGNID']
        c = config(im.header['CONFIGFL'])
        i1, i2 = match.match(300-np.arange(300),c['fiberId'])
        mag=c['h_mag']
    else :
        print(im.header['CONFID'])
        cid=str(im.header['CONFID'])
        des=im.header['DESIGNID']
        c = config(im.header['CONFID'],specid=1)
        i1, i2 = match.match(1+np.arange(500),c['fiberId'])
        mag=c['mag'][:,1]
    print('found match for {:d} fibers'.format(len(i1)))
    fig,ax=plots.multi(2,2)
    fig.suptitle('File: {:s}  Design: {:d}   Config: {:s}'.format(fil,des,cid))
    plots.plotp(ax[0,0],c['fiberId'][i2],flux[i1],xt='fiberId',yt='flux',color='r')
    assigned=np.where(c['assigned'][i2] == 1)[0]
    plots.plotp(ax[0,0],c['fiberId'][i2[assigned]],flux[i1[assigned]],color='g',size=20)
    sky=np.where(np.char.find(c['category'][i2],b'sky') >=0)[0]
    plots.plotp(ax[0,0],c['fiberId'][i2[sky]],flux[i1[sky]],color='b',size=20)

    gd=np.where(mag[i2] > 0)[0]
    plots.plotp(ax[0,1],mag[i2[gd]],-2.5*np.log10(flux[i1[gd]]),xt='mag',yt='-2.5*log10(flux)')
    #plots.plotc(ax[1,1],c['xFocal'][i2[gd]],c['yFocal'][i2[gd]],mag[i2[gd]]+2.5*np.log10(flux[i1[gd]]),xt='xFocal',yt='yFocal',size=3)
    gd=np.where(flux[i1] > 100)[0]
    plots.plotc(ax[1,1],c['xFocal'][i2[gd]],c['yFocal'][i2[gd]],mag[i2[gd]]+2.5*np.log10(flux[i1[gd]]),xt='xFocal',yt='yFocal',size=20,colorbar=True)
    fig.tight_layout()
    pdb.set_trace()
    plt.close()
