from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
import esutil
import glob
import numpy as np
import pdb
import os
from pydl.pydlutils.yanny import yanny
from tools import plots, match
from ccdproc import CCDData
import matplotlib.pyplot as plt
import multiprocessing as mp
import yaml
from pyvista import imred, spectra,sdss

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

    return data3d

def cds(file,dark=None) :
    """ CDS extraction of a cube
    """
    header = fits.open(file)[1].header
    cube = unzip(file)
    if dark is not None :
        try :
            cube = cube.astype(np.float32) - dark[0:len(cube)]
        except:
            print('not enough reads in dark')
            pdb.set_trace()
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

def visit(planfile,tracefile=None,clobber=False,db=None,schema='obs2',maxobj=None,threads=16) :
    """ Reduce an APOGEE visit
 
        Driver to do 3 chips in parallel
        Makes median flux plots
    """    

    # reduce channels in parallel
    chan=['a','b','c' ]
    procs=[]
    for channel in [1] :
        kw={'planfile' : planfile, 'channel' : channel, 'clobber' : clobber, 'maxobj' : maxobj, 'threads' : threads}
        procs.append(mp.Process(target=visit_channel,kwargs=kw))
    for proc in procs : proc.start()
    for proc in procs : proc.join()
    
def visit_channel(planfile=None,channel=0,clobber=False,nfibers=300,threads=24,maxobj=None,display=None) :
    """ Read raw image (eventually, 2D calibration) and extract,
        using specified flat/trace
    """
    chan=['a','b','c' ]
    plan=yaml.load(open(planfile,'r'), Loader=yaml.BaseLoader)
    dir=os.path.dirname(planfile)+'/'

    # are all files already created?
    done =True
    for obj in plan['APEXP'][0:maxobj] :
        exp_no = int(obj['name'])
        if obj['flavor']  != 'object' : continue
        name='ap1D-{:s}-{:08d}.fits'.format(chan[channel],exp_no)
        if not os.path.exists(dir+name) or clobber :  done = False
    if done :  return

    # set up Reducer
    red=imred.Reducer('APOGEE',dir=os.environ['APOGEE_DATA_N']+'/'+str(plan['mjd']))

    # get Dark
    if int(plan['darkid']) > 0 :
        name='apDark-{:s}-{:08d}.fits'.format(chan[channel],int(plan['darkid']))
        try :
           dark=fits.open('{:s}/{:s}/cal/{:s}/darkcorr/{:s}'.format(os.environ['APOGEE_REDUX'],plan['apogee_drp_ver'],plan['instrument'],name))[1].data
        except :
           dark=fits.open('/uufs/chpc.utah.edu/common/home/sdss/dr17/apogee/spectro/redux/dr17/cal/darkcorr/{:s}'.format(name))[1].data
    else : dark = None

    # get Trace/PSF if needed
    name='apTrace-{:s}-{:08d}.fits'.format(chan[channel],int(plan['psfid']))
    if os.path.exists(dir+name) and not clobber : 
        trace=spectra.Trace(dir+name)
    else :
        flat=red.reduce(int(plan['psfid']),channel=channel,dark=dark)
        trace=spectra.Trace(transpose=red.transpose,rad=2,lags=np.arange(-3,4))
        ff=np.sum(flat.data[:,1000:1100],axis=1)
        if channel==0 : thresh=40000
        else : thresh=40000
        peaks,fiber=spectra.findpeak(ff,diff=10,bundle=10000,thresh=thresh)
        print('found {:d} peaks'.format(len(peaks)))
        trace.trace(flat,peaks[0:nfibers],index=fiber[0:nfibers],skip=4)
        trace.write(dir+name)

    # now reduce and extract flux
    name='ap1D-{:s}-{:08d}.fits'.format(chan[channel],int(plan['fluxid']))
    print('flux: ', name)
    if os.path.exists(dir+name) and not clobber : 
        flux=CCDData.read(dir+name)
    else :
        im=red.reduce(int(plan['fluxid']),channel=channel,dark=dark)
        flux=trace.extract(im,threads=threads,nout=300)
        flux.write(dir+name,overwrite=True)
    f=np.median(flux.data[:,500:1500],axis=1)
    f/=np.median(f)
    np.savetxt(dir+name.replace('.fits','.txt'),f)
    fim=np.tile(f,(2048,1)).T

    # wavelength calibration
#    chan = ['a','b','c']
#    name='apWave-{:s}-{:08d}.fits'.format(chan[channel],int(plan['waveid']))
#    if os.path.exists(dir+name) and not clobber :
#        wavs=[]
#        rows=[]
#        wfits=fits.open(dir+name)
#        for w in wfits[1:] :
#            wav= spectra.WaveCal(w.data)
#            wavs.append(wav)
#            rows.append(wav.index)
#    else :
#        im=red.reduce(int(plan['waveid']),channel=channel)
#        arcec=trace.extract(im,threads=threads,nout=500,plot=display)
#        wav=spectra.WaveCal('APOGEE/APOGEE_{:s}_waves.fits'.format(chan[channel]))
#        wavs=[]
#        rows=[]
#        for irow in range(150,300) :
#            if irow in trace.index :
#                wav.identify(arcec[irow],plot=None,thresh=5,rad=5)
#                wavs.append(copy.deepcopy(wav))
#                rows.append(irow)
#
    #    wav=spectra.WaveCal('APOGEE/APOGEE_{:s}_waves.fits'.format(chan[channel]))
    #    for irow in range(149,-1,-1) :
    #        if irow in trace.index :
    #            wav.identify(arcec[irow],plot=None,thresh=5)
    #            wavs.append(copy.deepcopy(wav))
    #            rows.append(irow)
    #    wavs[0].index = rows[0]
    #    wavs[0].write(dir+name)
    #    for wav,row in zip(wavs[1:],rows[1:]) :
    #        wav.index = row
    #        wav.write(dir+name,append=True)
    #rows = np.array(rows)

    # now reduce and extract object
    for obj in plan['APEXP'][0:maxobj] :
        exp_no = int(obj['name'])
        if obj['flavor']  != 'object' : continue
        name='ap1D-{:s}-{:08d}.fits'.format(chan[channel],exp_no)
        if os.path.exists(dir+name) and not clobber : 
            out=CCDData.read(dir+name)
        else :
            im=red.reduce(exp_no,channel=channel,dark=dark,display=display)
            out=trace.extract(im,threads=threads,nout=300,display=display)
            out.data /= fim
            out.uncertainty.array /= fim
            out.write(dir+name,overwrite=True)

    return out
