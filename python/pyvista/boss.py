import numpy as np
import copy
import os
import pdb
import matplotlib.pyplot as plt
from pydl.pydlutils import yanny
from pyvista import imred, spectra, sdss
from tools import match,plots
from ccdproc import CCDData
import multiprocessing as mp
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time

def visit(planfile,clobber=False,maxobj=None,threads=12) :
    """ Reduce BOSS visit

        Driver for parallell processing of b and r channels
        Makes plots of median counts vs mag
    """
    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'
    mjd = int(plan['MJD'])

    # reduce b1 and r1 in parallel
    procs=[]
    for channel in [0,1] :
        kw={'planfile' : planfile, 'channel' : channel, 'clobber' : clobber, 'maxobj' : maxobj, 'threads' : threads}
        procs.append(mp.Process(target=visit_channel,kwargs=kw))
    for proc in procs : proc.start()
    for proc in procs : proc.join()

def visit_channel(planfile=None,channel=0,clobber=False,nfibers=500,threads=12,display=None,maxobj=None) :
    """ Read raw image (eventually, 2D calibration and extract,
        using specified flat/trace
    """
    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'

    # are all files already created?
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    done = True
    for obj in objs[0:maxobj] :
        name=plan['SPEXP']['name'][obj][channel].astype(str)
        if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
            name=plan['SPEXP']['name'][obj][2].astype(str)
        print(dir+name.replace('sdR','sp1D'))
        if not os.path.exists(dir+name.replace('sdR','sp1D')) or clobber :  done = False
    if done :  return

    # set up Reducer
    red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_N']+'/'+plan['MJD'])
   
    # make Trace/PSF 
    iflat=np.where(plan['SPEXP']['flavor'] == b'flat')[0]
    name=plan['SPEXP']['name'][iflat][0][channel].astype(str)
    if len(plan['SPEXP']['name'][iflat][0]) > 2  and channel == 1 :
        name=plan['SPEXP']['name'][iflat][0][2].astype(str)
    if os.path.exists(dir+name.replace('sdR','spTrace')) and not clobber : 
        trace=spectra.Trace(dir+name.replace('sdR','spTrace')) 
        f=np.loadtxt(dir+name.replace('sdR','spFlat1d').replace('.fit','.txt'))
    else :
        flat=red.reduce(name,channel=0)
        trace=spectra.Trace(transpose=red.transpose,rad=3,lags=np.arange(-3,4))
        ff=np.sum(flat.data[2000:2100],axis=0)
        if channel==0 : thresh=0.4e6
        else : thresh=0.2e6
        peaks,fiber=spectra.findpeak(ff,thresh=thresh,diff=10,bundle=20)
        print('found {:d} peaks'.format(len(peaks)))
        trace.trace(flat,peaks[0:nfibers],index=fiber[0:nfibers],skip=20,
                    plot=display)
        trace.write(dir+name.replace('sdR','spTrace')) 
        flat1d=trace.extract(flat,threads=threads,nout=500,plot=display)
        f=np.median(flat1d.data[:,1500:2500],axis=1)
        f/=np.median(f)
        np.savetxt(dir+name.replace('sdR','spFlat1d').replace('.fit','.txt'),f)

    # wavelength calibration
    chan = ['b1','r1']
    iarc=np.where(plan['SPEXP']['flavor'] == b'arc')[0]
    name=plan['SPEXP']['name'][iarc][0][channel].astype(str)
    if len(plan['SPEXP']['name'][iarc][0]) > 2  and channel == 1 :
        name=plan['SPEXP']['name'][iarc][0][2].astype(str)
    if os.path.exists(dir+name.replace('sdR','spWave')) and not clobber : 
        wavs=[]
        rows=[]
        wfits=fits.open(dir+name.replace('sdR','spWave')) 
        for w in wfits[1:] :
            wav= spectra.WaveCal(w.data)
            wavs.append(wav)
            rows.append(wav.index)
    else :
        im=red.reduce(name,channel=channel)
        arcec=trace.extract(im,threads=threads,nout=500,plot=display)
        wav=spectra.WaveCal('BOSS/BOSS_{:s}_waves.fits'.format(chan[channel]))
        wavs=[]
        rows=[]
        for irow in range(250,500) :
            if irow in trace.index :
                wav.identify(arcec[irow],plot=None,thresh=5,rad=5)
                wavs.append(copy.deepcopy(wav))
                rows.append(irow)

        wav=spectra.WaveCal('BOSS/BOSS_{:s}_waves.fits'.format(chan[channel]))
        for irow in range(249,-1,-1) :
            if irow in trace.index :
                wav.identify(arcec[irow],plot=None,thresh=5)
                wavs.append(copy.deepcopy(wav))
                rows.append(irow)
        wavs[0].index = rows[0]
        wavs[0].write(dir+name.replace('sdR','spWave'))
        for wav,row in zip(wavs[1:],rows[1:]) :
            wav.index = row
            wav.write(dir+name.replace('sdR','spWave'),append=True)
    rows = np.array(rows)

    # reduce and extract science frames
    # 1d flat
    fim=np.tile(f,(4224,1)).T
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    for obj in objs[0:maxobj] :
        name=plan['SPEXP']['name'][obj][channel].astype(str)
        if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
            name=plan['SPEXP']['name'][obj][2].astype(str)
        if os.path.exists(dir+name.replace('sdR','sp1D')) and not clobber : 
            out=CCDData.read(dir+name.replace('sdR','sp1D'))
        else :
            im=red.reduce(name,channel=channel)
            out=trace.extract(im,threads=threads,nout=500,plot=display)
            out.data /= fim
            out.uncertainty.array /= fim
            if int(plan['MJD']) > 59600 :
                plug,header,sky,stan = sdss.getconfig(config_id=out.header['CONFID'],specid=1)
            else :
                plug,header,sky,stan = sdss.getconfig(plugid=plan['SPEXP']['mapname'][objs[0]].astype(str),specid=1)

            # sky subtraction
            for fiber in range(1,501) :
                skyclose=np.where(np.abs(plug['fiberId'][sky]-fiber) < 50) [0]
                skyfibers=plug['fiberId'][sky[skyclose]]
                wave_index = np.where(rows == fiber-1)[0]
                if len(wave_index) > 0 :
                    wave_object = wavs[wave_index[0]].wave(image=out.data[fiber-1].shape)
                    sky_specs = []
                    for skyfiber in skyfibers :
                        sky_index = np.where(rows == skyfiber-1)[0]
                        if len(sky_index) > 0 :
                            sky_spec=wavs[sky_index[0]].scomb(out[skyfiber-1],wave_object)
                            sky_specs.append(sky_spec.data)
                    out.data[fiber-1] -= np.median(np.array(sky_specs),axis=0)

            out.write(dir+name.replace('sdR','sp1D'),overwrite=True)

    # populate wavelength image
    wave = np.full_like(out.data,np.nan)
    for j,row in enumerate(wave) :
        irow = np.where(rows == j)[0]
        if len(irow) > 0 :
            wave[j] = wavs[irow[0]].wave(image=row.shape)

    return out
