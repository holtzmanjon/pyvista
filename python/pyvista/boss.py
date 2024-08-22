import numpy as np
import copy
import os
import pdb
import matplotlib
import matplotlib.pyplot as plt
from pydl.pydlutils import yanny
from pyvista import imred, spectra, sdss, gaia, bitmask, stars, image
try :import gaiaxpy
except : print('no gaiaxpy...!')
from pyvista.dataclass import Data
from holtztools import match,plots,html
from scipy.ndimage import median_filter
import multiprocessing as mp
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
from astropy.io.votable import parse_single_table
import scipy.interpolate

crbox=None

def visit(planfile,clobber=False,maxobj=None,threads=12,flux=True, docombine=True) :
    """ Reduce BOSS visit

        Driver for parallell processing of b and r channels
        Makes plots of median counts vs mag
    """
    if not os.path.exists(planfile) :
        raise ValueError('no such file: {:s}'.format(planfile))
    plan=yanny.yanny(planfile)
    mjd = int(plan['MJD'])
    if plan['OBSERVATORY'] == 'apo' : channels= [0,1]
    else : channels = [2,3]


    # reduce b1 and r1 in parallel
    procs=[]
    for channel in channels :
        kw={'planfile' : planfile, 'channel' : channel, 'clobber' : clobber, 
                'maxobj' : maxobj, 'threads' : threads, 'plot' : False,
                'flux' : flux}
        procs.append(mp.Process(target=visit_channel,kwargs=kw))
    for proc in procs : proc.start()
    for proc in procs : proc.join()
    if docombine :
        combine(planfile)
        mkhtml(planfile)

def visit_channel(planfile=None,channel=0,clobber=False,threads=12,plot=True,
                  display=None,maxobj=None,skysub=True,flux=True) :
    """ Read raw image (eventually, 2D calibration and extract,
        using specified flat/trace
    """
    plan=yanny.yanny(planfile)
    outdir = os.path.dirname(planfile)+'/' if os.path.dirname(planfile) != '' else './'

    # are all files already created?
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    if len(objs) == 0 : return
    done = True
    for obj in objs[0:maxobj] :
        if len(plan['SPEXP']['name'][obj]) > 2  :
            name=plan['SPEXP']['name'][obj][channel].astype(str)
        else :
            name=plan['SPEXP']['name'][obj][channel%2].astype(str)
        print(outdir+name.replace('sdR','sp1D'))
        if not os.path.exists(outdir+name.replace('sdR','sp1D')) or clobber :  done = False
    if done : 
        return Data.read(outdir+name.replace('sdR','sp1D'))

    # set up Reducer
    nfibers=500
    bundle=20
    diff=8
    if plan['OBSERVATORY'] == 'apo' : 
        red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_N']+'/'+plan['MJD'])
        chan = ['b1','r1']
    else :
        red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_S']+'/'+plan['MJD'])
        chan = ['b2','r2']
        if int(plan['MJD']) > 59864 :
            bundle=10000
            diff=8
            nfibers=527

    # get Trace/PSF/flat1D
    iflat=np.where(plan['SPEXP']['flavor'] == b'flat')[0]
    if len(plan['SPEXP']['name'][iflat][0]) > 2 :
        name=plan['SPEXP']['name'][iflat][0][channel].astype(str)
    else :
        name=plan['SPEXP']['name'][iflat][0][channel%2].astype(str)
    trace,flat1d=mktrace(red,name,channel=channel,clobber=clobber,display=display,threads=threads,
                         bundle=bundle,diff=diff,nfibers=nfibers,outdir=outdir)
    # 1d flat to 2D
    #fim=np.tile(flat1d,(4224,1)).T

    # get wavelength calibration
    print('wave')
    iarc=np.where(plan['SPEXP']['flavor'] == b'arc')[0]
    if len(plan['SPEXP']['name'][iarc][0]) > 2 :
        name=plan['SPEXP']['name'][iarc][0][channel].astype(str)
    else :
        name=plan['SPEXP']['name'][iarc][0][channel%2].astype(str)
    wave= mkwave(red,trace,name,channel=channel,clobber=clobber,threads=threads,display=display,outdir=outdir,plot=False)

    # reduce and extract science frames
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    trace0=copy.deepcopy(trace)
    for obj in objs[0:maxobj] :
        if len(plan['SPEXP']['name'][obj]) > 2  :
            name=plan['SPEXP']['name'][obj][channel].astype(str)
        else :
            name=plan['SPEXP']['name'][obj][channel%2].astype(str)
        if os.path.exists(outdir+name.replace('sdR','sp1D')) and not clobber : 
            out=Data.read(outdir+name.replace('sdR','sp1D'))
        else :
            im=red.reduce(name,channel=channel,trim=True,crbox=crbox,crsig=10)
            trace=copy.deepcopy(trace0)
            if channel > 1 :
                trace.rows=[0,2048]
                trace.find(im,display=display,rad=2)
                for imod,mod in enumerate(trace.model) :
                    if mod.c0 < 2048 : trace.model[imod].c0 += trace.pix0
                trace.rows=[2049,4095]
                trace.find(im,display=display,rad=2)
                for imod,mod in enumerate(trace.model) :
                    if mod.c0 > 2048 : trace.model[imod].c0 += trace.pix0
                trace.pix0 = 0.
            else :
                trace.find(im,display=display,rad=2)

            out=trace.extract(im,threads=threads,nout=nfibers,plot=display,fit=False,rad=2)

            #out_fit=trace.extract(im,threads=threads,nout=nfibers,plot=display,fit=True)
            # 1d fiber-to-fiber flat
            out.data /= (flat1d.data/np.median(flat1d.data,axis=0))
            out.uncertainty.array /= (flat1d.data/np.median(flat1d.data,axis=0))

            #add wavelength
            out.wave = wave

            # need plugfile info for sky subtract and flux calibration
            if int(plan['MJD']) > 59600 :
                plug,header,sky,stan = sdss.getconfig(config_id=out.header['CONFID'],
                                                      specid=1,obs=plan['OBSERVATORY'])
            else :
                plug,header,sky,stan = sdss.getconfig(plugid=plan['SPEXP']['mapname'][objs[0]].astype(str),
                                                      specid=1,obs=plan['OBSERVATORY'])

            if skysub: skysubtract(out,plug['fiberId'][sky],nfibers=nfibers,display=display)
            if flux : mkflux(out,plug,planfile,channel=channel,plot=plot,display=display)

            out.write(outdir+name.replace('sdR','sp1D'),overwrite=True)

    return out

def skysubtract(out,skyfibers,display=None,dmax=100,nfibers=500) :
    """ Sky subtract data frame, using sky fibers

        Resamples sky fibers to wavelength of each object and subtract
        Sky fibers must be within dmax fibers along slit
        Median of these sky fibers is subtracted
        No uncertainty is propagated!
    """
    # sky subtraction
    raw=copy.deepcopy(out)
    if display is not None : 
        display.tv(out)
        fig,ax=plots.multi(1,2,hspace=0.001,sharex=True)
    out.add_sky(np.zeros_like(out.data))
    for fiber in range(1,nfibers) :
        skyclose=np.where(np.abs(skyfibers-fiber) < dmax) [0]
        if display is not None :
            print(fiber,skyfibers[skyclose])
            ax[0].plot(out.data[fiber-1])
        sky_specs = []
        for skyfiber in skyfibers[skyclose] :
            # get sky spectrum sampled at wavelengths of this fiber by interpolation
            sky_spec = np.interp(out.wave[fiber-1],raw.wave[skyfiber-1],raw.data[skyfiber-1])
            sky_specs.append(sky_spec)
            if display is not None: ax[1].plot(sky_spec)
        sky_fiber = np.nanmedian(np.array(sky_specs),axis=0)
        out.data[fiber-1] -= sky_fiber
        out.sky[fiber-1] = sky_fiber
        if display is not None :
            ax[0].plot(np.nanmedian(np.array(sky_specs),axis=0))
            ax[0].plot(out.data[fiber-1])
            plt.draw()
            ax[0].cla()
            ax[1].cla()


def mktrace(red,name,channel=0,clobber=False,nfibers=500,threads=0,display=None,skip=100,bundle=20,diff=11,outdir='./',new=True) :
    """ Create spTrace and spFlat1D files or read if they already exist
    """

    chan = ['b1','r1','b2','r2']
    outname=outdir+name.replace('sdR','spTrace')

    if os.path.exists(outname) and not clobber :
        trace=spectra.Trace(outname)
        #f=np.loadtxt(outname.replace('spTrace','spFlat1d').replace('.fit','.txt'))
        flat1d=Data.read(outname.replace('spTrace','spFlat1D'))
    else :
        print('creating Trace')
        flat=red.reduce(name,trim=True,channel=channel,crbox=crbox,crsig=10)

        if new :
            trace=spectra.Trace(transpose=red.transpose,rad=3,lags=np.arange(-3,4),sc0=2048,rows=[10,4090],degree=4)
            peaks,fiber=trace.findpeak(flat,diff=diff,bundle=bundle,thresh=75,smooth=2)
            print('found {:d} peaks'.format(len(peaks)))
            trace.trace(flat,peaks[0:nfibers],index=fiber[0:nfibers],skip=skip,
                        display=display,gaussian=True,thresh=100)
        else :
            trace=spectra.Trace('BOSS/BOSS_{:s}_trace.fit'.format(chan[channel]))
            trace.retrace(flat,skip=skip,display=display,gaussian=True,thresh=100)
        trace.write(outname) 
        flat1d=trace.extract(flat,threads=threads,nout=nfibers,display=display)
        flat1d.write(outname.replace('spTrace','spFlat1D'))

    return trace,flat1d

def mkwave(red,trace,name,channel=0,threads=0,clobber=False,display=None,plot=False,outdir='./') :
    """ Create spWave files or read if they already exist

        Return 2D wavelength image
    """

    chan = ['b1','r1','b2','r2']
    outname=outdir+name.replace('sdR','spWave')

    if display is not None : plot = True

    nfibers=trace.index.max()+1
    if os.path.exists(outname) and not clobber : 
        wav= spectra.WaveCal(outname)
        #wavs=[]
        #rows=[]
        #wfits=fits.open(outname) 
        #for w in wfits[1:] :
        #    wav= spectra.WaveCal(w.data)
        #    wavs.append(wav)
        #    rows.append(wav.index)
        im=red.reduce(name,channel=channel,trim=True,crbox=crbox,crsig=10)
        wave= wav.wave(image=(nfibers,im.data.shape[0]))
        #arcec=trace.extract(im,threads=threads,nout=nfibers,display=display)
    else :
        print('creating WaveCal')
        im=red.reduce(name,channel=channel,trim=True,crbox=crbox,crsig=10)
        arcec=trace.extract(im,threads=threads,nout=nfibers,display=display)
        wav0=spectra.WaveCal('BOSS/BOSS_{:s}_waves.fits'.format(chan[channel]))
        ngd=len(np.where(wav0.weights >0)[0])
        wav=copy.deepcopy(wav0)

        # get quick 2D solution from central regions of chip, use it to get initial wavelength estimates for  whole chip
        tmp=copy.deepcopy(arcec)
        tmp.data[0:150,:]= 0.
        tmp.data[-150:,:]= 0.
        wav.identify(tmp,thresh=20,rad=3,maxshift=10,nskip=1,display=display,plot=plot)

        # Now do the 2D solution with individual fiber shifts
        wav.type = 'chebyshev2D_shift'
        wav.identify(arcec,wav=wav.wave(image=arcec.data.shape),thresh=20,rad=3,maxshift=10,plot=plot,nskip=1,display=display)
        wav.write(outname)
        wave= wav.wave(image=arcec.shape)

        # old 1D wavelength solutions
        #wavs=[]
        #rows=[]
        #for irow in range(250,nfibers) :
        #    if irow in trace.index :
        #        print(irow)
        #        wav=copy.deepcopy(wav0)
        #        if irow % 5 == 0 : plot=True
        #        else : plot=False
        #        wav.identify(arcec[irow],plot=plot,thresh=20,rad=5,maxshift=10)
        #        if plot :
        #            plt.close(wav.fig)
        #            delattr(wav,'ax')
        #            delattr(wav,'fig')
        #        wavs.append(copy.deepcopy(wav))
        #        if len(np.where(wav.weights>0)[0]) == ngd : wav0 = copy.deepcopy(wav)
        #        rows.append(irow)
        #wav0=spectra.WaveCal('BOSS/BOSS_{:s}_waves.fits'.format(chan[channel]))
        #for irow in range(249,-1,-1) :
        #    if irow in trace.index :
        #        print(irow)
        #        wav=copy.deepcopy(wav0)
        #        if irow % 5 == 0 : plot=True
        #        else : plot=False
        #        wav.identify(arcec[irow],plot=plot,thresh=20,rad=5,maxshift=10)
        #        if plot :
        #            plt.close(wav.fig)
        #            delattr(wav,'ax')
        #            delattr(wav,'fig')
        #        wavs.append(copy.deepcopy(wav))
        #        if len(np.where(wav.weights>0)[0]) == ngd : wav0 = copy.deepcopy(wav)
        #        rows.append(irow)
        #wavs[0].index = rows[0]
        #wavs[0].write(outname)
        #for wav,row in zip(wavs[1:],rows[1:]) :
        #    wav.index = row
        #    wav.write(outname,append=True)
    #rows = np.array(rows)
    ## populate wavelength image, remember that BOSS is transposed, so axis 0 in 2D is wavelength
    #wave = np.empty((nfibers,im.data.shape[0]))
    #wave[:,:] = np.nan
    #for j,row in enumerate(wave) :
    #    irow = np.where(rows == j)[0]
    #    if len(irow) > 0 :
    #        wave[j] = wavs[irow[0]].wave(image=row.shape)

    return wave

def mkflux(out,plug,planfile,medfilt=25,plot=True,channel=0,
           minstandards=5,display=None) :

    """ Do flux calibration from GAIA spectra
    """
    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'
    name=os.path.basename(planfile)

    # get GAIA data
    print('getting gaia')
    gaia_posn=dir+name.replace('spPlan2d','spGaiaPosn').replace('.par','.xml')
    gaia_flux=dir+name.replace('spPlan2d','spGaiaFlux').replace('.par','.xml')
    if os.path.exists(gaia_posn) :
        g=parse_single_table(gaia_posn).array
        xc=parse_single_table(gaia_flux) #.array
        x,w=gaiaxpy.calibrate(xc.to_table().to_pandas())
    else :
        print('gaia query')
        g,xold,xc=gaia.get(plug['ra'],plug['dec'],vers='dr3_tap',posn_match=5,cols=[[plug['fiberId'],'fiberId']])
        x,w=gaiaxpy.calibrate(xc.to_table().to_pandas())
        g._votable.to_xml(gaia_posn)
        xc._votable.to_xml(gaia_flux)
    i1,i2=match.match(g['source_id'],x['source_id'])
    j1,j2=match.match(plug['fiberId'],g['fiberId'][i1])
    print('Number of standards: ', len(j1))
    if len(j1) < minstandards : return

    w = np.linspace(3360,10200,343)

    # set wavelength ranges for blue and red channels
    if channel == 0 or channel == 2 :
        wav=np.where((w>3500)&(w<7000))[0]
        wws = [4000,5000,6000] 
    else :
        wav=np.where((w>5500)&(w<10200))[0]
        wws = [6500,7500,8500]

    # add the stars for which we have GAIA spectra
    flx=spectra.FluxCal(degree=-1,median=True)
    # reject stars with bad pixels, but not corrected CRs
    pixmask=bitmask.PixelBitMask()
    badval=pixmask.badval()-pixmask.getval('CRPIX')
    for ind,j in enumerate(j1) :
        try :
            if plug['delta_ra'][j] > 0 or plug['delta_dec'][j] > 0 : continue
        except : pass
        row = plug['fiberId'][j]-1
        # if we don't have wavecal for this spectrum, skip
        if np.isnan(out.wave[row][0]) : continue
        flux = x['flux'][i2[j2[ind]]]
        stdflux=Table()
        stdflux['wave'] = w[wav]
        stdflux['flux'] = flux[wav]
        stdflux['bin'] = 20
        flx.addstar(out[row],out.wave[row],stdflux=stdflux,badval=badval,
                    extinct=False,pixelmask=out.bitmask[row])

    # make the response curve
    flx.response(legend=False,medfilt=medfilt,plot=plot)

    if plot :
        # plot cross-sections
        fig,ax=plots.multi(1,1)
        for ww in wws :
            j=np.where(w[wav]==ww)[0]
            rat=-2.5*np.log10(np.array(flx.obscorr)[:,j]/np.array(flx.true)[:,j])
            bins=np.arange(np.nanmedian(rat)-0.5,np.nanmedian(rat)+0.5,0.05)
            ax.hist(rat, histtype='step', label='{:d}'.format(ww),bins=bins)
        ax.legend(fontsize='x-small')
        ax.set_xlabel('zeropoint (mag)')
        fig.tight_layout()

    # apply flux curves
    flx.correct(out,out.wave,extinct=False)

    return

def combine(planfile,wnew=10.**(np.arange(3.5589,4.0151,1.e-4)),maxobj=None) :
    """ Combine two channels and resample to common wavelength scale
    """
    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'
    mjd = int(plan['MJD'])

    # get target information
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    if int(plan['MJD']) > 59600 :
        plug,header,sky,stan = \
            sdss.getconfig(config_id=plan['SPEXP']['mapname'][objs[0]].astype(int),specid=1,obs=plan['OBSERVATORY'])
    else :
        plug,header,sky,stan = sdss.getconfig(plugid=plan['SPEXP']['mapname'][objs[0]].astype(str),specid=1)

    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    coadd=[]
    for channel in [0,1] :
        for iobj,obj in enumerate(objs[0:maxobj]) :
            name=plan['SPEXP']['name'][obj][channel].astype(str)
            if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
                name=plan['SPEXP']['name'][obj][2].astype(str)
            im=Data.read(dir+name.replace('sdR','sp1D'))
            if iobj == 0 :
                sum = copy.deepcopy(im)
                sum.uncertainty.array = im.uncertainty.array**2
                sum.wave = im.wave
            else :
                sum.data += im.data
                sum.uncertainty.array += im.uncertainty.array**2
                sum.sky += im.sky
        sum.uncertainty.array = np.sqrt(sum.uncertainty.array)
        coadd.append(sum)

    comb = np.zeros([coadd[0].shape[0],len(wnew)])
    comberr = np.zeros([coadd[0].shape[0],len(wnew)])
    for irow in range(len(coadd[0].data)) :
        gd = np.where((coadd[0].wave[irow] > wnew[0]) & (coadd[0].wave[irow]<6200.) & np.isfinite(coadd[0].data[irow]))[0]
        if len(gd) == 0 : continue
        try :
          dspline = scipy.interpolate.CubicSpline(coadd[0].wave[irow,gd],coadd[0].data[irow,gd])
          vspline = scipy.interpolate.CubicSpline(coadd[0].wave[irow,gd],coadd[0].uncertainty.array[irow,gd]**2)
        except : continue
        bdata = dspline(wnew)
        bvar = vspline(wnew)
        bdata[np.where(wnew>6200)[0]] = 0.
        bvar[np.where(wnew>6200)[0]] = 1.e10

        gd = np.where((coadd[1].wave[irow] < wnew[-1]) & (coadd[1].wave[irow] > 6100.) & np.isfinite(coadd[1].data[irow]))[0]
        try :
          dspline = scipy.interpolate.CubicSpline(coadd[1].wave[irow,gd],coadd[1].data[irow,gd])
          vspline = scipy.interpolate.CubicSpline(coadd[1].wave[irow,gd],coadd[1].uncertainty.array[irow,gd]**2)
        except : continue
        rdata = dspline(wnew)
        rvar = vspline(wnew)
        rdata[np.where(wnew<6100)[0]] = 0.
        rvar[np.where(wnew<6100)[0]] = 1.e10

        comb[irow] = (bdata/bvar + rdata/rvar) / (1/bvar + 1/rvar)
        comberr[irow] = np.sqrt(1. / (1/bvar + 1/rvar))
    comb = Data(comb,uncertainty=comberr,wave=wnew,header=coadd[0].header)
    if int(plan['MJD']) > 59600 :
        comb.write('{:s}/spPlate-{:d}-{:d}.fit'.format(dir,comb.header['FIELDID'],comb.header['MJD']))
    else :
        comb.write('{:s}/spPlate-{:d}-{:d}.fit'.format(dir,comb.header['PLATEID'],comb.header['MJD']))


#    done = True
#    for obj in objs[0:maxobj] :
#        out=[]
#        for channel in [0,1] :
#            name=plan['SPEXP']['name'][obj][channel].astype(str)
#            if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
#                name=plan['SPEXP']['name'][obj][2].astype(str)
#            out.append(Data.read(dir+name.replace('sdR','sp1D')))
#
#        comb = np.zeros([out[0].shape[0],len(wnew)])
#        comberr = np.zeros([out[0].shape[0],len(wnew)])
#        for irow in range(len(out[0].data)) :
#            gd = np.where((out[0].wave[irow] > wnew[0]) & (out[0].wave[irow]<6200.) & np.isfinite(out[0].data[irow]))[0]
#            if len(gd) == 0 : continue
#            try :
#              dspline = scipy.interpolate.CubicSpline(out[0].wave[irow,gd],out[0].data[irow,gd])
#              vspline = scipy.interpolate.CubicSpline(out[0].wave[irow,gd],out[0].uncertainty.array[irow,gd]**2)
#            except : continue
#            bdata = dspline(wnew)
#            bvar = vspline(wnew)
#            bdata[np.where(wnew>6200)[0]] = 0.
#            bvar[np.where(wnew>6200)[0]] = 1.e10
#
#            gd = np.where((out[1].wave[irow] < wnew[-1]) & (out[1].wave[irow] > 6100.) & np.isfinite(out[1].data[irow]))[0]
#            try :
#              dspline = scipy.interpolate.CubicSpline(out[1].wave[irow,gd],out[1].data[irow,gd])
#              vspline = scipy.interpolate.CubicSpline(out[1].wave[irow,gd],out[1].uncertainty.array[irow,gd]**2)
#            except : continue
#            rdata = dspline(wnew)
#            rvar = vspline(wnew)
#            rdata[np.where(wnew<6100)[0]] = 0.
#            rvar[np.where(wnew<6100)[0]] = 1.e10
#
#            comb[irow] = (bdata/bvar + rdata/rvar) / (1/bvar + 1/rvar)
#            comberr[irow] = np.sqrt(1. / (1/bvar + 1/rvar))
#        comb = Data(comb,uncertainty=comberr,wave=wnew,header=out[0].header)
#        comb.write(dir+name.replace('sdR','spPlate').replace('-r1',''))

    return comb

def mkhtml(planfile,maxobj=None,channel=0,backend='Agg') :
    """ Create HTML file for visit
    """
    if not os.path.exists(planfile) :
        raise ValueError('no such file: {:s}'.format(planfile))
    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'
    mjd = int(plan['MJD'])

    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]

    if int(plan['MJD']) > 59600 :
        plug,header,sky,stan = \
            sdss.getconfig(config_id=plan['SPEXP']['mapname'][objs[0]].astype(int),specid=1,obs=plan['OBSERVATORY'])
    else :
        plug,header,sky,stan = sdss.getconfig(plugid=plan['SPEXP']['mapname'][objs[0]].astype(str),specid=1)

    matplotlib.use(backend)
    #for obj in objs[0:maxobj] :
    for icomb in range(1) :
        #name=plan['SPEXP']['name'][obj][1].astype(str)
        name = 'spPlate-{:d}-{:d}.fit'.format(int(plan['plateid']),mjd)
        fhtml=open(dir+name.replace('sdR','spPlate').replace('-r1','').replace('.fit','.html'),'w')
        fhtml.write('<HTML><BODY><TABLE BORDER=2>\n')
        fhtml.write('<TR><TD>Fiber<TD>CatalogID<TD>Category<TD>g<br>r<br>i<TD>Extracted spectra\n')

        # read the combined image
        #comb=Data.read(dir+name.replace('sdR','spPlate').replace('-r1',''))
        comb=Data.read(dir+name)

        # read the reduced sp1D images  (here only first one)
        out=[]
        obj=objs[0]
        for channel in [0,1] :
            name=plan['SPEXP']['name'][obj][channel].astype(str)
            if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
                name=plan['SPEXP']['name'][obj][2].astype(str)
            out.append(Data.read(dir+name.replace('sdR','sp1D')))


        try: os.mkdir(dir+'/plots')
        except: pass
        colors=['b','r']
        nfibers = len(comb.data)
        for fiber in range(1,nfibers+1) :
            j = np.where(plug['fiberId'] == fiber)[0]
            if len(j) == 0 : continue
            else : j=j[0]
            try :
                cat = plug['category'][j]
            except :
                cat = plug['objType'][j]
            try :
                catid = plug['catalogid'][j]
            except :
                catid = 0

            print(plug['fiberId'][j],cat)
            fhtml.write('<TR>\n')
            fhtml.write('<TD>{:d}\n'.format(fiber))
            fhtml.write('<TD>{:d}\n'.format(catid))
            fhtml.write('<BR>{:f}\n'.format(plug['ra'][j]))
            fhtml.write('<BR>{:f}\n'.format(plug['dec'][j]))
            fhtml.write('<TD>{:s}\n'.format(cat.decode()))
            fhtml.write('<TD>{:7.2f}<br>{:7.2f}<br>{:7.2f}\n'.format(*plug['mag'][j,1:4]))
            fig,ax=plots.multi(1,1,figsize=(8,6))
            gd=np.where(np.isfinite(comb.data[fiber-1]))[0]
            if len(gd) > 100 :
                plots.plotl(ax,comb.wave,comb.data[fiber-1])
                ymax = np.nanmax(median_filter(comb.data[fiber-1],100))
                try: ax.set_ylim(0,1.5*ymax)
                except: pass
            png = name.replace('sdR','spComb').replace('-r1','').replace('.fit','-{:03d}.png'.format(fiber))
            fig.savefig(dir+'/plots/'+png)
            plt.close()
            fhtml.write('<TD><A HREF=plots/{:s}><IMG SRC=plots/{:s}></A>\n'.format(png,png))

            fig,ax=plots.multi(1,1,figsize=(8,2))
            for channel in [0,1] :
                gd=np.where(np.isfinite(out[channel].data[fiber-1]))[0]
                if len(gd) > 100 :
                    plots.plotl(ax,out[channel].wave[fiber-1],out[channel].data[fiber-1],color=colors[channel])
                    ymax = np.nanmax(median_filter(out[channel].data[fiber-1],100))
                    try: ax.set_ylim(0,1.5*ymax)
                    except: pass
            png = name.replace('sdR','spPlate').replace('-r1','').replace('.fit','-{:03d}.png'.format(fiber))
            fig.savefig(dir+'/plots/'+png)
            plt.close()
            fhtml.write('<TD><A HREF=plots/{:s}><IMG SRC=plots/{:s}></A>\n'.format(png,png))

        fhtml.close()

def mkyaml(mjd,obs='apo') :
         
    if obs == 'apo' :
        red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_N']+'/'+str(mjd))
        if mjd < 59600 :
            files = red.log(cols=['DATE-OBS','PLATEID','FLAVOR','EXPTIME','HARTMANN','MAPID','NAME','PLATETYP'],
                            channel='-b1',hexdump=True)
            files['CONFID'] = files['NAME']
            files['FIELDID'] = files['PLATEID']
            if mjd < 59150 :
                gd=np.where(np.char.find(files['PLATETYP'],b'BOSS') >=0)[0]
                files=files[gd]
        else :
            files = red.log(cols=['DATE-OBS','FIELDID','FLAVOR','EXPTIME','HARTMANN','CONFID'],
                            channel='-b',hexdump=True)
    else :
        red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_S']+'/'+str(mjd))
        files = red.log(cols=['DATE-OBS','FIELDID','FLAVOR','EXPTIME','HARTMANN','CONFID'],
                        channel='-b',hexdump=True)

    files['FILE'] = np.char.replace(files['FILE'].astype(str),'.gz','')

    fields = set(files['FIELDID'])
    for field in fields :
        fp=open('spPlan2d-{:s}-{:d}.par'.format(field,mjd),'w')
        fp.write('plateid {:s}\n'.format(field))
        fp.write('MJD {:d}\n'.format(mjd))
        fp.write('OBSERVATORY {:s}\n'.format(obs))

        #planfile2d  'spPlan2d-20875-59571.par'  # Plan file for 2D spectral reductions (this file)
        #idlspec2dVersion 'v6_0_4'  # Version of idlspec2d when building plan file
        #idlutilsVersion 'v5_5_17'  # Version of idlutils when building plan file
        #speclogVersion 'trunk 27531'  # Version of speclog when building plan file

        fp.write('typedef struct {\n')
        fp.write('  int plateid;\n')
        fp.write('  int mjd;\n')
        fp.write('  char mapname[15];\n')
        fp.write('  char flavor[8];\n')
        fp.write('  float exptime;\n')
        fp.write('  char name[2][20];\n')
        fp.write('} SPEXP;\n')
        flat = np.where((files['FIELDID'] == field) & (files['FLAVOR'] == 'flat'))[0]
        if len(flat) == 0 :
          flat = np.where(files['FLAVOR'] == 'flat')[0]
        if len(flat) > 0 : 
            fp.write('SPEXP {:s} {:d} {:s} flat {:s} {{ {:s} {:s} }}\n'.format(
                     field,mjd, files[flat[0]]['CONFID'], files[flat[-1]]['EXPTIME'],
                     files[flat[-1]]['FILE'],files[flat[-1]]['FILE'].replace('-b','-r')))
        arc = np.where((files['FIELDID'] == field) & (files['FLAVOR'] == 'arc') & (files['HARTMANN'] == 'Out') )[0]
        if len(arc) > 0 : 
            fp.write('SPEXP {:s} {:d} {:s} arc {:s} {{ {:s} {:s} }}\n'.format(
                      field,mjd, files[arc[0]]['CONFID'], files[arc[0]]['EXPTIME'],
                      files[arc[0]]['FILE'],files[arc[0]]['FILE'].replace('-b','-r')))
        sci = np.where((files['FIELDID'] == field) & (files['FLAVOR'] == 'science'))[0]
        for s in sci :
            fp.write('SPEXP {:s} {:d} {:s} science {:s} {{ {:s} {:s} }}\n'.format(
                      field,mjd, files[s]['CONFID'], files[s]['EXPTIME'],
                      files[s]['FILE'],files[s]['FILE'].replace('-b','-r')))
        fp.close()


def arc_transform(mjd,obs='lco',refarc=None,thresh=400,nskip=40, rad=2,
                  clobber=False, outdir=None, threads=8, cams=None, 
                  vers='test/sean/v6_1_1-tracetweak', planfile=True, backend='Agg') :
    """ Get transformations from first arc for all arcs on a given MJD
        Make plots and HTML page

        Parameters
        ----------
        mjd  : int
               MJD to process
        obs  : str
               observatory = 'lco'|'apo', default='lco'
        refarc : str
               string identifying reference arc to use, default=None
               if None, use planfile or first arc if no planfile
        nskip : int
               use every nskip line, default=False
        clobber : bool
        outdir : str
        vers : str
               string for BOSS version, with planfile
        planfile : bool
               if True, use planfile, default=False
        backend : str
               default = 'Agg'
               
    """

    matplotlib.use(backend)

    grid=[]
    cams2channel = {'b1':0,'r1':1,'b2':2,'r2':3}
    if obs == 'lco' :
        data_env = 'BOSS_SPECTRO_DATA_S'
        if cams is None : cams = ['b2','r2']
        elif isinstance(cams,str) : cams = [cams]
        channels = [2,3]
    else :
        data_env = 'BOSS_SPECTRO_DATA_N'
        if cams is None : cams = ['b1','r1'] 
        elif isinstance(cams,str) : cams = [cams]
        channels = [0,1]
    channels = [cams2channel[cam] for cam in cams]

    if planfile :
        pfile = '{:s}/{:s}/trace/{:d}/spPlanTrace-{:d}_{:s}.par'.format( 
                 os.environ['BOSS_SPECTRO_REDUX'],vers,mjd,mjd,obs.upper())
        if not os.path.exists(pfile) :
            raise FileNotFoundError('no planfile: {:s}'.format(pfile))
        plan = yanny.yanny('{:s}/{:s}/trace/{:d}/spPlanTrace-{:d}_{:s}.par'.format(
                    os.environ['BOSS_SPECTRO_REDUX'],vers,mjd,mjd,obs.upper()))
        traceflat=np.where(plan['SPEXP']['flavor'] == b'TRACEFLAT')[0][0]
        tracearc=np.where(plan['SPEXP']['flavor'] == b'TRACEARC')[0][0]
        arcs=np.where(plan['SPEXP']['flavor'] == b'arc')[0]
        flats=np.where(plan['SPEXP']['flavor'] == b'flat')[0]
        outdir='{:s}/{:s}/trace/'.format(os.environ['BOSS_SPECTRO_REDUX'],vers)
    else :
        if outdir == None : outdir='./'
        try: os.makedirs('{:s}/{:d}'.format(outdir,mjd))
        except: pass

    boss=imred.Reducer('BOSS',dir=os.environ[data_env]+'/{:d}'.format(mjd))
    xt=[]
    for channel,cam in zip(channels,cams) :

        if planfile :
            flatfile = plan['SPEXP']['name'][traceflat][channel%2].astype(str)
            tracetab = fits.open('{:s}/{:s}/trace/{:d}/{:s}s.gz'.format(
                       os.environ['BOSS_SPECTRO_REDUX'],vers,mjd,flatfile.replace('sdR','spTraceFlat')))[5].data
            col0 = np.arange(len(tracetab[0]))

        if refarc is None :
            reffile = plan['SPEXP']['name'][tracearc][channel%2].astype(str)+'.gz'
        else :
            j = np.where(np.char.find(log['FILE'][arcs].astype(str),str(refarc)) >= 0)[0]
            reffile = log['FILE'][arcs[j]][0]

        im0=boss.reduce(reffile,channel=channel)
        print('finding lines in reference: ', reffile)
        lines0=stars.find(im0.data,thresh=thresh,
                          sharp=[0,0.5],round=[-0.25,0.75])[::nskip]
        print('automarking lines in reference: ', reffile)
        lines=stars.automark(im0.data,lines0,rad=rad,dx=0,dy=0,
                             background=False,func='marginal_gfit')

        if not planfile :
            log = boss.log(channel=cam)
            if len(log) == 0 : return
            arcs = np.where((log['FLAVOR'] == 'arc') & 
                            (log['HARTMANN'] == 'Out'))[0]
            if len(arcs) <= 1 : return

        col1=[]
        col2=[]
        yt=[]
        xt.append(cam)
        xt.append(cam)

        # process all of the arcs in parallel with multiprocessing pool if threads>0
        pars=[]
        for arc in arcs :
            # build up input parameters
            if planfile :
               arcfile = plan['SPEXP']['name'][arc][channel%2].astype(str)
            else :
               arcfile = os.path.basename(log['FILE'][arc])

            hard = '{:s}/{:d}/{:s}.png'.format( outdir,mjd,arcfile.replace('.fit.gz',''))
            if clobber or os.path.exists(hard.replace('.png','_compare.png'))==False :
                print(arcfile,channel)
                im=boss.reduce(arcfile+'.gz',channel=channel)
                pars.append((im0,im,lines,hard))

        # run the solutions for each arc
        if threads > 0 :
            pool = mp.Pool(threads)
            outputs = pool.map_async(transform_thread, pars).get()
            pool.close()
            pool.join()
        else :
            outputs=[]
            for par in pars :
                outputs.append(transform_thread(par))

        # create the outputs
        iout=0
        for arc in arcs :
            if planfile :
               arcfile = plan['SPEXP']['name'][arc][channel%2].astype(str)
            else :
               arcfile = os.path.basename(log['FILE'][arc])

            hard = '{:s}/{:d}/{:s}.png'.format( outdir,mjd,arcfile.replace('.fit.gz',''))
            col1.append('{:s}'.format(os.path.basename(hard)))
            yt.append(arcfile.split('.')[0].split('-')[2])
            if not clobber and  os.path.exists(hard.replace('.png','_compare.png')) :
                col2.append(os.path.basename(hard).replace('.png','_compare.png'))
                continue

            if outputs[iout] is None : 
                print('error with {:s}'.format(arcfile))
                col2.append('')
                continue

            linfit = outputs[iout][0]
            im = pars[iout][1]
            iout+=1

            if planfile :
                # calculate derived tracetable
                out=copy.copy(tracetab)
                for irow,row0 in enumerate(tracetab) :
                    fit = linfit(np.vstack([row0,col0]).T)
                    out[irow,:]= np.interp(np.arange(len(row0)),fit[:,1],fit[:,0])
                    #out[irow,:] = linfit(np.vstack([row0,col0]).T)[:,0]

                # write derived trace table
                hdu=fits.PrimaryHDU(out)
                hdu.writeto('{:s}/{:d}/{:s}'.format(outdir,mjd,
                    arcfile.replace('sdR','spTraceTab').replace('.fit','.fits')),overwrite=True)

                # plot difference between master trace and derived trace
                fig,ax=plots.multi(1,2)
                ax[0].set_title('difference between derived trace and {:s}'.format(flatfile))
                for a,b in zip(out,tracetab) :
                    ax[0].plot(a-b)
                ax[0].set_ylim(-5,5)
                ax[0].set_xlim(0,len(a))

                # if we have a flat at the field, compare it to derived trace
                field = plan['SPEXP']['fieldid'][arc]
                j=np.where(plan['SPEXP']['fieldid'][flats] == field)[0]
                if len(j) > 0 :
                    flat=plan['SPEXP']['name'][flats[j[-1]]][channel%2].astype(str)
                    try :
                        flat_data = fits.open('{:s}/{:s}/{:s}/{:s}s.gz'.format(
                           os.environ['BOSS_SPECTRO_REDUX'],vers,field.astype(str),
                           flat.replace('sdR','spFlat')))[0]
                        tab = fits.open('{:s}/{:s}/{:s}/{:s}s.gz'.format(
                           os.environ['BOSS_SPECTRO_REDUX'],vers,field.astype(str),
                           flat.replace('sdR','spFlat')))[5].data
                        ax[1].set_title('difference between derived trace and {:s}'.format(flat))
                        x=np.arange(len(out[0]))
                        ax[1].set_xlim(0,len(x))
                        for a,b,c in zip(out,tab,flat_data.data) :
                            p=ax[1].plot(x,a-b,ls=':')
                            gd =np.where(c>0)[0]
                            ax[1].plot(x[gd],a[gd]-b[gd],color=p[-1].get_color())
                        ax[1].set_ylim(-5,5)
                        dt=(flat_data.header['TAI-BEG']-im.header['TAI-BEG'])/3600.
                        ax[1].text(0.5,0.9,'dt: {:.2f}'.format(dt),transform=ax[1].transAxes)
                    except : pass
                fig.tight_layout()
                fig.savefig(hard.replace('.png','_compare.png'))
                plt.close()
                col2.append(os.path.basename(hard).replace('.png','_compare.png'))

        if channel==channels[0] :
            grid=np.vstack([np.array(col1),np.array(col2)])
        else :
            grid=np.vstack([grid,np.array(col1),np.array(col2)])

    html.htmltab(np.array(grid).T,file='{:s}/{:d}/arcs_{:d}_{:s}.html'.format(outdir,mjd,mjd,obs),
                 ytitle=yt,xtitle=xt)


def transform_thread(pars) :
    try :
        return image.transform(pars[0],pars[1],pars[2],hard=pars[3])
    except :
        return None

