import numpy as np
import copy
import os
import pdb
import matplotlib
import matplotlib.pyplot as plt
from pydl.pydlutils import yanny
from pyvista import imred, spectra, sdss, gaia
from pyvista.dataclass import Data
from tools import match,plots
from ccdproc import CCDData
from scipy.ndimage import median_filter
import multiprocessing as mp
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
from astropy.io.votable import parse_single_table
import scipy.interpolate

def visit(planfile,clobber=False,maxobj=None,threads=12) :
    """ Reduce BOSS visit

        Driver for parallell processing of b and r channels
        Makes plots of median counts vs mag
    """
    if not os.path.exists(planfile) :
        raise ValueError('no such file: {:s}'.format(planfile))
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

    print('back')

def visit_channel(planfile=None,channel=0,clobber=False,threads=12,display=None,maxobj=None,skysub=True,flux=True) :
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
    if done : return

    # set up Reducer
    red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_N']+'/'+plan['MJD'])
   
    # get Trace/PSF/flat1D
    print('trace')
    trace,flat1d=mktrace(planfile,channel=channel,clobber=clobber,display=display,threads=threads)
    # 1d flat to 2D
    fim=np.tile(flat1d,(4224,1)).T

    # get wavelength calibration
    print('wave')
    wave= mkwave(planfile,channel=channel,clobber=clobber,threads=threads,display=display)

    # reduce and extract science frames
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    for obj in objs[0:maxobj] :
        name=plan['SPEXP']['name'][obj][channel].astype(str)
        if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
            name=plan['SPEXP']['name'][obj][2].astype(str)
        if os.path.exists(dir+name.replace('sdR','sp1D')) and not clobber : 
            out=Data.read(dir+name.replace('sdR','sp1D'))
        else :
            im=red.reduce(name,channel=channel)
            out=trace.extract(im,threads=threads,nout=500,plot=display)
            # 1d fiber-to-fiber flat
            out.data /= fim
            out.uncertainty.array /= fim

            #add wavelength
            out.wave = wave

            # need plugfile info for sky subtract and flux calibration
            if int(plan['MJD']) > 59600 :
                plug,header,sky,stan = sdss.getconfig(config_id=out.header['CONFID'],specid=1)
            else :
                plug,header,sky,stan = sdss.getconfig(plugid=plan['SPEXP']['mapname'][objs[0]].astype(str),specid=1)

            if skysub: skysubtract(out,plug['fiberId'][sky])
            if flux : mkflux(out,plug,planfile,channel=channel)

            pdb.set_trace()
            out.write(dir+name.replace('sdR','sp1D'),overwrite=True)

    return out

def skysubtract(out,skyfibers,display=None) :

    # sky subtraction
    raw=copy.deepcopy(out)
    if display is not None : plt.figure()
    for fiber in range(1,501) :
        skyclose=np.where(np.abs(skyfibers-fiber) < 50) [0]
        if display is not None :
            print(fiber,skyfibers[skyclose])
            plt.plot(out.data[fiber-1])
        sky_specs = []
        for skyfiber in skyfibers[skyclose] :
            # get sky spectrum sampled at wavelengths of this fiber by interpolation
            sky_spec = np.interp(out.wave[fiber-1],out.wave[skyfiber-1],out.data[skyfiber-1])
            sky_specs.append(sky_spec)
        out.data[fiber-1] -= np.median(np.array(sky_specs),axis=0)
        if display is not None :
            plt.plot(np.median(np.array(sky_specs),axis=0))
            plt.plot(out.data[fiber-1])
            plt.draw()
            pdb.set_trace()
            plt.clf()


def mktrace(planfile,channel=0,clobber=False,nfibers=500,threads=0,display=None,skip=40) :
    """ Create spTrace and spFlat1d files or read if they already exist
    """

    plan=yanny.yanny(planfile)
    outdir=os.path.dirname(planfile)+'/'

    red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_N']+'/'+plan['MJD'])

    iflat=np.where(plan['SPEXP']['flavor'] == b'flat')[0]
    name=plan['SPEXP']['name'][iflat][0][channel].astype(str)
    if len(plan['SPEXP']['name'][iflat][0]) > 2  and channel == 1 :
        name=plan['SPEXP']['name'][iflat][0][2].astype(str)
    outname=outdir+name.replace('sdR','spTrace')

    if os.path.exists(outname) and not clobber :
        trace=spectra.Trace(outname)
        f=np.loadtxt(outname.replace('spTrace','spFlat1d').replace('.fit','.txt'))
    else :
        print('creating Trace')
        flat=red.reduce(name,channel=channel)
        trace=spectra.Trace(transpose=red.transpose,rad=3,lags=np.arange(-3,4))
        ff=np.sum(flat.data[2000:2100],axis=0)
        if channel==0 : thresh=0.4e6
        else : thresh=0.2e6
        peaks,fiber=spectra.findpeak(ff,thresh=thresh,diff=10,bundle=20)
        print('found {:d} peaks'.format(len(peaks)))
        trace.trace(flat,peaks[0:nfibers],index=fiber[0:nfibers],skip=skip,
                    display=display)
        trace.write(outname) 
        flat1d=trace.extract(flat,threads=threads,nout=500,display=display)
        f=np.median(flat1d.data[:,1500:2500],axis=1)
        f/=np.median(f)
        np.savetxt(outname.replace('spTrace','spFlat1d').replace('.fit','.txt'),f)

    return trace,f


def mkwave(planfile,channel=0,threads=0,clobber=False,display=None) :
    """ Create spWave files or read if they already exist

        Return 2D wavelength image
    """

    plan=yanny.yanny(planfile)
    outdir=os.path.dirname(planfile)+'/'
    red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_N']+'/'+plan['MJD'])

    # get trace
    trace,flat1d=mktrace(planfile,channel=channel,clobber=clobber,display=display,threads=threads)

    chan = ['b1','r1']
    iarc=np.where(plan['SPEXP']['flavor'] == b'arc')[0]
    name=plan['SPEXP']['name'][iarc][0][channel].astype(str)
    if len(plan['SPEXP']['name'][iarc][0]) > 2  and channel == 1 :
        name=plan['SPEXP']['name'][iarc][0][2].astype(str)
    outname=outdir+name.replace('sdR','spWave')

    if os.path.exists(outname) and not clobber : 
        wavs=[]
        rows=[]
        wfits=fits.open(outname) 
        for w in wfits[1:] :
            wav= spectra.WaveCal(w.data)
            wavs.append(wav)
            rows.append(wav.index)
        im=red.reduce(name,channel=channel)
    else :
        print('creating WaveCal')
        im=red.reduce(name,channel=channel)
        arcec=trace.extract(im,threads=threads,nout=500,display=display)
        wav=spectra.WaveCal('BOSS/BOSS_{:s}_waves.fits'.format(chan[channel]))
        wavs=[]
        rows=[]
        for irow in range(250,500) :
            if irow in trace.index :
                wav.identify(arcec[irow],plot=None,thresh=20,rad=5)
                wavs.append(copy.deepcopy(wav))
                rows.append(irow)
        wav=spectra.WaveCal('BOSS/BOSS_{:s}_waves.fits'.format(chan[channel]))
        for irow in range(249,-1,-1) :
            if irow in trace.index :
                wav.identify(arcec[irow],plot=None,thresh=20,rad=5,maxshift=2)
                wavs.append(copy.deepcopy(wav))
                rows.append(irow)
        wavs[0].index = rows[0]
        wavs[0].write(outname)
        for wav,row in zip(wavs[1:],rows[1:]) :
            wav.index = row
            wav.write(outname,append=True)
    rows = np.array(rows)
    # populate wavelength image, remember that BOSS is transposed, so axis 0 in 2D is wavelength
    wave = np.empty((500,im.data.shape[0]))
    wave[:,:] = np.nan
    for j,row in enumerate(wave) :
        irow = np.where(rows == j)[0]
        if len(irow) > 0 :
            wave[j] = wavs[irow[0]].wave(image=row.shape)

    return wave

#def mkflux(planfile,maxobj=None,channel=0,medfilt=15) :
def mkflux(out,plug,planfile,medfilt=15,plot=True,channel=0) :

    """ Do flux calibration from GAIA spectra
    """
    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'

#    mjd = int(plan['MJD'])
#
#    # get target information
#    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
#    if int(plan['MJD']) > 59600 :
#        plug,header,sky,stan = sdss.getconfig(config_id=plan['SPEXP']['mapname'][objs[0]].astype(int),specid=1)
#    else :
#        plug,header,sky,stan = sdss.getconfig(plugid=plan['SPEXP']['mapname'][objs[0]].astype(str),specid=1)

    # get GAIA data
    print('getting gaia')
    gaia_posn=dir+planfile.replace('spPlan2d','spGaiaPosn').replace('.par','.xml')
    gaia_flux=dir+planfile.replace('spPlan2d','spGaiaFlux').replace('.par','.xml')
    if os.path.exists(gaia_posn) :
        g=parse_single_table(gaia_posn).array
        x=parse_single_table(gaia_flux).array
    else :
        g,x=gaia.get(plug['ra'],plug['dec'],vers='dr3_tap',posn_match=5,cols=[[plug['fiberId'],'fiberId']])
        g._votable.to_xml(gaia_posn)
        x._votable.to_xml(gaia_flux)
    i1,i2=match.match(g['source_id'],x['source_id'])
    j1,j2=match.match(plug['fiberId'],g['fiberId'][i1])
    w = np.linspace(3360,10200,343)

    # set wavelength ranges for blue and red channels
    if channel == 0 :
        wav=np.where((w>3400)&(w<7000))[0]
        ww = [4000,5000,6000] 
    else :
        wav=np.where((w>5500)&(w<10200))[0]
        ww = [6500,7500,8500]

    # add the stars for which we have GAIA spectra
    flx=spectra.FluxCal(degree=-1)
    for ind,j in enumerate(j1) :
        if plug['delta_ra'][j] > 0 or plug['delta_dec'][j] > 0 : continue
        row = plug['fiberId'][j]-1
        flux = x['flux'][i2[j2[ind]]]
        flx.addstar(out[row],out.wave[row],cal=[w[wav],flux[wav],20],extinct=False)

    # make the response curve
    flx.response(legend=False,medfilt=medfilt)

    if plot :
        # plot cross-sections
        fig,ax=plots.multi(1,1)
        for ww in [4000,5000,6000] :
            j=np.where(w[wav]==ww)[0]
            rat=-2.5*np.log10(np.array(flx.obscorr)[:,j]/np.array(flx.true)[:,j])
            bins=np.arange(np.median(rat)-0.5,np.median(rat)+0.5,0.05)
            ax.hist(rat, histtype='step', label='{:d}'.format(ww),bins=bins)
        ax.legend(fontsize='x-small')
        ax.set_xlabel('zeropoint (mag)')
        fig.tight_layout()

    # apply flux curves
    flx.correct(out,out.wave)

    return

def combine(out) :
    """ Combine two channels and resample to common wavelength scale
    """
    wnew = 10.**(np.arange(3.5589,4.0151,1.e-4))
    comb = np.zeros([out[0].shape[0],len(wnew)])
    comberr = np.zeros([out[0].shape[0],len(wnew)])
    for irow in range(len(out[0].data)) :
        gd = np.where((out[0].wave[irow] > wnew[0]) & (out[0].wave[irow]<6200.) )[0]
        if len(gd) == 0 : continue
        try :
          dspline = scipy.interpolate.CubicSpline(out[0].wave[irow,gd],out[0].data[irow,gd])
          vspline = scipy.interpolate.CubicSpline(out[0].wave[irow,gd],out[0].uncertainty.array[irow,gd]**2)
        except : continue
        bdata = dspline(wnew)
        bvar = vspline(wnew)
        bdata[np.where(wnew>6200)[0]] = 0.
        bvar[np.where(wnew>6200)[0]] = 1.e10

        gd = np.where((out[0].wave[irow] < wnew[-1]) & (out[1].wave[irow] > 6100.) )[0]
        try :
          dspline = scipy.interpolate.CubicSpline(out[1].wave[irow,gd],out[1].data[irow,gd])
          vspline = scipy.interpolate.CubicSpline(out[1].wave[irow,gd],out[1].uncertainty.array[irow,gd]**2)
        except : continue
        rdata = dspline(wnew)
        rvar = vspline(wnew)
        rdata[np.where(wnew<6100)[0]] = 0.
        rvar[np.where(wnew<6100)[0]] = 1.e10

        comb[irow] = (bdata/bvar + rdata/rvar) / (1/bvar + 1/rvar)
        comberr[irow] = np.sqrt(1. / (1/bvar + 1/rvar))

    return Data(comb,uncertainty=comberr,wave=wnew)


def getwaves(planfile) :
    """ Load up 2D wavelength array from spWave file
    """

    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'

    iarc=np.where(plan['SPEXP']['flavor'] == b'arc')[0]
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    waves=[]
    for channel in [0,1] :
        name=plan['SPEXP']['name'][objs[0]][channel].astype(str)
        out=Data.read(dir+name.replace('sdR','sp1D'))

        name=plan['SPEXP']['name'][iarc][0][channel].astype(str)
        if len(plan['SPEXP']['name'][iarc][0]) > 2  and channel == 1 :
            name=plan['SPEXP']['name'][iarc][0][2].astype(str)
        wfits=fits.open(dir+name.replace('sdR','spWave'))

        # populate wavelength image
        wave = np.full_like(out.data,np.nan)
        for w in wfits[1:] :
            wav=spectra.WaveCal(w.data)
            wave[wav.index] = wav.wave(image=out.data.shape[1])
        waves.append(wave)

    return waves

def html(planfile,maxobj=None,channel=0) :
    """ Create HTML file for visit
    """
    if not os.path.exists(planfile) :
        raise ValueError('no such file: {:s}'.format(planfile))
    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'
    mjd = int(plan['MJD'])

    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]

    if int(plan['MJD']) > 59600 :
        plug,header,sky,stan = sdss.getconfig(config_id=plan['SPEXP']['mapname'][objs[0]].astype(int),specid=1)
    else :
        plug,header,sky,stan = sdss.getconfig(plugid=plan['SPEXP']['mapname'][objs[0]].astype(str),specid=1)

    matplotlib.use('Agg')
    for obj in objs[0:maxobj] :
        name=plan['SPEXP']['name'][obj][1].astype(str)
        fhtml=open(dir+name.replace('sdR','spPlate').replace('-r1','').replace('.fit','.html'),'w')
        fhtml.write('<HTML><BODY><TABLE BORDER=2>\n')
        fhtml.write('<TR><TD>Fiber<TD>CatalogID<TD>Category<TD>g<br>r<br>i<TD>Extracted spectra\n')

        # read the reduced sp1D images
        out=[]
        for channel in [0,1] :
            name=plan['SPEXP']['name'][obj][channel].astype(str)
            if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
                name=plan['SPEXP']['name'][obj][2].astype(str)
            out.append(Data.read(dir+name.replace('sdR','sp1D')))

        colors=['b','r']
        for fiber in range(1,501) :
            j = np.where(plug['fiberId'] == fiber)[0][0]
            print(plug['fiberId'][j],plug['category'][j])
            fhtml.write('<TR>\n')
            fhtml.write('<TD>{:d}\n'.format(fiber))
            fhtml.write('<TD>{:d}\n'.format(plug['catalogid'][j]))
            fhtml.write('<TD>{:s}\n'.format(plug['category'][j].decode()))
            fhtml.write('<TD>{:7.2f}<br>{:7.2f}<br>{:7.2f}\n'.format(*plug['mag'][j,1:4]))
            fig,ax=plots.multi(1,1,figsize=(8,2))
            for channel in [0,1] :
                gd=np.where(np.isfinite(out[channel].data[fiber-1]))[0]
                if len(gd) > 100 :
                    plots.plotl(ax,out[channel].wave[fiber-1],out[channel].data[fiber-1],color=colors[channel])
                    ymax = median_filter(out[channel].data[fiber-1],100).max()
                    ax.set_ylim(0,1.5*ymax)
            png = name.replace('sdR','spPlate').replace('-r1','').replace('.fit','-{:03d}.png'.format(fiber))
            try: os.mkdir(dir+'/plots')
            except: pass
            fig.savefig(dir+'/plots/'+png)
            plt.close()
            fhtml.write('<TD><A HREF=plots/{:s}><IMG SRC=plots/{:s}></A>\n'.format(png,png))

        fhtml.close()
