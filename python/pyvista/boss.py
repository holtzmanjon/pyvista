import numpy as np
import copy
import os
import pdb
import matplotlib
import matplotlib.pyplot as plt
from pydl.pydlutils import yanny
from pyvista import imred, spectra, sdss, gaia
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
    if done : return

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
        flat=red.reduce(name,channel=channel)
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
            raw=copy.deepcopy(out)
            if display is not None : plt.figure()
            for fiber in range(1,501) :
                skyclose=np.where(np.abs(plug['fiberId'][sky]-fiber) < 50) [0]
                skyfibers=plug['fiberId'][sky[skyclose]]
                wave_index = np.where(rows == fiber-1)[0]
                if display is not None :
                    print(fiber,wave_index,skyfibers)
                    plt.plot(out.data[fiber-1])
                if len(wave_index) > 0 :
                    wave_object = wavs[wave_index[0]].wave(image=out.data[fiber-1].shape)
                    sky_specs = []
                    for skyfiber in skyfibers :
                        sky_index = np.where(rows == skyfiber-1)[0]
                        if len(sky_index) > 0 :
                            sky_spec=wavs[sky_index[0]].scomb(raw[skyfiber-1],wave_object)
                            sky_specs.append(sky_spec.data)
                    out.data[fiber-1] -= np.median(np.array(sky_specs),axis=0)
                if display is not None :
                    plt.plot(np.median(np.array(sky_specs),axis=0))
                    plt.plot(out.data[fiber-1])
                    plt.draw()
                    pdb.set_trace()
                    plt.clf()

            out.write(dir+name.replace('sdR','sp1D'),overwrite=True)

    # populate wavelength image
    wave = np.full_like(out.data,np.nan)
    for j,row in enumerate(wave) :
        irow = np.where(rows == j)[0]
        if len(irow) > 0 :
            wave[j] = wavs[irow[0]].wave(image=row.shape)

    return out

def flux(planfile,maxobj=None,channel=0) :
    """ Do flux calibration from GAIA spectra
    """
    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'
    mjd = int(plan['MJD'])

    # get target information
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    if int(plan['MJD']) > 59600 :
        plug,header,sky,stan = sdss.getconfig(config_id=plan['SPEXP']['mapname'][objs[0]].astype(int),specid=1)
    else :
        plug,header,sky,stan = sdss.getconfig(plugid=plan['SPEXP']['mapname'][objs[0]].astype(str),specid=1)

    # get GAIA data
    print('getting gaia')
    gaiafile=dir+planfile.replace('spPlan2d','spGaia').replace('.par','.fits')
    if os.path.exists(gaiafile) :
        a=fits.open(gaiafile)
        g=a[1].data
        x=a[2].data
    else :
        g,x=gaia.get(plug['ra'],plug['dec'],vers='dr3_tap',posn_match=5,cols=[[plug['fiberId'],'fiberId']])
        hdu=fits.HDUList()
        hdu.append(fits.BinTableHDU(g))
        hdu.append(fits.BinTableHDU(x))
        hdu.writeto(gaiafile)
    i1,i2=match.match(g['source_id'],x['source_id'])
    j1,j2=match.match(plug['fiberId'],g['fiberId'][i1])
    w = np.linspace(3360,10200,343)

    # add the stars for which we have GAIA spectra
    print('getting waves')
    waves=getwaves(planfile)
    bflx=spectra.FluxCal(degree=-1)
    rflx=spectra.FluxCal(degree=-1)
    # set wavelength ranges for blue and red channels
    bwav=np.where((w>3400)&(w<7000))[0]
    rwav=np.where((w>5500)&(w<10200))[0]
    for obj in objs[0:maxobj] :
        # read the reduced sp1D images
        out=[]
        for channel in [0,1] :
            name=plan['SPEXP']['name'][obj][channel].astype(str)
            if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
                name=plan['SPEXP']['name'][obj][2].astype(str)
            out.append(CCDData.read(dir+name.replace('sdR','sp1D')))
        name=name.replace('sdR','spFlux').replace('.fit','')

        # loop over objects, loading spectra
        for ind,j in enumerate(j1) :
            if plug['delta_ra'][j] > 0 or plug['delta_dec'][j] > 0 : continue
            row = plug['fiberId'][j]-1
            flux = x['flux'][i2[j2[ind]]]
            bflx.addstar(out[0][row],waves[0][row],cal=[w[bwav],flux[bwav],20],extinct=False)
            rflx.addstar(out[1][row],waves[1][row],cal=[w[rwav],flux[rwav],20],extinct=False)

        # make the response curve
        bflx.response(legend=False,hard=dir+name.replace('-r1','-b1')+'.png')
        rflx.response(legend=False,hard=dir+name+'.png')

        # plot cross-sections
        fig,ax=plots.multi(1,2)
        for ww in [4000,5000,6000] :
          j=np.where(w[bwav]==ww)[0]
          rat=-2.5*np.log10(np.array(bflx.obscorr)[:,j]/np.array(bflx.true)[:,j])
          bins=np.arange(np.median(rat)-0.5,np.median(rat)+0.5,0.05)
          ax[0].hist(-2.5*np.log10(np.array(bflx.obscorr)[:,j]/np.array(bflx.true)[:,j]),
                   histtype='step', label='{:d}'.format(ww),bins=bins)
        ax[0].legend(fontsize='x-small')
        ax[0].set_xlabel('zeropoint (mag)')
        for ww in [6500,7500,8500] :
          j=np.where(w[rwav]==ww)[0]
          rat=-2.5*np.log10(np.array(rflx.obscorr)[:,j]/np.array(rflx.true)[:,j])
          bins=np.arange(np.median(rat)-0.5,np.median(rat)+0.5,0.05)
          ax[1].hist(-2.5*np.log10(np.array(rflx.obscorr)[:,j]/np.array(rflx.true)[:,j]),
                   histtype='step', label='{:d}'.format(ww),bins=bins)
        ax[1].legend(fontsize='x-small')
        ax[0].set_xlabel('zeropoint (mag)')
        fig.tight_layout()
        fig.savefig(dir+name.replace('-r1','').replace('Flux','FluxHist')+'.png')
        pdb.set_trace()
    return bflx,rflx

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
        out=CCDData.read(dir+name.replace('sdR','sp1D'))

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
    plan=yanny.yanny(planfile)
    dir=os.path.dirname(planfile)+'/'
    mjd = int(plan['MJD'])

    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    if int(plan['MJD']) > 59600 :
        plug,header,sky,stan = sdss.getconfig(config_id=plan['SPEXP']['mapname'][objs[0]].astype(int),specid=1)
    else :
        plug,header,sky,stan = sdss.getconfig(plugid=plan['SPEXP']['mapname'][objs[0]].astype(str),specid=1)

    iarc=np.where(plan['SPEXP']['flavor'] == b'arc')[0]
    waves=[]
    for channel in [0,1] :
        name=plan['SPEXP']['name'][objs[0]][channel].astype(str)
        out=CCDData.read(dir+name.replace('sdR','sp1D'))

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

    matplotlib.use('Agg')
    for obj in objs[0:maxobj] :
        fhtml=open(dir+name.replace('sdR','spPlate').replace('-r1','').replace('.fit','.html'),'w')
        fhtml.write('<HTML><BODY><TABLE BORDER=2>\n')
        fhtml.write('<TR><TD>Fiber<TD>CatalogID<TD>Category<TD>g<br>r<br>i<TD>Extracted spectra\n')

        # read the reduced sp1D images
        out=[]
        for channel in [0,1] :
            name=plan['SPEXP']['name'][obj][channel].astype(str)
            if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
                name=plan['SPEXP']['name'][obj][2].astype(str)
            out.append(CCDData.read(dir+name.replace('sdR','sp1D')))

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
                plots.plotl(ax,waves[channel][fiber-1],out[channel].data[fiber-1],color=colors[channel])
            png = name.replace('sdR','spPlate').replace('-r1','').replace('.fit','-{:03d}.png'.format(fiber))
            fig.savefig(dir+png)
            plt.close()
            fhtml.write('<TD><A HREF={:s}><IMG SRC={:s}></A>\n'.format(png,png))

        fhtml.close()
