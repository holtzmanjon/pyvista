from astropy.io import fits
import astropy.units as u
import glob
import numpy as np
import pdb
import os
from pydl.pydlutils.yanny import yanny
from tools import plots, match
import matplotlib
import matplotlib.pyplot as plt

def qlhtml(mjd5,clobber=False) :
    """ Make HTML file for a night. Constructs individual exposure plots only as needed,
          so can be run as each exposure comes in
    """

    files = glob.glob('/data/apogee/quickred/{:d}/apq*.fits'.format(mjd5))
    fp =open('/data/apogee/quickred/{:d}/{:d}.html'.format(mjd5,mjd5),'w')
    fp.write('<HTML><BODY>\n')
    fp.write('<H2>Quickred plots for '+str(mjd5)+'</H2>\n')
    fp.write('<TABLE BORDER=2>\n')

    for file in np.sort(files) :
        im=int(file.split('.')[0].split('-')[1])
        print(im)
        q=fits.open('/data/apogee/quickred/{:d}/apq-{:d}.fits'.format(mjd5,im))
        if q[0].header['EXPTYPE'] == 'OBJECT' :
            if clobber or not os.path.exists('/data/apogee/quickred/{:d}/apq-{:d}_flux.png'.format(mjd5,im)) :
                flux(im,hard=True)
            ra=q[0].header['BOREOFFX']
            dec=q[0].header['BOREOFFY']
            rot=q[0].header['ROTPOS']
            fp.write('<TR><TD>{:d}<TD>{:f}<br>{:f}<br>{:f}<TD><A HREF=apq-{:d}_flux.png> <IMG SRC=apq-{:d}_flux.png WIDTH=800></A></TD>\n'.format(
                                  im,ra,dec,rot,im,im))
            fp.write('<TD><A HREF=apRaw-{:d}_thumb.png> <IMG SRC=apRaw-{:d}_thumb.png WIDTH=500></A></TD>\n'.format(im,im))

    fp.write('</TABLE></BODY></HTML>\n')
    fp.close()

def config(cid,specid=2,struct='FIBERMAP') :
    """ Get FIBERMAP structure from configuration file for specified instrument
           including getting parent_configuration if needed (for scrambled configurations)
    """
    if isinstance(cid,str):
        conf = yanny(cid)
    else :
        conf = yanny(os.environ['SDSSCORE_DIR']+'/apo/summary_files/{:04d}XX/confSummary-{:d}.par'.format(cid//100,cid))
        try :
            parent = int(conf['parent_configuration'])
            if parent > 0 :
                conf = yanny(os.environ['SDSSCORE_DIR']+'/apo/summary_files/{:04d}XX/confSummary-{:d}.par'.format(parent//100,parent))
        except :  pass

    gd =np.where((conf[struct]['spectrographId'] == specid) & (conf[struct]['fiberId'] > 0) )[0]
    return conf[struct][gd]

def flux(im,inst='APOGEE',thresh=100,cid=None,hard=False) :
    """ Plots of median flux vs various
    """
    if hard : 
        backend = matplotlib.get_backend()
        plt.switch_backend('Agg')
    if inst == 'APOGEE' :
        if isinstance(im,int) :
            # use APOGEE quickred apq files
            fil = str(im)
            mjd5 = im//10000+55562
            apq=fits.open('/data/apogee/quickred/{:d}/apq-{:08d}.fits'.format(mjd5,im)) 
            des=int(apq[0].header['DESIGNID'])
            if cid is None : cid=int(apq[0].header['CONFIGID'])
            fiberid=apq[2].data['fiberid']
            flux = apq[2].data['flux']
            nframes = apq[0].header['NFRAMES']
        else :
            fil=os.path.basename(im.header['FILENAME'])
            des=im.header['DESIGNID']
            nframes = im.header['NFRAMES']
            if cid is None : cid=im.header['CONFIGID']
            fiberid=300-np.arange(300)
            flux = np.median(im.data,axis=1)
        # get the configuration, allowing for a parent configuration
        c = config(cid,specid=2)
        # match the fiberids
        i1, i2 = match.match(fiberid,c['fiberId'])
        mag=c['h_mag']
        cmag='H'
        zr=[12,20]
    else :
        fil=im.header['FILENAME']
        print(im.header['CONFID'])
        cid=im.header['CONFID']
        des=im.header['DESIGNID']
        c = config(cid,specid=1)
        i1, i2 = match.match(1+np.arange(500),c['fiberId'])
        if fil.find('-b1-') > 0 :
            mag=c['mag'][:,1]
            cmag='g'
        else :
            mag=c['mag'][:,2]
            cmag='r'
        flux = np.median(im.data,axis=1)
        zr=[15,25]

    print('found match for {:d} fibers'.format(len(i1)))
    # get assigned, sky, and science indices
    assigned=np.where(c['assigned'][i2] == 1)[0]
    sky=np.where(np.char.find(c['category'][i2],b'sky') >=0)[0]
    sci=np.where(np.char.find(c['category'][i2[assigned]],b'sky') <0)[0]
    sci=assigned[sci]

    # rough sky subtraction
    if len(sky) > 0 : skyval=np.median(flux[i1[sky]])
    else : skyval = np.median(flux)
    flux -= skyval

    # setup plots
    fig,ax=plots.multi(4,1,figsize=(15,3))
    fig.suptitle('File: {:s}  Design: {:d}   Config: {:d}'.format(fil,des,cid))
    # flux vs fiber
    plots.plotp(ax[0],c['fiberId'][i2],flux[i1],xt='fiberId',yt='flux',color='r')
    plots.plotp(ax[0],c['fiberId'][i2[sci]],flux[i1[sci]],color='g',size=20,label='science')
    plots.plotp(ax[0],c['fiberId'][i2[sky]],flux[i1[sky]],color='b',size=20,label='sky')
    ax[0].legend()

    # -2.5log(flux) vs mag
    plots.plotp(ax[1],mag[i2[sci]],-2.5*np.log10(flux[i1[sci]]),xt=cmag+' mag',yt='-2.5*log10(flux)')
    if inst == 'APOGEE' :
        zero=19+2.5*np.log10((nframes-2)/(47-2))
        ax[1].plot([8,14],[8-zero,14-zero])
    ylim=ax[1].get_ylim()
    ax[1].set_ylim([ylim[1],ylim[0]])

    # cumulative histogram of flux
    bins=np.arange(-100,1000,10)
    bins[-1] = 1000000.
    ax[2].hist(flux[i1[sci]],cumulative=-1,histtype='step',bins=bins,color='g',label='science')
    ax[2].hist(flux[i1[sky]],cumulative=-1,histtype='step',bins=bins,color='b',label='sky')
    ax[2].set_xlim(-100,1000)
    ax[2].legend()
    ax[2].set_xlabel('flux')
    ax[2].set_ylabel('N(>flux)')
    gd=np.where(flux[i1] > thresh)[0]
    ax[2].text(0.95,0.5,'Number > {:d} : {:d}'.format(thresh,len(gd)),transform=ax[2].transAxes,ha='right')

    # map of "zeropoints"
    plots.plotc(ax[3],c['xFocal'][i2[sci]],c['yFocal'][i2[sci]],mag[i2[sci]]+2.5*np.log10(flux[i1[sci]]),
                xt='xFocal',yt='yFocal',size=20,colorbar=True,zt='mag-inst',zr=zr)
    #plots.plotc(ax[1,0],c['xFocal'][i2],c['yFocal'][i2],flux[i1],
    #            xt='xFocal',yt='yFocal',size=20,colorbar=True,zt='flux')
    fig.tight_layout()
    if hard :
        fig.savefig('/data/apogee/quickred/{:d}/apq-{:08d}_flux.png'.format(mjd5,im)) 
        plt.switch_backend(backend)
    else :
        pdb.set_trace()
    plt.close()
    return flux

def map(ims,keys,cid=None,thresh=100) :
    """ Plots for multiple input images
    """
    if cid is None : cid=ims[keys[0]].header['CONFIGID']
    print(cid)
    c = config(cid,specid=2)
    i1, i2 = match.match(300-np.arange(300),c['fiberId'])
    assigned=np.where(c['assigned'][i2] == 1)[0]
    sky=np.where(np.char.find(c['category'][i2],b'sky') >=0)[0]
    sci=np.where(np.char.find(c['category'][i2[assigned]],b'sky') <0)[0]
    sci=assigned[sci]
    fig,ax=plots.multi(1,2,figsize=(4,10),hspace=0.2)
    for i,key in enumerate(keys) :
        flux=np.median(ims[key].data,axis=1)
        if len(sky) > 0 : skyval=np.median(flux[i1[sky]])
        else : skyval = np.median(flux)
        flux -= skyval
        ax[0].plot(c['fiberId'][i2],flux[i1],label=key)
        gd=np.where(flux>thresh)[0]
        print(key,len(gd))
        ax[1].scatter(c['xFocal'][i2[gd]]+i*5,c['yFocal'][i2[gd]],s=20)
    ax[0].legend()
    pdb.set_trace()
    plt.close()
        

