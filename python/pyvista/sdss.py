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

def config(cid,specid=2,struct='FIBERMAP',useparent=True,useconfF=False) :
    """ Get FIBERMAP structure from configuration file for specified instrument
           including getting parent_configuration if needed (for scrambled configurations)
    """
    if useconfF : confname='confSummaryF'
    else :confname='confSummary'
    if isinstance(cid,str):
        conf = yanny(cid)
    else :
        conf = yanny(os.environ['SDSSCORE_DIR']+'/apo/summary_files/{:04d}XX/{:s}-{:d}.par'.format(cid//100,confname,cid))
        if useparent :
            try :
                parent = int(conf['parent_configuration'])
                if parent > 0 :
                    conf = yanny(os.environ['SDSSCORE_DIR']+'/apo/summary_files/{:04d}XX/{:s}-{:d}.par'.format(parent//100,confname,parent))
            except :  pass

    if conf == None or len(conf) == 0 :
        raise FileNotFoundError('error opening file',cid)

    gd =np.where((conf[struct]['spectrographId'] == specid) & (conf[struct]['fiberId'] > 0) )[0]
    return conf[struct][gd],conf.new_dict_from_pairs()

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
        
def db_exp(exp_no,cam,header) :

    tab_exp=Table()
    tab_exp['exp_no' ] = [exp_no]
    tab_exp['camera' ] = [cam]
    tab_exp['exptime' ] = [header['EXPTIME']]
    tab_exp['dateobs' ] = [header['DATE-OBS']]
    tab_exp['ra' ] = [header['RA']]
    tab_exp['dec' ] = [header['DEC']]
    tab_exp['cherno_offset_ra' ] = [0.]
    tab_exp['cherno_offset_dec' ] = [0.]
    tab_exp['pa' ] = [header['ROTPOS']]
    tab_exp['ipa' ] = [header['IPA']]
    tab_exp['secz' ] = [1/np.cos((90-header['ALT'])*np.pi/180.)]
    tab_exp['config_id' ] = [config_id]
    try: tab_exp['design_id' ] = [header['DESIGNID']]
    except :tab_exp['design_id' ] = [0]
    try: seeing = header['SEEING']
    except : seeing = 0.
    tab_exp['seeing' ] = [seeing]
    tab_exp['fwhm'] = [np.array([seeing,seeing,seeing])]
    tab_exp['gdrms'] = [0.] 
    tab_exp['guider_zero' ] = [0.]
    tab_exp['dithered' ] = [0]
    tab_exp['flag' ] = [0]
    gd = np.where(hmag[i2] > 0)[0]
    tab_exp['zeronorm' ] = [[np.nan.np.nan,np.nan]]

    return tab_exp

def db_spec(plug, header, confSummary = True) :

    tab_spec=Table()
    tab_spec['fiber'] = plug['fiberId']
    if confSummary :
        tab_spec['catalogid' ] = plug['catalogid'][i2]
        tab_spec['assigned'] = plug['assigned'][i2]
        tab_spec['on_target'] = plug['on_target'][i2]
        tab_spec['valid'] = plug['valid'][i2]
        tab_spec['cadence'] = plug['cadence'][i2].astype(str)
        tab_spec['program'] = plug['program'][i2].astype(str)
        tab_spec['category'] = plug['category'][i2].astype(str)
        tab_spec['racat'] = plug['racat'][i2]
        tab_spec['deccat'] = plug['deccat'][i2]
        tab_spec['offset_ra'] = plug['delta_ra'][i2]
        tab_spec['offset_dec'] = plug['delta_dec'][i2]
        tab_spec['delta_ra'] = plug['delta_ra'][i2]
        tab_spec['delta_dec'] = plug['delta_dec'][i2]
        dt = (float(header['epoch'])-2457204.)/365.26
        j = np.where((plug['pmdec']<-998) & (plug['pmdec']>-1000) )[0]
        plug['pmdec'][j] = 0.
        plug['pmra'][j] = 0.
        tab_spec['delta_ra'] = ((plug['ra']-plug['racat'])*np.cos(float(header['decCen'])*np.pi/180.)*3600.
                                -plug['pmra']/1000.*dt)[i2]
        tab_spec['delta_dec'] = ((plug['dec']-plug['deccat'])*3600.
                                 -plug['pmdec']/1000.*dt)[i2]

        tab_spec['xfocal'] = plug['xfocal'][i2]
        tab_spec['yfocal'] = plug['yfocal'][i2]
        if 'xFVC' in plug.keys() :
            tab_spec['xFVC'] = plug['xFVC'][i2]
            tab_spec['yFVC'] = plug['yFVC'][i2]
        else :
            tab_spec['xFVC'] = 0.
            tab_spec['yFVC'] = 0.
        tab_spec['alpha'] = plug['alpha'][i2]
        tab_spec['beta'] = plug['beta'][i2]
        if 'mag_g' in plug.keys() :
            tab_spec['mag_g'] = plug['mag_g'][i2]
            tab_spec['mag_r'] = plug['mag_r'][i2]
            tab_spec['mag_i'] = plug['mag_i'][i2]
        else :
            tab_spec['mag_g'] = -999.
            tab_spec['mag_r'] = -999.
            tab_spec['mag_i'] = -999.
        tab_spec['bp_mag'] = plug['bpmag'][i2]
        tab_spec['rp_mag'] = plug['rpmag'][i2]
        tab_spec['mag'] = plug['mag'][i2]
    else :
        for key in ['catalogid','assigned','on_target','valid'] : tab_spec[key]=0
        for key in ['cadence','program','category'] : tab_spec[key]=''
        for key in ['racat','deccat','offset_ra','offset_dec', 'xFVC','yFVC',
                    'alpha','beta','mag_g','mag_r','mag_i','bp_mag','rp_mag','hmag' ] :
                tab_spec[key] = 0.
        tab_spec['category'] = plug['objType'][i2].astype(str)
        tab_spec['hmag'] = hmag[i2]
        tab_spec['ra'] = plug['ra'][i2]
        tab_spec['dec'] = plug['dec'][i2]
        tab_spec['xfocal'] = plug['xFocal'][i2]
        tab_spec['yfocal'] = plug['yFocal'][i2]
        tab_spec['mag'] = plug['mag'][i2]

def db_visit(mjd, field) :

    tab_visit=Table()
    tab_visit['mjd'] = [mjd]
    tab_visit['field_id'] =  [field]

    return tab_visit

