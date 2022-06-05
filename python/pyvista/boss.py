import numpy as np
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

def visit(planfile,tracefile=None,clobber=False,db=None, schema='obs2',maxobj=None) :
    """ Reduce BOSS visit

        Driver for parallell processing of b and r channels
        Makes plots of median counts vs mag
    """
    plan=yanny.yanny(planfile)
    mjd = int(plan['MJD'])

    # reduce b1 and r1 in parallel
    procs=[]
    for channel in [0,1] :
        kw={'planfile' : planfile, 'channel' : channel, 'clobber' : clobber, 'maxobj' : maxobj}
        procs.append(mp.Process(target=do_visit,kwargs=kw))
    for proc in procs : proc.start()
    for proc in procs : proc.join()

    try: plates=fits.open('/home/sas/dr17/plates-dr17.fits')[1].data
    except: plates=None
    try: gcam=fits.open('{:s}/{:d}/gcam-{:d}.fits'.format(os.environ['GCAM_DATA_N'],mjd,mjd))[1].data
    except: gcam=None


    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    fig,ax=plots.multi(1,2)
    allmags=[]
    allinst=[]
    chan = ['b1','r1']
    for channel in [0,1] :
      mags=[]
      inst=[]
      for obj in objs[0:maxobj] :
        name=plan['SPEXP']['name'][obj][channel].astype(str)
        if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
            name=plan['SPEXP']['name'][obj][2].astype(str)
        exp_no = int(name.replace('.fit','').split('-')[2])
        print(name)
        out=CCDData.read(name.replace('sdR','sp1D'))
        if channel == 0 : 
            mag='g'
            imag=1
            p1=1220+48
            p2=2587+48
        else : 
            mag='i'
            imag=3
            p1=1297+48
            p2=2399+48

        if int(plan['MJD']) > 59600 :
            conf=True
            config_id = out.header['CONFID']
            plug,header=sdss.config(out.header['CONFID'],specid=1,useconfF=True)
            isky=np.where(plug['category'] == b'sky_boss')[0]
            x = plug['bp_mag']-plug['rp_mag']
            x2 = x * x
            x3 = x * x * x
            gaia_G = plug['gaia_g_mag']
            gaia_sdss_g = -1 * (0.13518 - 0.46245 * x - 0.25171 * x2 + 0.021349 * x3) + gaia_G
            #gaia_sdss_r = -1 * (-0.12879 + 0.24662 * x - 0.027464 * x2 - 0.049465 * x3) + gaia_G
            gaia_sdss_i = -1 * (-0.29676 + 0.64728 * x - 0.10141 * x2) + gaia_G
            smag = plug['mag'][:,imag]
            j = np.where(gaia_G < 15)[0]
            if mag == 'g' :
                smag[j] = gaia_sdss_g[j]
            elif mag == 'i' :
                smag[j] = gaia_sdss_i[j]
        else :
            conf=False
            config_id = out.header['PLATEID']
            mapname=plan['SPEXP']['mapname'][obj].astype(str)
            plug,header=sdss.config(os.environ['MAPPER_DATA_N']+'/'+mapname.split('-')[1]+'/plPlugMapM-'+mapname+'.par',specid=1,struct='PLUGMAPOBJ')
            isky=np.where(plug['objType'] == b'SKY')[0]
            smag = plug['mag'][:,imag]

        i1,i2=match.match(np.arange(500)+1,plug['fiberId'])
        skyfiber=plug['fiberId'][isky]
        sky=np.nanmedian(out.data[skyfiber-1,:],axis=0)
        np.savetxt(name.replace('.fit','.txt'),sky)
        try : stan=np.where(np.char.find(plug['category'][i2].astype(str),'standard') >= 0)[0]
        except : stan=np.where(np.char.find(plug['objType'][i2].astype(str),'STD') >= 0)[0]
        rad=np.sqrt(plug['xFocal'][i2]**2+plug['yFocal'][i2]**2)
        plots.plotp(ax[channel],smag[i2[stan]],
              2.5*np.log10(np.median((out.data[:,p1:p2]-sky[p1:p2])/out.header['EXPTIME']*900,axis=1))[i1[stan]],
              color=None,
              zr=[0,300],xr=[10,20],yr=[5,10],size=20,label=name,xt=mag,yt='-2.5*log(cnts/exptime)')
        plots.plotp(ax[channel],smag[i2[stan]],
              2.5*np.log10(np.median((out.data[:,p1:p2]-sky[p1:p2])/out.header['EXPTIME']*900.,axis=1))[i1[stan]],
              color=None,
              zr=[0,300],xr=[10,20],yr=[5,10],size=20,label=name,xt=mag,yt='-2.5*log(cnts/exptime)')
        mags.append(smag[i2])
        spectroflux =np.median((out.data[:,p1:p2]-sky[p1:p2])/(out.header['EXPTIME'])*900.,axis=1)[i1]
        instmag = -2.5*np.log10(spectroflux)
        zeronorm = smag[i2]-instmag
        bd = np.where(~np.isfinite(zeronorm))[0]
        zeronorm[bd] = np.nan
        inst.append(instmag)

        if db is not None :
            cam = chan[channel]

            try: seeing = out.header['SEEING']
            except : 
                seeing = 0.
                if gcam != None :
                    mjd_obs = Time(out.header['DATE-OBS']).mjd
                    gnearest = np.argmin(np.abs(gcam['mjd']-mjd_obs))
                    gd = np.where(gcam['nstars'][gnearest]>0)[0]
                    seeing = np.median(gcam['fwhm'][gnearest][gd])
                elif plates != None :
                    try: 
                        j=np.where((plates['PLATE'] == int(plan['plateid'])) & 
                               (plates['MJD'] == int(plan['MJD']) ) )[0]
                        if len(j) > 0 : seeing = plates['SEEING50'][j[0]]
                    except : pass
                out.header['SEEING'] = seeing
            print('seeing: ', seeing)
            tab_exp = sdss.db_exp(exp_no,cam,out.header,config=header)
            gd = np.where(plug['mag'][i2,imag] > 0)[0]
            perc=np.nanpercentile(zeronorm[gd],[50,25,75])
            tab_exp['zeronorm' ] = [perc]

            tab_spec=sdss.db_spec(plug[i2],header,confSummary=conf)
            tab_spec['spectroflux'] = spectroflux
            tab_spec['zeronorm'] = zeronorm

            mjd = int(plan['MJD'])
            try : field = int(plan['SPEXP']['fieldid'][0])
            except : field = int(plan['plateid'])
            tab_visit=sdss.db_visit(mjd,field)

            db.ingest(schema+'.visit',tab_visit,onconflict='update')
            out=db.query(
                sql=('SELECT visit_pk from {:s}.visit where mjd = {:d} and field_id = {:d}')
                     .format(schema,mjd,field))
            tab_exp['visit_pk' ] = [out['visit_pk'][0]]
            try : db.ingest(schema+'.exposure',tab_exp,onconflict='update')
            except : pdb.set_trace()

            out=db.query(
                sql=("SELECT exp_pk,config_id from {:s}.exposure where exp_no = {:d} and camera = '{:s}'")
                     .format(schema,exp_no,cam))
            tab_spec['exp_pk'] = out['exp_pk'][0]
            db.ingest(schema+'.spectrum',tab_spec,onconflict='update')

      ax[channel].grid()
      ax[channel].legend()
      allmags.append(mags)
      allinst.append(inst)
    fig.suptitle(os.path.basename(planfile))
    fig.tight_layout()
    fig.savefig(planfile.replace('.par','.png'))
    return allmags,allinst
    
def do_visit(planfile=None,channel=0,clobber=False,nfibers=500,threads=12,display=None,maxobj=None) :
    """ Read raw image (eventually, 2D calibration and extract,
        using specified flat/trace
    """
    plan=yanny.yanny(planfile)

    # are all files already created?
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    for obj in objs[0:maxobj] :
        name=plan['SPEXP']['name'][obj][channel].astype(str)
        if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
            name=plan['SPEXP']['name'][obj][2].astype(str)
        if not os.path.exists(name) or clobber :  done = False
    if done :  return

    # set up Reducer
    red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_N']+'/'+plan['MJD'])
   
    # make Trace/PSF 
    iflat=np.where(plan['SPEXP']['flavor'] == b'flat')[0]
    name=plan['SPEXP']['name'][iflat][0][channel].astype(str)
    if len(plan['SPEXP']['name'][iflat][0]) > 2  and channel == 1 :
        name=plan['SPEXP']['name'][iflat][0][2].astype(str)
    if os.path.exists(name.replace('sdR','spTrace')) and not clobber : 
        trace=spectra.Trace('./'+name.replace('sdR','spTrace')) 
        f=np.loadtxt('./'+name.replace('sdR','spFlat1d').replace('.fit','.txt'))
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
        trace.write(name.replace('sdR','spTrace')) 
        flat1d=trace.extract(flat,threads=threads,nout=500,plot=display)
        f=np.median(flat1d.data[:,1500:2500],axis=1)
        f/=np.median(f)
        np.savetxt(name.replace('sdR','spFlat1d').replace('.fit','.txt'),f)


    # reduce and extract science frames
    # 1d flat
    fim=np.tile(f,(4224,1)).T
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    for obj in objs[0:maxobj] :
        name=plan['SPEXP']['name'][obj][channel].astype(str)
        if len(plan['SPEXP']['name'][obj]) > 2  and channel == 1 :
            name=plan['SPEXP']['name'][obj][2].astype(str)
        if os.path.exists(name.replace('sdR','sp1D')) and not clobber : 
            out=CCDData.read(name.replace('sdR','sp1D'))
        else :
            im=red.reduce(name,channel=channel)
            out=trace.extract(im,threads=threads,nout=500,plot=display)
            out.data /= fim
            out.uncertainty.array /= fim
            out.write(name.replace('sdR','sp1D'),overwrite=True)

    return out
