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

def visit(planfile,tracefile=None,clobber=False,db=None,schema='obs2',maxobj=None) :
    """ Reduce an APOGEE visit
 
        Driver to do 3 chips in parallel
        Makes median flux plots
    """    

    # reduce channels in parallel
    chan=['a','b','c' ]
    procs=[]
    for channel in [1] :
        kw={'planfile' : planfile, 'channel' : channel, 'clobber' : clobber, 'maxobj' : maxobj}
        procs.append(mp.Process(target=do_visit,kwargs=kw))
    for proc in procs : proc.start()
    for proc in procs : proc.join()

    plan=yaml.load(open(planfile,'r'), Loader=yaml.BaseLoader)

    fig,ax=plots.multi(1,1)
    allmags=[]
    allinst=[]

    # read reduced files and configuration/plugmap file
    for ichan,channel in enumerate([1]) :
      mags=[]
      inst=[]
      for obj in plan['APEXP'][0:maxobj] :
        exp_no = int(obj['name'])
        if obj['flavor']  != 'object' : continue
        name='ap1D-{:s}-{:08d}.fits'.format(chan[channel],exp_no)
        out=CCDData.read(name)
        print(name,out.header['NREAD'])
        mapname=plan['plugmap']
        if np.char.find(mapname,'conf') >=0 :
            conf= True
            config_id = out.header['CONFIGID']
            plug,header=sdss.config(config_id,specid=2,useconfF=True)
            hmag=plug['h_mag']
        else :
            conf= False
            plug,header=sdss.config(os.environ['MAPPER_DATA_N']+'/'+
                  mapname.split('-')[1]+'/plPlugMapM-'+mapname+'.par',
                  specid=2,struct='PLUGMAPOBJ')
            plate=int(mapname.split('-')[0])
            # substitute H mag from plateHoles
            holes=yanny('{:s}/plates/{:04d}XX/{:06d}/plateHolesSorted-{:06d}.par'.format(
                  os.environ['PLATELIST_DIR'],plate//100,plate,plate))
            h=esutil.htm.HTM()
            m1,m2,rad=h.match(plug['ra'],plug['dec'],
                  holes['STRUCT1']['target_ra'],holes['STRUCT1']['target_dec'],
                  0.1/3600.,maxmatch=500)
            hmag=plug['mag'][:,1]
            hmag[m1]=holes['STRUCT1']['tmass_h'][m2]
            plug['mag'][:,1] = hmag
            star = np.where(plug['objType'] != b'SKY')[0]
            std = np.where(plug['objType'] != b'SPECTROPHOTO_STD')[0]

        # make sure we have objects (could be APOGEE with all BOSS targets)
        if len(np.where(hmag > 0.)[0]) == 0 : continue

        # match configuration/plugmap file with fiber
        i1,i2=match.match(300-np.arange(300),plug['fiberId'])
        mag='H'
        rad=np.sqrt(plug['xFocal'][i2]**2+plug['yFocal'][i2]**2)
        plots.plotp(ax,hmag[i2],
            +2.5*np.log10(np.median(out.data/(out.header['NREAD']-2),axis=1))[i1],
            color=None, zr=[0,300],xr=[8,15],size=20,label=name,xt=mag,
            yt='-2.5*log(cnts/read)')
        mags.append(hmag[i2])
        spectroflux =np.median(out.data/(out.header['NREAD']-2),axis=1)[i1]
        instmag = -2.5*np.log10(np.median(out.data/(out.header['NREAD']-2),axis=1))[i1]
        zeronorm = hmag[i2]-instmag
        bd = np.where(~np.isfinite(zeronorm))[0]
        zeronorm[bd] = np.nan
        inst.append(instmag)

        if db is not None :
            cam = chan[channel]
            tab_exp = sdss.db_exp(exp_no,cam,out.header,config=header)
            gd = np.where(hmag[i2] > 0)[0]
            perc=np.nanpercentile(zeronorm[gd],[50,25,75])
            tab_exp['zeronorm' ] = [perc]
      
            tab_spec=sdss.db_spec(plug[i2],header,confSummary=conf)
            tab_spec['spectroflux'] = spectroflux
            tab_spec['zeronorm'] = zeronorm

            mjd = int(plan['mjd'])
            try : field = int(plan['field_id'])
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
 
      ax.grid()
      ax.legend()
      allmags.append(mags)
      allinst.append(inst)
    fig.suptitle(os.path.basename(planfile))
    fig.tight_layout()
    fig.savefig(os.path.basename(planfile).replace('.yaml','.png'))

    return allmags,allinst
    
def do_visit(planfile=None,channel=0,clobber=False,nfibers=300,threads=24,maxobj=None) :
    """ Read raw image (eventually, 2D calibration) and extract,
        using specified flat/trace
    """
    chan=['a','b','c' ]
    plan=yaml.load(open(planfile,'r'), Loader=yaml.BaseLoader)
    # are all files already created?
    done =True
    for obj in plan['APEXP'][0:maxobj] :
        exp_no = int(obj['name'])
        if obj['flavor']  != 'object' : continue
        name='ap1D-{:s}-{:08d}.fits'.format(chan[channel],exp_no)
        if not os.path.exists(name) or clobber :  done = False
    if done :  return

    # set up Reducer
    red=imred.Reducer('APOGEE',dir=os.environ['APOGEE_DATA_N']+'/'+str(plan['mjd']))

    # get Dark
    if int(plan['darkid']) > 0 :
        name='apDark-{:s}-{:08d}.fits'.format(chan[channel],int(plan['darkid']))
        try :
           dark=fits.open('{:s}/{:s}/cal/{:s}/darkcorr/{:s}'.format(os.environ['APOGEE_REDUX'],plan['apogee_drp_ver'],plan['instrument'],name))[1].data
        except :
           dark=fits.open(os.environ['APOGEE_REDUX']+'/cal/darkcorr/{:s}'.format(name))[1].data
    else : dark = None

    # get Trace/PSF if needed
    name='apTrace-{:s}-{:08d}.fits'.format(chan[channel],int(plan['psfid']))
    if os.path.exists(name) and not clobber : 
        trace=spectra.Trace('./'+name)
    else :
        flat=red.reduce(int(plan['psfid']),channel=channel,dark=dark)
        trace=spectra.Trace(transpose=red.transpose,rad=2,lags=np.arange(-3,4))
        ff=np.sum(flat.data[:,1000:1100],axis=1)
        if channel==0 : thresh=40000
        else : thresh=40000
        peaks,fiber=spectra.findpeak(ff,thresh=thresh)
        print('found {:d} peaks'.format(len(peaks)))
        trace.trace(flat,peaks[0:nfibers],index=fiber[0:nfibers],skip=4)
        trace.write(name)

    # now reduce and extract flux
    name='ap1D-{:s}-{:08d}.fits'.format(chan[channel],int(plan['fluxid']))
    print('flux: ', name)
    if os.path.exists(name) and not clobber : 
        flux=CCDData.read(name)
    else :
        im=red.reduce(int(plan['fluxid']),channel=channel,dark=dark)
        flux=trace.extract(im,threads=threads,nout=300)
        flux.write(name,overwrite=True)

    # now reduce and extract object
    for obj in plan['APEXP'][0:maxobj] :
        exp_no = int(obj['name'])
        if obj['flavor']  != 'object' : continue
        name='ap1D-{:s}-{:08d}.fits'.format(chan[channel],exp_no)
        if os.path.exists(name) and not clobber : 
            out=CCDData.read(name)
        else :
            im=red.reduce(exp_no,channel=channel,dark=dark)
            out=trace.extract(im,threads=threads,nout=300)
            out.write(name,overwrite=True)

    return out
