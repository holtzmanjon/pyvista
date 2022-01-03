import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from pydl.pydlutils import yanny
from pyvista import imred, spectra, sdss
from tools import match,plots
from ccdproc import CCDData
import multiprocessing as mp

def visit(planfile,tracefile=None) :
    """ Reduce BOSS visit

        Driver for parallell processing of b and r channels
        Makes plots of median counts vs mag
    """
    # reduce b1 and r1 in parallel
    procs=[]
    for channel in [0,1] :
        kw={'planfile' : planfile, 'channel' : channel, 'clobber' : False}
        procs.append(mp.Process(target=do_visit,kwargs=kw))
    for proc in procs : proc.start()
    for proc in procs : proc.join()

    plan=yanny.yanny(planfile)

    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    fig,ax=plots.multi(1,2)
    allmags=[]
    allinst=[]
    for channel in [0,1] :
      mags=[]
      inst=[]
      for obj in objs :
        name=plan['SPEXP']['name'][obj][channel].astype(str)
        print(name)
        out=CCDData.read(name.replace('sdR','sp1D'))
        mapname=plan['SPEXP']['mapname'][obj].astype(str)
        if mapname == 'fps' :
            plug=sdss.config(out.header['CONFID'],specid=1)[0]
            isky=np.where(plug['category'] == b'sky_boss')[0]
        else :
            plug=sdss.config(os.environ['MAPPER_DATA_N']+'/'+mapname.split('-')[1]+'/plPlugMapM-'+mapname+'.par',specid=1,struct='PLUGMAPOBJ')[0]
            isky=np.where(plug['objType'] == b'SKY')[0]
        i1,i2=match.match(np.arange(500)+1,plug['fiberId'])
        if channel == 0 : 
            mag='g'
            imag=1
        else : 
            mag='i'
            imag=3
        skyfiber=plug['fiberId'][isky]
        sky=np.median(out.data[skyfiber-1,:])
        print(len(skyfiber),sky)
        rad=np.sqrt(plug['xFocal'][i2]**2+plug['yFocal'][i2]**2)
        plots.plotp(ax[channel],plug['mag'][i2,imag],2.5*np.log10(np.median((out.data-sky)/out.header['EXPTIME'],axis=1))[i1],color=None,
                    zr=[0,300],xr=[10,20],yr=[0,5],size=20,label=name,xt=mag,yt='-2.5*log(cnts/exptime)')
        mags.append(plug['mag'][i2,imag])
        inst.append(-2.5*np.log10(np.median((out.data-sky)/out.header['EXPTIME'],axis=1))[i1])
      ax[channel].grid()
      ax[channel].legend()
      allmags.append(mags)
      allinst.append(inst)
    fig.suptitle(planfile)
    fig.tight_layout()
    fig.savefig(planfile.replace('.par','.png'))
    return allmags,allinst
    
def do_visit(planfile=None,channel=0,clobber=False,nfibers=50) :
    """ Read raw image (eventually, 2D calibration and extract,
        using specified flat/trace
    """
    plan=yanny.yanny(planfile)

    # are all files already created?
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    for obj in objs :
        name=plan['SPEXP']['name'][obj][channel].astype(str)
        if not os.path.exists(name) or clobber :  done = False
    if done :  return

    # set up Reducer
    red=imred.Reducer('BOSS',dir=os.environ['BOSS_SPECTRO_DATA_N']+'/'+plan['MJD'])
   
    # make Trace/PSF 
    iflat=np.where(plan['SPEXP']['flavor'] == b'flat')[0]
    name=plan['SPEXP']['name'][iflat][0][channel].astype(str)
    if os.path.exists(name.replace('sdR','spTrace')) and not clobber : 
        trace=spectra.Trace('./'+name.replace('sdR','spTrace')) 
    else :
        flat=red.reduce(name,channel=0)
        trace=spectra.Trace(transpose=red.transpose,rad=3,lags=np.arange(-3,4))
        ff=np.sum(flat.data[2000:2100],axis=0)
        if channel==0 : thresh=0.4e6
        else : thresh=0.2e6
        peaks,fiber=spectra.findpeak(ff,thresh=thresh)
        print('found {:d} peaks'.format(len(peaks)))
        trace.trace(flat,peaks[0:nfibers],index=fiber[0:nfibers])
        trace.write(name.replace('sdR','spTrace')) 

    # reduce and extract science frames
    objs=np.where(plan['SPEXP']['flavor'] == b'science')[0]
    for obj in objs :
        name=plan['SPEXP']['name'][obj][channel].astype(str)
        if os.path.exists(name.replace('sdR','sp1D')) and not clobber : 
            out=CCDData.read(name.replace('sdR','sp1D'))
        else :
            im=red.reduce(name,channel=channel)
            out=trace.extract(im,threads=1,nout=500)
            out.write(name.replace('sdR','sp1D'),overwrite=True)

    return out
