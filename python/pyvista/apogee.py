from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.nddata import support_nddata
import astropy.units as u
import esutil
import glob
import numpy as np
import pdb
import os
from pydl.pydlutils.yanny import yanny
from holtztools import plots, match
from pyvista.dataclass import Data
import matplotlib.pyplot as plt
import multiprocessing as mp
import yaml
from pyvista import imred, spectra,sdss,stars,image,refcorr

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
          data3d=np.zeros([nreads,2048,2560],dtype=np.int16)
          data3d[0]=data[0:2048,0:2560]
        else :
          data = np.add(data,raw,dtype=np.int16)
          data = np.add(data,avg_dcounts,dtype=np.int16)
          data3d[read-1]=data[0:2048,0:2560]

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
    out= (cube[-1,0:2048,0:2048].astype(np.float32) - cube[1,0:2048,0:2048].astype(np.float32) )
    return Data(data=vert(out),header=header,unit=u.dimensionless_unscaled)

def utr(file,dark=None) :
    header = fits.open(file)[1].header
    cube = unzip(file)
    #cube,mask,readmask=refcorr.refcorr(cube,header)
    nreads=len(cube)
    x=np.arange(nreads-1)
    fit = np.polyfit(x,cube[1:,:,0:2048].reshape(nreads-1,2048*2048),deg=1)
    out=fit[0,:].reshape(2048,2048)*nreads-1
    return Data(data=out,header=header,unit=u.dimensionless_unscaled)

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
    for channel in [0,1,2] :
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
    if dir == '/' : dir='./'

    # are all files already created?
    done =True
    for obj in plan['APEXP'][0:maxobj] :
        exp_no = int(obj['name'])
        if obj['flavor']  != 'object' : continue
        name='ap1D-{:s}-{:08d}.fits'.format(chan[channel],exp_no)
        if not os.path.exists(dir+name) or clobber :  done = False
    if done :  return

    # set up Reducer
    if plan['instrument'] == 'apogee-n' :
        red=imred.Reducer('APOGEE',dir=os.environ['APOGEE_DATA_N']+'/'+str(plan['mjd']))
        prefix='ap'
    else :
        red=imred.Reducer('APOGEES',dir=os.environ['APOGEE_DATA_S']+'/'+str(plan['mjd']))
        prefix='as'

    # get Dark
    if int(plan['darkid']) > 0 :
        name=prefix+'Dark-{:s}-{:08d}.fits'.format(chan[channel],int(plan['darkid']))
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
        trace=spectra.Trace(transpose=red.transpose,rad=2,lags=np.arange(-3,4),sc0=1024,rows=[4,2045])
        ff=np.sum(flat.data[:,1000:1100],axis=1)
        if channel==0 : thresh=40000
        else : thresh=40000
        #peaks,fiber=spectra.findpeak(ff,diff=10,bundle=10000,thresh=thresh)
        peaks,fiber=trace.findpeak(flat,diff=11,bundle=10000,thresh=100,smooth=2)
        print('found {:d} peaks'.format(len(peaks)))
        trace.trace(flat,peaks[0:nfibers],index=fiber[0:nfibers],skip=4)
        trace.write(dir+name)

    # now reduce and extract flux for 1D flat
    name='ap1D-{:s}-{:08d}.fits'.format(chan[channel],int(plan['fluxid']))
    print('flux: ', name)
    if os.path.exists(dir+name) and not clobber : 
        flux=Data.read(dir+name)
    else :
        im=red.reduce(int(plan['fluxid']),channel=channel,dark=dark)
        flux=trace.extract(im,threads=threads,nout=300)
        flux.write(dir+name,overwrite=True)
    f=np.median(flux.data[:,500:1500],axis=1)
    f/=np.percentile(f,[90])[0]
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
            out=Data.read(dir+name)
        else :
            im=red.reduce(exp_no,channel=channel,dark=dark,display=display)
            out=trace.extract(im,threads=threads,nout=300,display=display)
            out.data /= fim
            out.uncertainty.array /= fim
            out.write(dir+name,overwrite=True)

    return out

def mkyaml(mjd,obs='apo') :

    if obs == 'apo' :
        red=imred.Reducer('APOGEE',dir=os.environ['APOGEE_DATA_N']+'/'+str(mjd))
        obj0=-1  # image to use to match flat
    else :
        red=imred.Reducer('APOGEES',dir=os.environ['APOGEE_DATA_S']+'/'+str(mjd))
        obj0=0   # image to use to match flat

    if mjd > 59600 :
        files = red.log(cols=['DATE-OBS','FIELDID','EXPTYPE','CONFIGID'],ext='apz',hdu=1,channel='-b-')
        files['NAME'] = files['CONFIGID']
    else :
        files = red.log(cols=['DATE-OBS','EXPTYPE','PLATEID','NAME'],ext='apz',hdu=1,channel='-b-')
        files['CONFIGID'] = files['PLATEID']
        files['FIELDID'] = files['PLATEID']

    fields = set(files['FIELDID'])
    for field in fields :

        obj = np.where((files['FIELDID'].astype(int) == int(field)) & (files['EXPTYPE'] == 'OBJECT'))[0]
        if len(obj) < 1 : continue

        darkid=0
        fp = open('apPlan-{:s}-{:d}.yaml'.format(field,mjd),'w')
        fp.write('plateid: {:s}\n'.format(field))
        if obs == 'lco' :
            fp.write('telescope: lco25m\n')
            fp.write('instrument: apogee-s\n')
        else :
            fp.write('telescope: apo25m\n')
            fp.write('instrument: apogee-n\n')
        fp.write('mjd: {:d}\n'.format(mjd))
        fp.write('darkid: {:d}\n'.format(darkid))

        # get QUARTZFLAT nearest in time for psfid
        flats = np.where((files['EXPTYPE'] == 'QUARTZFLAT'))[0]
        jmin= np.argmin(np.abs(Time(files['DATE-OBS'][flats])-Time(files['DATE-OBS'][obj[obj0]])))
        expno=files[flats[jmin]]['FILE'].split('-')[2].replace('.apz','')
        psfid=expno
        fp.write('psfid: {:s}\n'.format(psfid))

        # get DOMEFLAT nearest in time for fluxid
        flats = np.where((files['EXPTYPE'] == 'DOMEFLAT'))[0]
        jmin= np.argmin(np.abs(Time(files['DATE-OBS'][flats])-Time(files['DATE-OBS'][obj[obj0]])))
        expno=files[flats[jmin]]['FILE'].split('-')[2].replace('.apz','')
        fluxid=expno
        fp.write('fluxid: {:s}\n'.format(expno))

        if len(obj) > 0 :
            fp.write('plugmap: {:s}\n'.format(files[obj[0]]['NAME'].lower()))
            fp.write('APEXP:\n')
            for o in obj :
                expno=files[o]['FILE'].split('-')[2].replace('.apz','')
                fp.write('- name: {:s}\n'.format(expno))
                fp.write('  flavor: {:s}\n'.format(files[o]['EXPTYPE'].lower()))
                fp.write('  single: -1\n')
                fp.write('  singlename: none\n')
        fp.close()

@support_nddata
def test_fit_lines(data,coeffs,skip=10,thresh=3000,binned=False,sub=False,size=4,nherm=4,threads=0) :
    """ Fit 2D gaussians to 'arc' frame and get surface fits of parameters
    """
    lines = stars.find(data,thresh=thresh)[::skip]
    for i,(x,y) in enumerate(zip(lines['x'],lines['y'])) :
        p0=[x,y]
        low=[0.,0.]
        high=[2048.,2048.]
        for coeff in coeffs :
            val=image.mk2d(x,y,coeff)
            p0.extend(image.mk2d(x,y,val))
            low.extend(0.999*val)
            high.extend(1.001*val)
        p0.extend(0.)
        low.extend(-np.inf)
        high.extend(-np.inf)
        p0=np.array(p0)
        low=np.array(low)
        high=np.array(high)
        low[5] = 0.
        high[5] = np.inf
        
        image.ghfit2d(data,x,y,size=size,nherm=nherm,p0=p0,bounds=(low,high))


@support_nddata
def fit_lines(data,uncertainty=None,display=None,skip=10,thresh=3000,binned=False,sub=False,size=5,nherm=1,threads=0) :
    """ Fit PSF function to 'arc' frame and get surface fits of parameters
    """

    # find objects to fit
    lines = stars.find(data,thresh=thresh)[::skip]
    print(len(lines),' lines found')

    # set up input parameters
    pars=[]
    for i,(x,y) in enumerate(zip(lines['x'],lines['y'])) :
        print(i)
        pars.append((data,x,y,size,binned,nherm))

    # run them all, with multithreading if desired
    if threads > 0 :
        pool = mp.Pool(threads)
        outputs = pool.map_async(image.ghfit2d_thread, pars).get()
        pool.close()
        pool.join()
    else :
        outputs=[]
        for par in pars :
            try : outputs.append(image.ghfit2d_thread(par))
            except :
                print('error with fit: ', x,y)

    # subtract fits from frame if requested. Can't do in ghfit2d if multithreaded
    if sub :
        for par,output in zip(pars,outputs) :
            param=output[0]
            x0,y0=par[1],par[2]
            # use initial guess to get peak
            z=data[int(y0)-size:int(y0)+size+1,int(x0)-size:int(x0)+size+1]
            # refine subarray around peak
            ycen,xcen=np.unravel_index(np.argmax(z),z.shape)
            xcen+=(int(x0)-size)
            ycen+=(int(y0)-size)
            y,x=np.mgrid[ycen-size:ycen+size+1,xcen-size:xcen+size+1]
            try :data[ycen-size:ycen+size+1,xcen-size:xcen+size+1]-=image.gh2d_wrapper(np.array([x,y]),nherm,param).reshape(2*size+1,2*size+1)
            except : pass

    # load up output parameters and residuals
    params=[]
    res=[]
    for output in outputs : 
        try :
            res.append((output[2]['fvec']**2).sum())
            params.append(output[0])
        except : pass
    params=np.array(params)
    res=np.array(res)

    # select good fits and do quadratic fits to the parameters
    gd=np.where((params[:,0] > 0) & (params[:,0]<2048) & (params[:,1] > 0) & (params[:,1] < 2048) &
                (np.abs(params[:,2]) < 1) & (np.abs(params[:,4]) < 1) & (res<1.e6) )[0]
    y,x=np.mgrid[0:2048,0:2048]

    coeffs=[]
    fig,ax=plots.multi(3,1,figsize=(12,4),hspace=0.001,wspace=0.001,sharex=True,sharey=True)
    fit=[2,3,4]
    for i,ifit in enumerate(fit) :
        c=image.fit2d(params[gd,0],params[gd,1],params[gd,ifit])
        surf = image.mk2d(x.flatten(),y.flatten(),c).reshape(2048,2048)
        min,max=image.minmax(params[gd,ifit],low=3,high=3)
        ax[i].imshow(surf,vmin=min,vmax=max,origin='lower')
        ax[i].scatter(params[gd,0],params[gd,1],c=params[gd,ifit],vmin=min,vmax=max,s=2)
        coeffs.append(c)

    fig,ax=plots.multi(nherm,nherm,figsize=(12,12),hspace=0.001,wspace=0.001,sharex=True,sharey=True)
    for i,ifit in enumerate(range(nherm*nherm)) :
        vals=params[gd,5+ifit]/params[gd,5]
        c=image.fit2d(params[gd,0],params[gd,1],vals)
        surf = image.mk2d(x.flatten(),y.flatten(),c).reshape(2048,2048)
        min,max=image.minmax(vals,low=3,high=3)
        ax[i//nherm,i%nherm].imshow(surf,vmin=min,vmax=max,origin='lower')
        ax[i//nherm,i%nherm].scatter(params[gd,0],params[gd,1],c=vals,vmin=min,vmax=max,s=2)
        coeffs.append(c)

    if display is not None :
        display.tv(data)
        plots.plotc(display.ax,params[:,0],params[:,1],params[:,2],size=30,zr=[1,3])
        pdb.set_trace()
        plots.plotc(display.ax,params[:,0],params[:,1],params[:,3],size=30,zr=[1,3])
        pdb.set_trace()
        plots.plotc(display.ax,params[:,0],params[:,1],params[:,4],size=30)
        pdb.set_trace()

    return params, coeffs

def get_fiberind(lines,trace) :
    """ Get associated fiber with every line, given a Trace stobject
    """

    # get the row of each trace at the central column
    yc=[]
    for tr in trace.model :
        yc.append(tr(1024))
    yc=np.array(yc)

    lines['fiberind'] = -1
    for line in lines :
        # search traces where line['y'] is within 20 pixels of central pixel
        gd = np.where(np.abs(line['y']-yc) < 20)[0]
        i1=gd.min()
        i2=gd.max()
        for i,(tr,ind) in enumerate(zip(trace.model[i1:i2+1],trace.index[i1:i2+1])) :
            # get y position of trace at x position of line
            y=tr(line['x'])
            d=np.abs(line['y']-y)
            if d.min() < 2 :
                line['fiberind'] = trace.index[gd.min()+i]
                print(line['x'],line['y'],line['fiberind'])



