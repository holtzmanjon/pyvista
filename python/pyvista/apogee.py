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

# refsub subtracts the reference array from each quadrant with proper flipping
def refcorr_sub(image,ref):
    revref = np.flip(ref,axis=1)
    image[:,0:512] -= ref
    image[:,512:1024] -= revref
    image[:,1024:1536] -= ref
    image[:,1536:2048] -= revref
    return image


def refcorr(cube,head,mask=None,indiv=3,vert=True,horz=True,cds=True,noflip=False,
            silent=False,readmask=None,lastgood=None,plot=False,q3fix=False,keepref=False):
    """
    This corrects a raw APOGEE datacube for the reference pixels
    and reference output

    Parameters
    ----------
    cube : numpy array
       The raw APOGEE datacube with reference array.  This
         will be updated with the reference subtracted cube.
    head : Header
       The header for CUBE.
    mask : numpy array, optional
       Input bad pixel mask.
    indiv : int, optional
       Subtract the individual reference arrays after NxN median filter. If 
        If <0, subtract mean reference array. If ==0, no reference array subtraction
        Default is indiv=3.
    vert : bool, optional
       Use vertical ramp.  Default is True.
    horz : bool, optional
       Use horizontal ramp.  Default is True.
    cds : bool, optional
       Perform double-correlated sampling.  Default is True.
    noflip : bool, optional
       Do not flip the reference array.
    q3fix : bool, optional
       Fix issued with the third quadrant for MJD XXX.
    keepref : bool, optional
       Return the reference array in the output.
    silent : bool, optional
       Don't print anything to the screen.

    Returns
    -------
    out : numpy array
       The reference-subtracted cube.
    mask : numpy array
       The flag mask array.
    readmask : numpy array
       Mask indicating if reads are bad (0-good, 1-bad).

    Example
    -------

    out,mask,readmask = refcorr(cube,head)

    By J. Holtzman   2011
    Incorporated into ap3dproc.pro  D.Nidever May 2011
    Translated to Python  D.Nidever  Nov 2023
    """

    t0 = time.time()
    
    # refcorr does the "bias" subtraction, using the reference array and
    #    the reference pixels. Subtract a mean reference array (or individual
    #    with /indiv), then subtract vertical ramps from each quadrant using
    #    reference pixels, then subtract smoothed horizontal ramps

    # Number of reads
    ny,nx,nread = cube.shape

    # Create long output
    out = np.zeros((2048,2048,nread),int)
    if keepref:
        refout = np.zeros((512,2048,nread),int)

    # Ignore reference array by default
    # Default is to do CDS, vertical, and horizontal correction
    print('in refcorr, indiv: '+str(indiv))

    satval = 55000
    snmin = 10
    if indiv>0:
        hmax = 1e10
    else:
        hmax = 65530

    # Initalizing some output arrays
    if mask is None:
        mask = np.zeros((2048,2048),int)
    readmask = np.zeros(nread,int)

    # Calculate the mean reference array
    if silent==False:
        print('Calculating mean reference')
    meanref = np.zeros((2048,512),float)
    nref = np.zeros((2048,512),int)
    for i in range(nread):
        ref = cube[:,2048:2560,i].astype(float)
        # Calculate the relevant statistics
        mn = np.mean(ref[128:2048-128,128:512-128])
        std = np.std(ref[128:2048-128,128:512-128])
        hm = np.max(ref[128:2048-128,128:512-128])
        ref[ref>=satval] = np.nan        
        # SLICE business is just for special fast handling, ignored if
        #   not in header
        card = 'SLICE%03d' % i
        iread = head.get(card)
        if iread is None:
            iread = i+1
        if silent==False:
            print("\rreading ref: {:3d} {:3d}".format(i,iread), end='')
        # skip first read and any bad reads
        if (iread > 1) and (mn/std > snmin) and (hm < hmax):
            good = (np.isfinite(ref))
            meanref[good] += (ref[good]-mn)
            nref[good] += 1
            readmask[i] = 0
        else:
            if silent==False:
                print('\nRejecting: ',i,mn,std,hm)
            readmask[i] = 1
            
    meanref /= nref

    if silent == False:
        print('\nReference processing ')
        
    # Create vertical and horizontal ramp images
    rows = np.arange(2048,dtype=float)
    cols = np.ones(512,dtype=int)
    vramp = (rows.reshape(-1,1)*cols.reshape(1,-1))/2048
    vrramp = 1-vramp   # reverse
    cols = np.arange(2048,dtype=float)
    rows = np.ones(2048,dtype=int)
    hramp = (rows.reshape(-1,1)*cols.reshape(1,-1))/2048
    hrramp = 1-hramp
    clo = np.zeros(2048,float)
    chi = np.zeros(2048,float)

    if cds:
        cdsref = cube[:,:2048,1]
        
    # Loop over the reads
    lastgood = nread-1
    for iread in range(nread):
        # Do all operations as floats, then convert to int at the end
        
        # Subtract mean reference array
        im = cube[:,0:2048,iread].astype(float)

        # Deal with saturated pixels
        sat = (im > satval)
        nsat = np.sum(sat)
        if nsat > 0:
            if iread == 0:
                nsat0 = nsat
            im[sat] = 65535
            mask[sat] = (mask[sat] | pixelmask.getval('SATPIX'))
            # If we have a lot of saturated pixels, note this read (but don't do anything)
            if nsat > nsat0+2000:
                if lastgood == nread-1:
                    lastgood = iread-1
        else:
            nsat0 = 0
            
        # Pixels that are identically zero are bad, see these in first few reads
        bad = (im == 0)
        nbad = np.sum(bad)
        if nbad > 0:
            mask[bad] = (mask[bad] | pixelmask.getval('BADPIX'))
        
        if silent==False:
            print("\rRef processing: {:3d}  nsat: {:5d}".format(iread+1,nsat), end='')            

        # Skip this read
        if readmask[iread] > 0:
            im = np.nan
            out[:,:,iread] = int(-1e10)   # np.nan, int cannot be NaN
            if keepref:
                refout[:,:,iread] = 0
            continue
        
        # With cds keyword, subtract off first read before getting reference pixel values
        if cds:
            im -= cdsref.astype(int)

        # Use the reference array information
        ref = cube[:,2048:2560,iread].astype(int)
        # No reference array subtraction
        if indiv is None or indiv==0:
            pass
        # Subtract full reference array
        elif indiv==1:
            im = refcorr_sub(im,ref)
            ref -= ref
        # Subtract median-filtered reference array
        elif indiv>1:
            mdref = medfilt2d(ref,[indiv,indiv])
            im = refcorr_sub(im,mdref)
            ref -= mdred
        # Subtract mean reference array
        elif indiv<0:
            im = refcorr_sub(im,meanref)
            ref -= meanref
            
        # Subtract vertical ramp, using edges
        if vert:
            for j in range(4):
                rlo = np.nanmean(im[2:4,j*512:(j+1)*512])
                rhi = np.nanmean(im[2045:2047,j*512:(j+1)*512])
                im[:,j*512:(j+1)*512] = (im[:,j*512:(j+1)*512].astype(float) - rlo*vrramp).astype(int)
                im[:,j*512:(j+1)*512] = (im[:,j*512:(j+1)*512].astype(float) - rhi*vramp).astype(int)
                #im[:,j*512:(j+1)*512] -= rlo*vrramp
                #im[:,j*512:(j+1)*512] -= rhi*vramp                
                
        # Subtract horizontal ramp, using smoothed left/right edges
        if horz:
            clo = np.nanmean(im[:,1:4],axis=1)
            chi = np.nanmean(im[:,2044:2047],axis=1)
            sm = 7
            slo = utils.nanmedfilt(clo,sm,mode='edgecopy')
            shi = utils.nanmedfilt(chi,sm,mode='edgecopy')

            # in the IDL code, this step converts "im" from int to float
            if noflip:
                im = im.astype(float) - slo.reshape(-1,1)*hrramp
                im = im.astype(float) - shi.reshape(-1,1)*hramp
                #im -= slo.reshape(-1,1)*hrramp
                #im -= shi.reshape(-1,1)*hramp                
            else:
                #bias = (rows#slo)*hrramp+(rows#shi)*hramp
                # just use single bias value of minimum of left and right to avoid bad regions in one
                bias = np.min([slo,shi],axis=0).reshape(-1,1) * np.ones((1,2048))
                fbias = bias.copy()
                fbias[:,512:1024] = np.flip(bias[:,512:1024],axis=1)
                fbias[:,1536:2048] = np.flip(bias[:,1536:2048],axis=1)
                im = im.astype(float) - fbias
                #im -= fbias.astype(int)                
                
        # Fix quandrant 3 issue
        if q3fix:
            q2m = np.median(im[:,923:1024],axis=1)
            q3a = np.median(im[:,1024:1125],axis=1)
            q3b = np.median(im[:,1435:1536],axis=1)
            q4m = np.median(im[:,1536:1637],axis=1)
            q3offset = ((q2m-q3a)+(q4m-q3b))/2.
            im[:,1024:1536] += medfilt(q3offset,7).reshape(-1,1)*np.ones((1,512))
            
        # Make sure saturated pixels are set to 65535
        #  removing the reference values could have
        #  bumped them lower
        if nsat > 0:
            im[sat] = 65535

        # Stuff final values into our output arrays
        #   and convert form float to int
        out[:,:,iread] = im.astype(int)
        if keepref:
            refout[:,:,iread] = ref
            
    # Mask the reference pixels
    mask[0:4,:] = (mask[0:4,:] | pixelmask.getval('BADPIX'))
    mask[2044:2048,:] = (mask[2044:2048,:] | pixelmask.getval('BADPIX'))
    mask[:,0:4] = (mask[:,0:4] | pixelmask.getval('BADPIX'))
    mask[:,2044:2048] = (mask[:,2044:2048] | pixelmask.getval('BADPIX'))

    if silent==False:
        print('')
        print('lastgood: ',lastgood)
        
    # Keep the reference array in the output
    if keepref:
        out = np.hstack((out,refout))
        
    return out,mask,readmask
    
