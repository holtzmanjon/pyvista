import numpy as np
import pdb
import time
from pyvista import bitmask
from scipy.signal import medfilt2d

pixelmask=bitmask.PixelBitMask()

def refcorr_sub(image,ref):
    revref = np.flip(ref,axis=1)
    image[:,0:512] -= ref
    image[:,512:1024] -= revref
    image[:,1024:1536] -= ref
    image[:,1536:2048] -= revref
    return image


def refcorr(cube,head,mask=None,indiv=3,vert=False,horz=False,cds=True,noflip=False,
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
    nread,ny,nx = cube.shape

    # Create long output
    out = np.zeros((nread,2048,2048),int)
    if keepref:
        refout = np.zeros((nread,2048,512),int)

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
        ref = cube[i,:,2048:2560].astype(float)
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
        if (iread > 1) and (hm<hmax) : # and (mn/std > snmin) :
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
        cdsref = cube[1,:,:2048]
        
    # Loop over the reads
    lastgood = nread-1
    for iread in range(nread):
        # Do all operations as floats, then convert to int at the end
        
        # Subtract mean reference array
        im = cube[iread,:, 0:2048].astype(float)

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
            out[iread,:,:] = int(-1e10)   # np.nan, int cannot be NaN
            if keepref:
                refout[iread,:,:] = 0
            continue
        
        # With cds keyword, subtract off first read before getting reference pixel values
        if cds:
            im -= cdsref.astype(int)

        # Use the reference array information
        ref = cube[iread,:,2048:2560].astype(int)
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
            ref -= mdref
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
                
        # Fix quadrant 3 issue
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
        out[iread,:,:] = im.astype(int)
        if keepref:
            refout[iread,:,:] = ref
            
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

