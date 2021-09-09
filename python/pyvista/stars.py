# routines to deal with stellar images

import copy
import numpy as np
import pdb
import astropy
import os
import subprocess
import tempfile
import multiprocessing as mp
from astropy.io import fits
from astropy.table import Table, Column, vstack
from astropy.nddata import support_nddata
from astropy.time import Time
from pyvista import mmm
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture, CircularAnnulus,aperture_photometry

def mark(tv,stars=None,rad=3,auto=False,color='m',new=False,exit=False):
    """ Interactive mark stars on TV, or recenter current list 

    Args : 
           tv  : TV instance from which user will mark stars
           stars =   : existing star table
           auto=  (bool) : if True, recentroid from existing position
           radius= (int): radius to use for centroiding and for size of circles (default=3)
           color= (char) : color for circles (default='m')
    """

    # clear display and mark current star list( if not new)
    if new: tv.tvclear()
    try: dateobs=Time(tv.hdr['DATE-OBS'],format='fits')
    except: dateobs=None
    cards=['EXPTIME','FILTER','AIRMASS']
    types=['f4','S','f4']
    if stars == None :
        stars = Table(names=('id','x', 'y'), dtype=('i4','f4', 'f4'))
        stars['x'].info.format = '.2f'
        stars['y'].info.format = '.2f'
        if dateobs is not None :
            stars.add_column(Column([],name='MJD',dtype=('f8')))
            stars['MJD'].info.format = '.6f'
        for icard,card in enumerate(cards) :
            try: stars.add_column(Column([],name=card,dtype=(types[icard])))
            except: pass
    else :
        if auto :
            # with auto option, recentroid and update from current header
            for star in stars :
                x,y = centroid(tv.img,star['x'],star['y'],rad)
                star['x'] = x
                star['y'] = y
                if dateobs is not None : star['MJD'] = dateobs.mjd
                for icard,card in enumerate(cards) :
                    try: star[card] = tv.hdr[card]
                    except: pass
        # display stars
        for star in stars : tv.tvcirc(star['x'],star['y'],rad,color=color)
        if exit : return stars

    istar=len(stars)+1
    while True :
        key,x,y = tv.tvmark()
        if key == 'q' or key == 'e' : break
        if key == 'i' :
            # add at nearest integer pixel
            x = round(x)
            y = round(y)
        elif key == 'c' :
            # centroid around marked position
            x,y = centroid(tv.img,x,y,rad)

        # add blank row, recognizing that we may have added other columns
        stars.add_row()
        stars[len(stars)-1]['id'] = istar
        stars[len(stars)-1]['x'] = x
        stars[len(stars)-1]['y'] = y
        tv.tvcirc(x,y,rad,color=color)
        if dateobs is not None :
            stars[len(stars)-1]['MJD'] = dateobs.mjd
        for icard,card in enumerate(cards) :
            try: stars[len(stars)-1][card] = tv.hdr[card]
            except: pass
        #if exptime is not None :
        #    stars[len(stars)-1]['EXPTIME'] = exptime
        #if filt is not None :
        #    stars[len(stars)-1]['FILTER'] = filt
        istar+=1

    return stars

@support_nddata
def add_coord(data,stars,wcs=None) :

    if wcs is not None :
        ra,dec=wcs.wcs_pix2world(stars['x'],stars['y'],0)
        stars['RA'] = ra
        stars['DEC'] = dec

@support_nddata
def photom(data,stars,uncertainty=None,rad=[3],skyrad=None,tv=None,gain=1,rn=0,mag=True,utils=True) :
    """ Aperture photometry of input image with current star list
    """

    # input radius(ii) in a list
    if type(rad) is int or type(rad) is float: rad = [rad]
   
    # uncertainty either specified in array, or use gain/rn, but not both
    if uncertainty is not None :
        if type(uncertainty) is not astropy.nddata.nduncertainty.StdDevUncertainty :
           raise Exception('uncertainty must be StdDevUncertainty ')
        uncertainty_data = uncertainty.array
    else :
        uncertainty_data = np.sqrt(data/gain + rn**2)
        
    # Add new output columns to table, removing them first if they exist already
    emptycol = Column( np.empty(len(stars))*np.nan )
    for r in rad :
        if type(r) is int : fmt='{:d}'
        else : fmt='{:.1f}'
        for suffix in ['','err'] :
            name=('aper'+fmt+suffix).format(r)
            try : stars.remove_column(name)
            except: pass
            stars.add_column(emptycol,name=name)
            if mag : stars[name].info.format = '.3f'
            else : stars[name].info.format = '.1f'
    try : stars.remove_column('sky')
    except: pass
    stars.add_column(emptycol,name='sky')
    stars['sky'].info.format = '.2f'
    try : stars.remove_column('skysig')
    except: pass
    stars.add_column(emptycol,name='skysig')
    stars['skysig'].info.format = '.2f'
    cnts=[]
    cntserr=[]

    # Create pixel index arrays
    pix = np.mgrid[0:data.shape[0],0:data.shape[1]]
    ypix = pix[0]
    xpix = pix[1]

    # loop over each stars
    for istar in range(len(stars)) :
        star=stars[istar]
        dist2 = (xpix-star['x'])**2 + (ypix-star['y'])**2

        # get sky if requested
        if skyrad is not None :
            if utils :
                try :
                    sky_aperture = CircularAnnulus((star['x'],star['y']),
                                        r_in=skyrad[0], r_out=skyrad[1]) 
                    sky_mask = sky_aperture.to_mask(method='center')
                    mask=sky_mask.data
                    skymean, skymedian, skysig = sigma_clipped_stats(
                                        sky_mask.multiply(data)[mask>0])
                    sky=skymean
                    sigsq=skysig**2
                except :
                    sky = 0.
                    sigsq = 0.
            else :
                gd = np.where((dist2 > skyrad[0]**2) & 
                              (dist2 < skyrad[1]**2) ) 
                sky,skysig,skyskew,nsky = mmm.mmm(data[gd[0],gd[1]].flatten())
                sigsq=skysig**2/nsky
            if tv is not None :
                tv.tvcirc(star['x'],star['y'],skyrad[0],color='g')
                tv.tvcirc(star['x'],star['y'],skyrad[1],color='g')
        else : 
            sky =0.
            skysig= 0.
            sigsq =0.

        # photutils aperture photometry handles pixels on the edges
        apertures = [ CircularAperture((star['x'],star['y']),r) for r in rad ]
        aptab = aperture_photometry(data,apertures,error=uncertainty_data)

        # loop over apertures
        for irad,r in enumerate(rad) :
            #column names for sum and uncertainty
            if type(r) is int : fmt='{:d}'
            else : fmt='{:.1f}'
            name=('aper'+fmt).format(r)
            ename=('aper'+fmt+'err').format(r)

            # pixels within aperture
            area = np.pi*r**2

            if utils :
                tot = aptab['aperture_sum_{:d}'.format(irad)]
                unc = aptab['aperture_sum_err_{:d}'.format(irad)]

            else :
                # here include pixel only if center is within aperture (not so good)
                gd = np.where(dist2 < r**2)
                # sum counts, subtract sky
                tot =data[gd[0],gd[1]].sum()
                # uncertainty
                unc = np.sqrt(
                      (uncertainty_data[gd[0],gd[1]]**2).sum()+
                      sigsq*area)

            # subtract sky, load columns
            stars[istar][name] = tot - sky*area
            stars[istar][ename] = unc

            # instrumental magnitudes if requested
            if mag : 
                stars[istar][ename] = (
                    1.086*(stars[istar][ename]/stars[istar][name]) )
                try : stars[istar][name] = -2.5 * np.log10(stars[istar][name])
                except : stars[istar][name] = 99.999

            if tv is not None :
                tv.tvcirc(star['x'],star['y'],r,color='b')
        stars[istar]['sky'] = sky
        stars[istar]['skysig'] = skysig
           
    return stars

def get(file) :
    """ Read FITS table into internal photometry list """
    stars=Table.read(file)
    return stars

def save(file,stars) :
    """ Save internal photometry list to FITS table"""
    stars.write(file,overwrite=True)

def centroid(data,x,y,r,verbose=True) :
    """ Get centroid in input data around input position, with given radius
    """
    # create arrays of pixel numbers for centroiding
    ys=int(y-2*r)
    ye=int(y+2*r)
    xs=int(x-2*r)
    xe=int(x+2*r)
    tmpdata=data[ys:ye,xs:xe]
    pix = np.mgrid[0:tmpdata.shape[0],0:tmpdata.shape[1]]
    ypix = pix[0]+ys
    xpix = pix[1]+xs

    xold=0
    yold=0
    iter=0
    while iter<10 :
        dist2 = (xpix-round(x))**2 + (ypix-round(y))**2
        # get pixels to use for background, and get background
        gd = np.where((dist2 < r**2) & (dist2 > (r-1)**2))
        back = np.median(tmpdata[gd[0],gd[1]])
        # get the centroid
        gd = np.where(dist2 < r**2)
        norm=np.sum(tmpdata[gd[0],gd[1]]-back)
        x = np.sum((tmpdata[gd[0],gd[1]]-back)*xpix[gd[0],gd[1]]) / norm
        y = np.sum((tmpdata[gd[0],gd[1]]-back)*ypix[gd[0],gd[1]]) / norm
        if round(x) == xold and round(y) == yold : break
        xold = round(x)
        yold = round(y)
        if verbose: print(iter,x,y)
        iter+=1
    return x,y

def process_all(files,red,tab,bias=None,dark=None,flat=None,threads=8) :
    """ multi-threaded processing of files
    """

    pars=[]
    for file in files :
        pars.append((file,red,tab,bias,dark,flat))

    if threads == 0 :
        output=[]
        for par in pars :
            output.append(process_thread(par))
    else :
        pool = mp.Pool(threads)
        output = pool.map_async(process_thread, pars).get()
        pool.close()
        pool.join()

    all=[]
    for out in output :
        all=vstack([all,out])
    return all

def process_thread(pars) :

    file = pars[0] 
    red = pars[1] 
    tab = pars[2]
    bias = pars[3]
    dark = pars[4]
    flat = pars[5]

    rcent=15
    rad=[10,15,20]
    skyrad=[25,30]
    cards=['EXPTIME','FILTER','AIRMASS']

    return process(file,red,tab,bias=bias,dark=dark,flat=flat,rcent=rcent,rad=rad,skyrad=skyrad,cards=cards)

def process(file,red,tab,bias=None,dark=None,flat=None,disp=None,
            rcent=7,rad=[3,5,7],skyrad=[10,15],cards=['EXPTIME','FILTER','AIRMASS']):

    """ Process and do photometry on input file
    """

    # work in temporary directory
    try:
      with tempfile.TemporaryDirectory() as tempdir :

        cwd = os.getcwd()
        os.chdir(tempdir)

        # process file
        a=red.reduce(file,superdark=dark,superbias=bias,superflat=flat,solve=True)
        dateobs=Time(a.header['DATE-OBS'],format='fits')

        # get x,y positions from RA/DEC and load into photometry table
        x,y=a.wcs.wcs_world2pix(tab['RA'],tab['DEC'],0)
        phot=copy.copy(tab)
        phot['x']=x
        phot['y']=y

        # re-centroid stars
        if disp is not None :
            disp.tv(a)
            mark(disp,phot,exit=True,auto=False,color='r',new=True,rad=rcent)
            mark(disp,phot,exit=True,auto=True,color='g',rad=rcent)
        else :
            for star in phot :
                x,y = centroid(a.data,star['x'],star['y'],rcent)
                star['x'] = x
                star['y'] = y

        # do photometry 
        try : phot=photom(a,phot,rad=rad,skyrad=skyrad)
        except : pass
        phot.add_column(Column([file]*len(tab),name='FILE',dtype=str))
        for card in cards :
            phot[card] = [a.header[card]]*len(tab)
        pdb.set_trace()
        phot['MJD'] = [Time(a.header['DATE-OBS'],format='fits').mjd]*len(tab)
    except :
        print('Error in process')
        pdb.set_trace()
        phot=copy.copy(tab)

    os.chdir(cwd)
    return phot

