# routines to deal with stellar images

import copy
import glob
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
from pyvista import mmm, tv, spectra
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture, CircularAnnulus,aperture_photometry
from photutils.aperture import ApertureStats
from photutils.detection import DAOStarFinder
from holtztools import plots,html
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astroquery.sdss import SDSS
from astropy import coordinates as coord
import astropy.units as u

@support_nddata
def find(data,fwhm=4,thresh=4000,sharp=[0.,1.],round=[-2.,2.]) :
    """ Star finding using DAOStarfinder

        Parameters
        ----------
        data   : array-like
                 2D data array
        fwhm   : optional, float
                 FWHM (pixels) for matching, default=4 
        thresh : optional, float
                 Threshold above background, default=4000
        sharp  : optional, [float,float]
                 Low and high bounds for sharpness, default=[0.,1.]
        round  : optional, [float,float]
                 Low and high bounds for roundness, default=[-2.,2]
    """
    daofind = DAOStarFinder(fwhm=fwhm,threshold=thresh,sharplo=sharp[0],sharphi=sharp[1],roundlo=round[0],roundhi=round[1])
    sources=daofind(data)
    sources.rename_column('xcentroid','x')
    sources.rename_column('ycentroid','y')
    return sources

@support_nddata
def automark(data,stars,rad=3,func='centroid',plot=None,dx=0,dy=0,verbose=False,background=True) :
    """ Recentroid existing star list on input data array
    """
    if func == 'centroid' : 
        center = centroid
    elif func == 'gfit' :
        center = gfit
    elif func == 'gfit2' :
        center = gfit2
    new=copy.deepcopy(stars)
    for i,star in enumerate(new) :
        try :
            x,y = center(data,star['x']+dx,star['y']+dy,rad,plot=plot,verbose=verbose,background=background)
            new[i]['x'] = x
            new[i]['y'] = y
        except :
            new[i]['x'] = np.nan
            new[i]['y'] = np.nan
    return new

def mark(tv,stars=None,rad=3,auto=False,color='m',new=False,exit=False,id=False,func='centroid'):
    """ Interactive mark stars on TV, or recenter current list 

    Args : 
           tv  : TV instance from which user will mark stars
           stars =   : existing star table
           auto=  (bool) : if True, recentroid from existing position
           radius= (int): radius to use for centroiding and for size of circles (default=3)
           color= (char) : color for circles (default='m')
    """

    if func == 'centroid' : 
        center = centroid
    elif func == 'gfit' :
        center = gfit
    elif func == 'gfit2' :
        center = gfit2

    # clear display and mark current star list( if not new)
    if new: tv.tvclear()
    try: dateobs=Time(tv.hdr['DATE-OBS'],format='fits')
    except: dateobs=None
    cards=['EXPTIME','FILTER','AIRMASS']
    types=['f4','S','f4']
    if stars is None :
        stars = Table(names=('id','x', 'y'), dtype=('i4','f4', 'f4'))
        stars['x'].info.format = '.2f'
        stars['y'].info.format = '.2f'
        if dateobs is not None :
            stars.add_column(Column([],name='MJD',dtype=('f8')))
            stars['MJD'].info.format = '.6f'
        for icard,card in enumerate(cards) :
            try: stars.add_column(Column([],name=card,dtype=(types[icard])))
            except: pass
        stars['AIRMASS'].info.format = '.3f'
    else :
        if auto :
            # with auto option, recentroid and update from current header
            try: 
              for icard,card in enumerate(cards) :
                try: stars[card] = tv.hdr[card]
                except KeyError: stars.add_column(0.,name=card)
              stars['AIRMASS'].info.format = '.3f'
              try: stars['MJD'] = tv.hdr['MJD']
              except KeyError: 
                stars.add_column(0.,name='MJD')
                stars['MJD'].info.format = '.6f'
            except: pass
            for star in stars :
                x,y = center(tv.img,star['x'],star['y'],rad)
                star['x'] = x
                star['y'] = y
                if dateobs is not None : star['MJD'] = dateobs.mjd
                for icard,card in enumerate(cards) :
                    try: star[card] = tv.hdr[card]
                    except: pass
        # display stars
        for star in stars : 
            tv.tvcirc(star['x'],star['y'],rad,color=color)
            if id : tv.tvtext(star['x'],star['y'],star['id'],color=color)
        if exit : return stars

    istar=len(stars)+1
    print('Hit c near desired star(s) to get centroid position\n'+
          '    i to use integer position of cursor\n'+
          '    n to get ID of nearest star\n'+
          '    q or e to quit')
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
        elif key == 'g' :
            # gaussian fit to marginal distribution around marked position
            x,y= gfit2(tv.img,x,y,rad,plot=tv)
            print(x,y)
        elif key == 'n' :
            j=np.argmin((x-stars['x'])**2+(y-stars['y'])**2)
            print(j)
            print('Star: {:d} at ({:f},{:f})'.format(j,stars['x'][j],stars['y'][j]))
            continue

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

def sdss_label(t,im,label='psfmag_g',maxmag=19, rad=0.25,xoff=0,yoff=-10) :
   """ inital stab for getting SDSS coords and labelling image
   """
   pos = coord.SkyCoord('{:s} {:s}'.format(
          im.header['RA'].replace(' ',':'),im.header['DEC'].replace(' ',':')),
          unit=(u.hour,u.degree))
   sdss = SDSS.query_region(pos,radius=rad*u.degree,
             photoobj_fields=['ra','dec','psfmag_g','psfmag_r','psfmag_i','type'])
   x,y = im.wcs.wcs_world2pix(sdss['ra'],sdss['dec'],0)
   sdss['x'] = x+xoff
   sdss['y'] = y+yoff
   sdss['id'] = sdss[label].astype('|S4')
   t.tv(im)
   gd = np.where((sdss[label] < maxmag) & (sdss[label]>0) & (sdss['type'] == 6) )[0]

   mark(t,sdss[gd],rad=0,id=True,color='b')

@support_nddata
def add_coord(data,stars,wcs=None) :

    if wcs is not None :
        ra,dec=wcs.wcs_pix2world(stars['x'],stars['y'],0)
        stars['RA'] = ra
        stars['DEC'] = dec

@support_nddata
def photom(data,stars,uncertainty=None,rad=[3],skyrad=None,display=None,
           gain=1,rn=0,mag=True,utils=True) :
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
        uncertainty_data = np.sqrt(data/gain + rn**2/gain**2)
        
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
    try : stars.remove_column('peak')
    except: pass
    stars.add_column(emptycol,name='peak')
    stars['peak'].info.format = '.1f'
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
            if display is not None :
                display.tvcirc(star['x'],star['y'],skyrad[0],color='g')
                display.tvcirc(star['x'],star['y'],skyrad[1],color='g')
        else : 
            sky =0.
            skysig= 0.
            sigsq =0.

        # photutils aperture photometry handles pixels on the edges
        apertures = [ CircularAperture((star['x'],star['y']),r) for r in rad ]
        aptab = aperture_photometry(data,apertures,error=uncertainty_data)
        # run ApertureStats on first aperture to get peak
        apstats = ApertureStats(data,apertures[0],error=uncertainty_data)

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

            if display is not None :
                display.tvcirc(star['x'],star['y'],r,color='b')
        stars[istar]['sky'] = sky
        stars[istar]['skysig'] = skysig
        stars[istar]['peak'] = apstats.max
           
    return stars

def get(file) :
    """ Read FITS table into internal photometry list """
    stars=Table.read(file)
    return stars

def save(file,stars) :
    """ Save internal photometry list to FITS table"""
    stars.write(file,overwrite=True)

def centroid(data,x,y,r,verbose=False,plot=None,background=True) :
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
        if background :
            gd = np.where((dist2 > r**2) & (dist2 < (r+1)**2))
            back = np.nanmedian(tmpdata[gd[0],gd[1]])
        else :
            back = 0.
        # get the centroid
        gd = np.where(dist2 < r**2)
        norm=np.sum(tmpdata[gd[0],gd[1]]-back)
        if verbose: print(iter,x,y,back,norm)
        x = np.sum((tmpdata[gd[0],gd[1]]-back)*xpix[gd[0],gd[1]]) / norm
        y = np.sum((tmpdata[gd[0],gd[1]]-back)*ypix[gd[0],gd[1]]) / norm

        if round(x) == xold and round(y) == yold : break
        xold = round(x)
        yold = round(y)
        if verbose: print(iter,x,y)
        iter+=1
    if iter > 9 : print('possible centroiding convergence issues, consider using a larger radius?')
    return x,y

def gfit(data,x,y,rad,verbose=False,background=True,plot=False) :
    """ Gaussian fit to marginal distribution
    """
    xold=0
    yold=0
    iter=0
    while iter<10 :
        x0=int(x)
        y0=int(y)
        coeff = spectra.gfit(data[y0-rad:y0+rad+1,x0-2*rad:x0+2*rad+1].sum(axis=0),rad*2,sig=rad/2.,rad=rad,back=background)
        x = coeff[1]+x0-2*rad
        coeff = spectra.gfit(data[y0-2*rad:y0+2*rad+1,x0-rad:x0+rad+1].sum(axis=1),rad*2,sig=rad/2.,rad=rad,back=background)
        y = coeff[1]+y0-2*rad
        if round(x) == xold and round(y) == yold : break
        xold = round(x)
        yold = round(y)
        iter += 1
        if verbose: print(iter,x,y)

    if iter > 9 : print('possible centroiding convergence issues, consider using a larger radius?')

    return x, y

    xx = np.arange(x0-rad,x0+rad+1)
    yy = data[x0-rad:x0+rad+1]
    peak=yy.argmax()+x0-rad
    xx = np.arange(peak-rad,peak+rad+1)
    yy = data[peak-rad:peak+rad+1]
    if back == None : p0 = [data[peak],peak,sig]
    else : p0 = [data[peak],peak,sig,back]

    coeff, var_matrix = curve_fit(gauss, xx, yy, p0=p0)
    fit = gauss(xx,*coeff)

def gauss3(x,*p) :
    if len(p) == 10 : A, mu, sigma, B, Bmu, Bsigma, C, Cmu, Csigma, back = p
    elif len(p) == 7 : 
        A, mu, sigma, B, Bmu, Bsigma, back = p
        C, Cmu, Csigma = 0., 0., 1.
    elif len(p) == 4 : 
        A, mu, sigma, back = p
        B, Bmu, Bsigma = 0., 0., 1.
        C, Cmu, Csigma = 0., 0., 1.
    return (A*np.exp(-(x-mu)**2/(2.*sigma**2))+
            B*np.exp(-(x-Bmu)**2/2.*Bsigma**2)+
            C*np.exp(-(x-Cmu)**2/2.*Csigma**2)+
            back)

    
def gfit2(data,x,y,rad,verbose=False,plot=None,background=True) :
    """ Gaussian fit to marginal distribution
    """
    xold=0
    yold=0
    iter=0
    while iter<1 :
        x0=int(x)
        y0=int(y)
        xdata=data[y0-rad:y0+rad+1,x0-2*rad:x0+2*rad+1].sum(axis=0)
        back=xdata[0]
        peak=xdata.argmax()
        xx=np.arange(4*rad+1)
        p0=[xdata[peak]-back,peak,1.,
           (xdata[peak]-back)/10.,peak-3,2.,
           (xdata[peak]-back)/10.,peak+3,2.,back]
        ok = True
        ngx=3
        try: 
            xcoeff, var_matrix = curve_fit(gauss3, xx, xdata, p0=p0)
            j=np.argmax(xcoeff[0:9:3]/np.sqrt(np.abs(xcoeff[2:11:3])))
            x = xcoeff[j*3+1]+x0-2*rad
        except: ok = False
        if not ok or np.abs(xcoeff[1]-xcoeff[4]) < 1 :
            p0=[xdata[peak]-back,peak,1.,
               (xdata[peak]-back)/10.,peak-3,2.,
               back]
            ok = True
            ngx=2
            try: 
                xcoeff, var_matrix = curve_fit(gauss3, xx, xdata, p0=p0)
                j=np.argmax(xcoeff[0:6:3]/np.sqrt(np.abs(xcoeff[2:8:3])))
                x = xcoeff[j*3+1]+x0-2*rad
            except: ok= False
            if not ok or np.abs(xcoeff[1]-xcoeff[4]) < 1 :
                p0=[xdata[peak]-back,peak,1., back]
                ngx=1
                try: xcoeff, var_matrix = curve_fit(gauss3, xx, xdata, p0=p0)
                except: 
                    xcoeff=p0
                    ngx=0
                x = xcoeff[1]+x0-2*rad

        ydata=data[y0-2*rad:y0+2*rad+1,x0-rad:x0+rad+1].sum(axis=1)
        peak=ydata.argmax()
        xx=np.arange(4*rad+1)
        back=ydata[0]
        peak=ydata.argmax()
        p0=[ydata[peak]-back,peak,1.,
            (ydata[peak]-back)/3.,peak-5,2.,
            (ydata[peak]-back)/3.,peak+5,2.,back]
        ok = True
        ngy=3
        try: 
            ycoeff, var_matrix = curve_fit(gauss3, xx, ydata, p0=p0)
            j=np.argmax(ycoeff[0:9:3]/np.sqrt(np.abs(ycoeff[2:11:3])))
            y = ycoeff[j*3+1]+y0-2*rad
        except: ok = False
        if not ok or np.abs(ycoeff[1]-ycoeff[4]) < 1 :
            p0=[ydata[peak]-back,peak,1.,
               (ydata[peak]-back)/10.,peak-3,2.,
               back]
            ok = True
            ngy=2
            try: 
                ycoeff, var_matrix = curve_fit(gauss3, xx, ydata, p0=p0)
                j=np.argmax(ycoeff[0:6:3]/np.sqrt(np.abs(ycoeff[2:8:3])))
                y = ycoeff[j*3+1]+y0-2*rad
            except: ok= False
            if not ok or np.abs(ycoeff[1]-ycoeff[4]) < 1 :
                ngy=1
                p0=[ydata[peak]-back,peak,1., back]
                try: ycoeff, var_matrix = curve_fit(gauss3, xx, ydata, p0=p0)
                except: 
                    ycoeff=p0
                    ngy=0
                y = ycoeff[1]+y0-2*rad

        if round(x) == xold and round(y) == yold : break
        xold = round(x)
        yold = round(y)
        if verbose: print(iter,x,y)
        iter+=1

    print(ngx,ngy)
    if plot is not None :
        xx=np.arange(4*rad+1)
        plot.plotax1.plot(xx,data[y0-rad:y0+rad+1,x0-2*rad:x0+2*rad+1].sum(axis=0))
        plot.plotax1.plot(xx,gauss3(xx,*xcoeff))
        plot.plotax1.cla()
        for i in range(0,9,3) :
            try: plot.plotax1.plot([xcoeff[i+1],xcoeff[i+1]],[0,xcoeff[i]])
            except: pass
        print(xcoeff)

        xx=np.arange(4*rad+1)
        plot.plotax2.cla()
        plot.plotax2.plot(xx,data[y0-2*rad:y0+2*rad+1,x0-rad:x0+rad+1].sum(axis=1))
        plot.plotax2.plot(xx,gauss3(xx,*ycoeff))
        for i in range(0,9,3) :
            try: plot.plotax2.plot([ycoeff[i+1],ycoeff[i+1]],[0,ycoeff[i]])
            except: pass
        print(ycoeff)
        plt.show()

    return x, y

def process_all(files,red,tab,bias=None,dark=None,flat=None,threads=8, display=None, solve=True,
            seeing=15,rad=[3,5,7],skyrad=[10,15],cards=['EXPTIME','FILTER','AIRMASS']):
    """ multi-threaded processing of files
    """

    pars=[]
    for file in files :
        pars.append((file,red,tab,bias,dark,flat,solve,seeing,rad,skyrad,cards))

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
    solve = pars[6]
    seeing= pars[7]
    rad= pars[8]
    skyrad= pars[9]
    cards= pars[10]

    return process(file,red,tab,bias=bias,dark=dark,flat=flat,
                   rad=rad,skyrad=skyrad,seeing=seeing,cards=cards)

def process(file,red,tab,bias=None,dark=None,flat=None,display=None, solve=True,
            seeing=15,rad=[3,5,7],skyrad=[10,15],cards=['EXPTIME','FILTER','AIRMASS']):

    """ Process and do photometry on input file
    """

    # work in temporary directory
    cwd = os.getcwd()
    try:
      with tempfile.TemporaryDirectory(dir='./') as tempdir :

        os.chdir(tempdir)

        # process file
        a=red.reduce(file,dark=dark,bias=bias,flat=flat,solve=solve,
                     seeing=seeing,display=display)
        dateobs=Time(a.header['DATE-OBS'],format='fits')

        # get x,y positions from RA/DEC and load into photometry table
        x,y=a.wcs.wcs_world2pix(tab['RA'],tab['DEC'],0)
        phot=copy.copy(tab)
        phot['x']=x
        phot['y']=y

        # re-centroid stars
        if display is not None :
            display.tv(a)
            mark(display,phot,exit=True,auto=False,color='r',new=True,
                 rad=seeing/red.scale)
            mark(display,phot,exit=True,auto=True,color='g',rad=seeing/red.scale)
        else :
            for star in phot :
                x,y = centroid(a.data,star['x'],star['y'],seeing/red.scale)
                star['x'] = x
                star['y'] = y

        # do photometry 
        try : phot=photom(a,phot,rad=rad,skyrad=skyrad,display=display)
        except : 
            print('Error with photom')
        phot.add_column(Column([file]*len(tab),name='FILE',dtype=str))
        for card in cards :
            phot[card] = [a.header[card]]*len(tab)
        phot['MJD'] = [Time(a.header['DATE-OBS'],format='fits').mjd]*len(tab)
    #except :
    #    print('Error in process')
    #    pdb.set_trace()
    #    phot=copy.copy(tab)
        os.chdir(cwd)
    except OSError : 
        print('OSError')

    return phot

def dostar(red,obj,date,filts=['SR+D25'],seeing=12,dark=None,flats=[None],solve=True,
           rad=np.arange(5,45,5), skyrad=[50,60],clobber=False,threads=32,display=None) :

    if dark is None : print('No dark frame')
    try : tab = Table.read(obj+'.fits')
    except : tab=None


    grid=[]
    for filt,flat in zip(filts,flats) :
        if flat is None : print('No flat frame')
    
        files= glob.glob(red.dir+'/*'+obj+'*'+filt+'*')
        files.sort()
        if tab == None :
            print('no existing star table ....')
            out = red.reduce(files[0],dark=dark,flat=flat,solve=solve,seeing=seeing)
            t=tv.TV()
            t.tv(out)
            print('mark desired stars...')
            tab=mark(t,rad=seeing/red.scale)
            add_coord(out,tab)
            tab.write(obj+'.fits')
      
        sav='{:s}.{:s}.{:s}'.format(obj,date,filt) 
        if not os.path.exists(sav+'.fits') or clobber :
            out = process_all(files,red,tab,flat=flat,dark=dark,seeing=seeing,solve=solve,
                              rad=rad,skyrad=skyrad,threads=threads,display=display)
            out.write(sav+'.fits',overwrite=True)
        else :
            out=Table.read(sav+'.fits')

        diffphot(out,title=sav, hard=sav)
        grid.append([sav+'_mjd.png',sav+'_air.png'])
    html.htmltab(grid,file=obj+'.'+date+'.html')

def diffphot(tab,aper='aper35.0',yr=0.1,title=None,hard=None) :
    """ Make differential photometry plots
           including airmass detrending
    """
    nstars = len(set(tab['id']))
    nmjd = len(set(tab['MJD']))
    dat = np.zeros([nmjd,nstars])
    daterr = np.zeros([nmjd,nstars])
    x = np.zeros([nmjd])
    air = np.zeros([nmjd])

    # two plots, one vs MJD and one vs airmass
    fig,ax=plots.multi(nstars,nstars,figsize=(14,8),hspace=0.001,wspace=0.5)
    airfig,airax=plots.multi(nstars,nstars,figsize=(14,8),hspace=0.001,wspace=0.5)

    # loop over all MJDs
    for i,mjd in enumerate(sorted(set(tab['MJD']))) :
        # load up x, air, dat, and daterr arrays
        x[i]=mjd
        for j in range(nstars):
            ii=np.where((tab['MJD'] == mjd) & (tab['id'] == j+1) ) [0]
            dat[i,j] =  tab[aper][ii]
            daterr[i,j] =  tab[aper+'err'][ii]
            air[i] = tab['AIRMASS'][ii]

    # total the comp stars
    #afig,aax = plots.multi(1,1)
    #tot = -2.5*np.log10((10**(-0.4*dat[:,1:])).sum(axis=1))
    #aax.scatter(x,dat[:,0]-tot)
    #aax.set_ylim(aax.get_ylim()[1],aax.get_ylim()[0])
        
    
    # make plots of all pairs of stars
    for j in range(nstars) :
        for k in range(j,nstars) :
            diff=dat[:,j]-dat[:,k]
            err = np.sqrt(daterr[:,j]**2+daterr[:,k]**2)
            med=np.median(diff)
            std=np.std(diff)
            mad=np.median(np.abs(diff-med))

            # airmass detrending fit
            fit=np.polyfit(air,diff,1)
            diff_fit=diff-fit[0]*(air-np.median(air))
            med_fit=np.median(diff_fit)
            std_fit=np.std(diff_fit)
            mad_fit=np.median(np.abs(diff_fit-med_fit))

            # plots
            ax[j,k].scatter(x,diff,edgecolors='g',facecolors='none')
            ax[j,k].errorbar(x,diff,yerr=err,fmt='none',color='g')
            ax[j,k].scatter(x,diff_fit,color='b')
            ax[j,k].errorbar(x,diff_fit,yerr=err,fmt='none',color='b')
            ax[j,k].set_xlabel('MJD')
            ax[j,k].scatter(x,dat[:,j]-np.median(dat[:,j])+np.median(diff)+0.01,edgecolors='r',facecolors='none')
            ax[j,k].scatter(x,dat[:,k]-np.median(dat[:,k])+np.median(diff)-0.01,edgecolors='m',facecolors='none')
            ax[j,k].set_ylim(med+yr,med-yr)
            bd =np.where(diff>(med+yr))[0]
            ax[j,k].scatter(x[bd],len(bd)*[med+yr-.01*yr],marker=6,color='r')
            bd =np.where(diff<(med-yr))[0]
            ax[j,k].scatter(x[bd],len(bd)*[med-yr+.01*yr],marker=7,color='r')
            ax[j,k].text(0.,0.9,'std: {:.3f}  MAD: {:.3f}'.format(std,mad),
                         transform=ax[j,k].transAxes,color='g')
            ax[j,k].text(0.,0.1,'std: {:.3f}  MAD: {:.3f}'.format(
                      std_fit,mad_fit),transform=ax[j,k].transAxes,color='b')

            airax[j,k].scatter(air,diff,color='g')
            airax[j,k].errorbar(air,diff,yerr=err,fmt='none',color='g')
            airax[j,k].scatter(air,diff_fit,color='b')
            airax[j,k].errorbar(air,diff_fit,yerr=err,fmt='none',color='b')
            airax[j,k].scatter(air,dat[:,j]-np.median(dat[:,j])+np.median(diff)+0.01,edgecolors='r',facecolors='none')
            airax[j,k].scatter(air,dat[:,k]-np.median(dat[:,k])+np.median(diff)-0.01,edgecolors='m',facecolors='none')
            airax[j,k].set_ylim(med+yr,med-yr)
            bd =np.where(diff>(med+yr))[0]
            airax[j,k].scatter(air[bd],len(bd)*[med+yr-.01*yr],
                               marker=6,color='r')
            bd =np.where(diff<(med-yr))[0]
            airax[j,k].scatter(air[bd],len(bd)*[med-yr+.01*yr],
                               marker=7,color='r')
            airax[j,k].set_xlabel('Airmass')
            airax[j,k].text(0.,0.9,'std: {:.3f}  MAD: {:.3f}'.format(std,mad),
                            transform=airax[j,k].transAxes,color='g')
            airax[j,k].text(0.,0.1,'std: {:.3f}  MAD: {:.3f}'.format(
                    std_fit,mad_fit),transform=airax[j,k].transAxes,color='b')

    # remove unfilled plots
    for j in range(nstars) :
        for k in range(j) :
            ax[j,k].set_visible(False)
            airax[j,k].set_visible(False)

    if title != None : 
        fig.suptitle(title)
        airfig.suptitle(title)

    if hard != None :
        fig.savefig(hard+'_mjd.png')
        airfig.savefig(hard+'_air.png')

    #fig.tight_layout()
    #plt.draw()
    #airfig.tight_layout()
    #plt.draw()

    #pdb.set_trace()
    #plt.close()
    #plt.close()
    #plt.close()

    return x,dat
