# routines to deal with getting positions of stellar images

import matplotlib.pyplot as plt
import numpy as np
import pdb
from astropy.nddata import support_nddata
from pyvista import spectra, mmm

from collections import namedtuple

Center = namedtuple('Center', ['x', 'y', 'tot', 'meanprof','varprof'])

@support_nddata
def centroid(data,x,y,r,verbose=False,plot=None,background=True) :
    """ Get centroid in input data around input position, with given radius

        Parameters
        ----------
        data : array-like
               Input data
        x,y  : float
               Initial position guess
        r    : float
               Radius to use for centroid  
        background : bool, option
               Subtract background value from perimeter first, default=True
        plot  : not used

        Returns
        -------
        Center namedtuple 
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
            gd = np.where((dist2 > (2*r)**2) & (dist2 < (2*r+1)**2))
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
    center = Center(x,y,norm,None,None)
    return center

@support_nddata
def peak(data,x,y,rad) :
    """ Return location of peak in input data

        Parameters
        ----------
        data : array-like
               Input data
        x,y  : float
               Initial position guess
        rad  : float
               radius to search in

        Returns
        -------
        Center namedtuple 
    """

    sky,skysig,skyskew,nsky = mmm.mmm(data.flatten())
    ys=int(y-rad)
    ye=int(y+rad)+1
    xs=int(x-rad)
    xe=int(x+rad)+1
    yp,xp=np.unravel_index(np.argmax(data[ys:ye,xs:xe]-sky),data[ys:ye,xs:xe].shape)
    center=Center(xp+xs,yp+ys,np.max(data-sky),None,None)
    return center


@support_nddata
def marginal_gfit(data,x,y,rad,verbose=False,background=True,plot=False) :
    """ Get position from Gaussian fit to marginal distribution

        Parameters
        ----------
        data : array-like
               Input data
        x,y  : float
               Initial position guess
        rad  : float
               radius to search in
        background : bool, option
               Subtract background value from perimeter first, default=True
        plot  : not used
       

        Returns
        -------
        Center namedtuple 
    """
    xold=0
    yold=0
    iter=0
    while iter<10 :
        x0=int(x)
        y0=int(y)
        coeff = spectra.gfit(data[y0-rad:y0+rad+1,x0-2*rad:x0+2*rad+1].sum(axis=0),rad*2,sig=rad/2.,rad=rad,back=background)
        x = coeff[1]+x0-2*rad
        xtot = coeff[0]*np.sqrt(2*np.pi)*coeff[2]
        coeff = spectra.gfit(data[y0-2*rad:y0+2*rad+1,x0-rad:x0+rad+1].sum(axis=1),rad*2,sig=rad/2.,rad=rad,back=background)
        y = coeff[1]+y0-2*rad
        ytot = coeff[0]*np.sqrt(2*np.pi)*coeff[2]
        if round(x) == xold and round(y) == yold : break
        xold = round(x)
        yold = round(y)
        iter += 1

    if iter > 9 : print('possible centroiding convergence issues, consider using a larger radius?')

    center = Center(x,y,(xtot+ytot)/2.,None,None)
    return center

def gauss3(x,*p) :
    """ Evaluates 1D Gaussian function at input position(s)

        Parameters
        ----------
        x : float
            position(s) to evaluate Gaussian at
        p : array-like
            Gaussian parameters
            if len(p) == 10 : 3 gaussians + background
            if len(p) == 7 : 2 gaussians + background
            if len(p) == 4 : 1 gaussians + background
    """


    if len(p) == 10 : 
        A, mu, sigma, B, Bmu, Bsigma, C, Cmu, Csigma, back = p
        return (A*np.exp(-(x-mu)**2/(2.*sigma**2))+
                B*np.exp(-(x-Bmu)**2/2.*Bsigma**2)+
                C*np.exp(-(x-Cmu)**2/2.*Csigma**2)+
                back)
    elif len(p) == 7 : 
        A, mu, sigma, B, Bmu, Bsigma, back = p
        return (A*np.exp(-(x-mu)**2/(2.*sigma**2))+
                B*np.exp(-(x-Bmu)**2/2.*Bsigma**2)+
                back)
    elif len(p) == 4 : 
        A, mu, sigma, back = p
        return (A*np.exp(-(x-mu)**2/(2.*sigma**2))+
                back)
    
@support_nddata
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

    center = Center(x,y,None,None,None)
    return center

"""
Routines for calculating radial asymmetry centroid
   rasym_centroid()
   rprof()

Adapted from Russell Owen's PyGuide (e.g., https://github.com/ApachePointObservatory/PyGuide)
which in turn are adapted from implementation by Jim Gunn

asymm     measure of asymmetry:
              sum over radindex of var(radindex)^2 / weight(radindex)
          where weight is the expected sigma of var(rad) due to pixel noise:
              weight(radindex) = pixNoise(radindex) * sqrt(2(numPix(radindex) - 1))/numPix(radindex)
              pixNoise(radindex) = sqrt((readNoise/ccdGain)^2 + (meanVal(rad)-bias)/ccdGain)
"""

@support_nddata
def rasym_centroid(data,x0,y0,rad=25,weight=False,mask=None,verbose=False,skyrad=None,maxiter=10,plot=None) :
    """ Get centroid via calculation of minimum asymmetry

    Parameters
    ----------
    data : array-like
           Input data array
    x0, y0 : float
           Initial position guess
    rad : integer, default=25
           Maximum extent to calculate radial profile and sum asymmetry over
    """

    if skyrad is not None :
        pix = np.mgrid[0:data.shape[0],0:data.shape[1]]
        ypix = pix[0]
        xpix = pix[1]
        dist2 = (xpix-x0)**2 + (ypix-y0)**2
        gd = np.where((dist2 > skyrad[0]**2) & 
                      (dist2 < skyrad[1]**2) ) 
        sky,skysig,skyskew,nsky = mmm.mmm(data[gd[0],gd[1]].flatten())
        sigsq=skysig**2/nsky
    else :
        sky=0

    # we will iterate 3x3 calculation of minimum asymmetry until minimum is at central point
    iter = 0
    while True :
        if verbose : print('iter: ', iter)
        minasym=1.e100
        asym = np.zeros([3,3])
        if plot is not None : 
            plot.plotax2.cla()
            plot.plotax2.text(0.05,0.9,'Iter {:d}'.format(iter),transform=plot.plotax2.transAxes)
        for dy in range(-1,2) :
            for dx in range(-1,2) :
                prof=rprof(data-sky,round(x0)+dx,round(y0)+dy,rad=rad,weight=weight,inmask=mask,verbose=verbose)
                if plot is not None: plot.plotax2.plot(prof[3],label='{:d} {:d}'.format(dx,dy))
                asym[dy+1,dx+1]=prof[0]
                if verbose : print(round(x0)+dx,round(y0)+dy,asym)
                if asym[dy+1,dx+1]<minasym :
                    # if this is lowest asymmetry, save point
                    x1 = round(x0)+dx
                    y1 = round(y0)+dy
                    minasym=asym[dy+1,dx+1]
                    tot=prof[1]
                    minprof=prof[2]
                    minvar=prof[3]

        if plot : plot.plotax2.legend()
        # if central point hasn't changed, we are done
        if x1==round(x0) and y1==round(y0) : 
            # if central point hasn't changed, we are done
            break
        else : 
            # otherwise, update central point
            x0=x1
            y0=y1
            iter+=1
            if iter>maxiter: 
                print('exceeded {:d} iterations'.format(maxiter))
                center = Center(-1, -1,tot,minprof,minvar)
                return center

    # now do parabolic fit to get fractional centroid
    ai = 0.5 * (asym[2,1] - 2*asym[1,1] + asym[0,1])
    bi = 0.5 * (asym[2,1] - asym[0,1])
    aj = 0.5 * (asym[1,2] - 2*asym[1,1] + asym[1,0])
    bj = 0.5 * (asym[1,2] - asym[1,0])
    di = -0.5*bi/ai
    dj = -0.5*bj/aj
    if verbose : 
        print(iter,x1,y1)
        print(x1+dj,y1+di)

    center = Center(x1+dj,y1+di,tot,minprof,minvar)
    if plot is not None : 
        plot.tvclear()
        plot.tvcirc(center.x,center.y,rad)
        plot.tvcirc(center.x,center.y,2)
    return center


def rprof(indata,x0,y0,rad=25,weight=False,inmask=None,gain=1,rn=0,bias=0,verbose=False) :
    """ Calculate asymmetry profile and total asymmetry

    Parameters
    ----------
    data : array-like
           Input data array
    x0, y0 : integer
           Pixel position to calculate asymmetry around
    rad : integer
          Maximum radius
    """

    # subarray needed for speed!
    data = indata[int(y0-rad-2):int(y0+rad+2),int(x0-rad-2):int(x0+rad+2)]
    if inmask is not None :
        mask = inmask[int(y0-rad-2):int(y0+rad+2),int(x0-rad-2):int(x0+rad+2)]

    # calculate r**2 array of distances from input center
    y,x = np.mgrid[0:data.shape[0],0:data.shape[1]]
    #r2 = (x-int(x0))**2 + (y-int(y0))**2
    r2 = (x-(rad+2))**2 + (y-(rad+2))**2

    # determine the radius index array that we will use to
    #   determine the pixels that go into each "radius"

    # Algorithm (Mirage convention?) is
    #   radial index[rad**2] = 0, 1, 2, int(sqrt(rad**2)+1.5) for rad**2>2\n

    rind = np.zeros_like(r2).astype(int)
    j=np.where(r2==0)
    rind[j]=0
    j=np.where(r2==1)
    rind[j]=1
    j=np.where(r2==2)
    rind[j]=2
    for r in range(3,rad**2) :
        j=np.where(r2==r)
        rind[j]=int(np.sqrt(r)+1.5)

    if inmask is not None :
        j=np.where(mask)
        rind[j] = -1

    # Now loop over all radial indices, and determine mean and variance 
    #   over all pixels at each index. Sum the variances into asym
    mean=np.zeros(rad)
    var=np.zeros(rad)
    asym=0
    tot=0
    for r in range(1,rad) :
        j=np.where(rind==r)
        npix = len(j[0])
        if npix > 3 :
            m = np.nanmean(data[j])
            v=np.nanvar(data[j])
            mean[r]=m
            var[r]=v
            if weight :
                noise=np.sqrt((rn/gain)**2 + (np.max([m-bias,0.1]))/gain)
                w = noise*np.sqrt(2*(npix-1))/npix
                if verbose : print(r,npix,noise,w,v**2/w)
            else :
                w = 1
                if verbose : print(r,npix,v**2)
            asym+=v**2/w
            tot+=data[j].sum()

    # Return total asymmetry, and mean and variance profiles
    return asym, tot, mean, var
