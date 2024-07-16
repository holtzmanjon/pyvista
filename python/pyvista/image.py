from __future__ import print_function
import numpy as np
import copy
from holtztools import plots
from astropy import units as u
from astropy.io import fits, ascii
from astropy.modeling import models, fitting
from astropy.nddata import StdDevUncertainty, support_nddata
from pyvista import stars
import scipy.signal
import scipy.ndimage
from scipy.optimize import curve_fit
from skimage.transform import SimilarityTransform, AffineTransform
import matplotlib.pyplot as plt
import glob
import bz2
import os
import pdb

sig2fwhm = 2*np.sqrt(2*np.log(2))
#sig2fwhm=np.float64(2.354)

class BOX() :
    """ 
    Defines BOX class
    """
    def __init__(self,n=None,nr=None,nc=None,sr=1,sc=1,cr=None,cc=None,xr=None,yr=None) :
        """ Define a BOX

            Args :
               n (int) : size of box (if square)
               nr (int) : number of rows 
               nc (int) : number of cols
               sr (int) : start row
               sc (int) : start column
               cr (int) : central row (supercedes sr)
               cc (int) : central column (supercedes sc)
               xr       : [xmin,xmax]  (supercedes cc and sc)
               yr       : [ymin,ymax]  (supercedes cr and sr)
        """
        if nr is None and nc is None and n is None and xr is None and yr is None:
            print('You must specify either n=, or nr= and nc=')
            return
        elif nr is None and nc is None :
            nr=n
            nc=n
        elif nr is None :
            print('You much specify nr= with nc=')
        elif nc is None :
            print('You much specify nc= with nr=')

        if cr is not None and cc is not None :
            sr=cr-nr//2
            sc=cc-nr//2

        if xr is not None :
            self.xmin=xr[0]
            self.xmax=xr[1]
        else :
            self.xmin = sc
            self.xmax = sc+nc-1
        if yr is not None :
            self.ymin=yr[0]
            self.ymax=yr[1]
        else :
            self.ymin = sr
            self.ymax = sr+nr-1

    def set(self,xmin,xmax,ymin,ymax):
        """ Resets limits of a box
        
            Args:
                xmin : lower x value
                xmax : higher x value
                ymin : lower y value
                ymax : higher xyvalue
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def bin(self,binfactor) :
        self.xmin //= binfactor
        self.xmax //= binfactor
        self.ymin //= binfactor
        self.ymax //= binfactor

    def nrow(self):
        """ Returns number of rows in a box

            Returns :
                number of rows 
        """
        return(self.ymax-self.ymin+1)

    def ncol(self):
        """ Returns number of columns in a box

            Returns :
                number of columns
        """
        return(self.xmax-self.xmin+1)

    def show(self,header=True):
        """ Prints box limits
        """
        #if header : print('    SC    NC    SR    NR  Exp       Date     Name')
        if header : print('    SC    NC    SR    NR')
        print('{:6d}{:6d}{:6d}{:6d} '.format(
              self.xmin,self.ncol(),self.ymin,self.nrow()))

    def mean(self,data):
        """ Returns mean of data in box

            Args :
                data : input data (Data or np.array)
 
            Returns:
                mean of data in box
        """
        if self.nrow() <= 0 or self.ncol() <= 0 : return 0.
        return data[self.ymin:self.ymax+1,self.xmin:self.xmax+1].mean() 

    def stdev(self,data):
        """ Returns standard deviation of data in box

            Args :
                data : input data (Data or np.array)

            Returns:
                standard deviation of data in box
        """
        if self.nrow() == 0 or self.ncol() == 0 : return 0.
        return data[self.ymin:self.ymax+1,self.xmin:self.xmax+1].std() 

    def max(self,data):
        """ Returns maximum of data in box

            Args :
                data : input data (Data or np.array)

            Returns:
                maximum of data in box
        """
        if self.nrow() == 0 or self.ncol() == 0 : return 0.
        return data[self.ymin:self.ymax+1,self.xmin:self.xmax+1].max() 

    def min(self,data):
        """ Returns minimum of data in box

            Args :
                data : input data (Data or np.array)

            Returns:
                minimum of data in box
        """
        if self.nrow() == 0 or self.ncol() == 0 : return 0.
        return data[self.ymin:self.ymax+1,self.xmin:self.xmax+1].min() 

    def median(self,data):
        """ Returns median of data in box

            Args :
                data : input data (Data or np.array)

            Returns:
                median of data in box
        """
        if self.nrow() == 0 or self.ncol() == 0 : return 0.
        return np.median(data[self.ymin:self.ymax+1,self.xmin:self.xmax+1])

    def setval(self,data,val):
        """ Sets data in box to specified value
        """
        if self.nrow() == 0 or self.ncol() == 0 : return 0.
        data[self.ymin:self.ymax+1,self.xmin:self.xmax+1] = val

    def setbit(self,data,val):
        """ Sets bit of data in box to specified value
        """
        if self.nrow() == 0 or self.ncol() == 0 : return 0.
        data[self.ymin:self.ymax+1,self.xmin:self.xmax+1] |= val

    def getval(self,data):
        """ Returns data in box

            Args :
                data : input data (Data or np.array)
 
            Returns:
                data in box
        """
        if self.nrow() <= 0 or self.ncol() <= 0 : return 0.
        return data[self.ymin:self.ymax+1,self.xmin:self.xmax+1]


@support_nddata
def abx(data,box) :
    """
    Returns dictionary with image statistics in box.

    Args :
        data  : input data (Data or np.array)
        box   : pyvista BOX

    Returns :
        dictionary with image statistics : 'mean', 'stdev', 'min', 'max', 'peakx', 'peaky'
    """
    return {'mean': box.mean(data),
            'stdev': box.stdev(data),
            'max': box.max(data),
            'min': box.min(data),
            'peakx': np.unravel_index(
                        data[box.ymin:box.ymax,box.xmin:box.xmax].argmax(),
                        (box.nrow(),box.ncol()) )[1]+box.xmin,
            'peaky': np.unravel_index(
                        data[box.ymin:box.ymax,box.xmin:box.xmax].argmax(),
                        (box.nrow(),box.ncol()) )[0]+box.ymin}

def gauss2d_binned(X, amp, x0, y0, a, b, c, back) :
    return gauss2d(X, amp, x0, y0, a, b, c, back, binned=True) 


def gauss2d(X, amp, x0, y0, a, b, c, back, binned=False) :
    """ Evaluate Gaussian 2D function  

        Form: amp*exp(-a(x-x0)**2 - b(x-x0)*(y-y0) - c(y-y0)**2) + const

        Parameters
        ----------
        X : arraylike [2,npts]
            x,y positions to evaluate at
        amp, x0, y0, a, b, c, back : float
            coefficients of function
    """
    x = X[0]
    y = X[1]

    if binned :
        pdb.set_trace()
        # use 10x10 sub-bins
        yy,xx=np.mgrid[y.min()-0.5:y.max()+0.5:0.1,x.min()-0.5:x.max()+0.5:0.1]
        out= (amp/100.*(np.exp(-a*(xx-x0)**2-b*(xx-x0)*(yy-y0)-c*(yy-y0)**2))).reshape(x.shape[0],10,y.shape[0],10).sum(3).sum(1)+back
    else :
        out= amp*(np.exp(-a*(x-x0)**2-b*(x-x0)*(y-y0)-c*(y-y0)**2))+back

    return out.flatten()

def gh2d_wrapper(X, N, *args) :
    gpars = args[0][:5]
    hpars = np.array(args[0][5:5+N**2]).reshape(N,N)
    back = args[0][-1]
    return gh2d(X, gpars, hpars, back)

def gh2d(X, gpars, hpars, back, binned=False) :
    """ Evaluate Gauss-Hermite 2D function  

        Form: exp(-a(x-x0)**2 - b(x-x0)*(y-y0) - c(y-y0)**2) + const

        Parameters
        ----------
        X : arraylike [2,npts]
            x,y positions to evaluate at
        x0, y0, a, b, c, back : float
            coefficients of function
    """
    x = X[0]
    y = X[1]

    x0, y0, a, b, c = gpars
    if len(hpars) > 0 :
        #p = np.polynomial.hermite.hermval2d((x-x0),(y-y0),hpars)
        nherm = len(hpars)
        p=0
        for iy in range(nherm) :
            for ix in range(nherm) :
                p+=hpars[iy,ix]*scipy.special.eval_hermitenorm(ix,x-x0)*scipy.special.eval_hermitenorm(iy,y-y0)
    else :
        p = 1
    if binned :
        # use 10x10 sub-bins
        yy,xx=np.mgrid[y.min()-0.5:y.max()+0.5:0.1,x.min()-0.5:x.max()+0.5:0.1]
        out= (p/100.*(np.exp(-a*(xx-x0)**2-b*(xx-x0)*(yy-y0)-c*(yy-y0)**2))).reshape(x.shape[0],10,y.shape[0],10).sum(3).sum(1)+back
    else :
        out= p*(np.exp(-a*(x-x0)**2-b*(x-x0)*(y-y0)-c*(y-y0)**2))+back

    return out.flatten()

def abc2fwxfwytheta(a,b,c) :
    """ Convert a,b,c, from gauss2d to xfwhm, yfwhm, theta
    """
    theta=0.5*np.arctan(-b/(a-c))
    xfwhm=np.sqrt(1/(2*a*np.cos(theta)**2-2*b*np.cos(theta)*np.sin(theta)+2*c*np.sin(theta)**2))*sig2fwhm
    yfwhm=np.sqrt(1/(2*a*np.sin(theta)**2+2*b*np.cos(theta)*np.sin(theta)+2*c*np.cos(theta)**2))*sig2fwhm

    return xfwhm, yfwhm, theta

def fwxfwytheta2abc(xfwhm,yfwhm,theta) :
    """ Convert xfwhm, yfwhm, theta to a,b,c for gauss2d
    """

    sigx = xfwhm / sig2fwhm
    sigy = yfwhm / sig2fwhm
    a = np.cos(theta)**2/(2*sigx**2) + np.sin(theta)**2/(2*sigy**2)
    b = -np.sin(2*theta)/(2*sigx**2) + np.sin(2*theta)/(2*sigy**2)
    c = np.sin(theta)**2/(2*sigx**2) + np.cos(theta)**2/(2*sigy**2)

    return a,b,c

def gfit2d(data,x0,y0,size=5,fwhm=3.,sub=True,plot=None,fig=1,scale=1,pafixed=False,astropy=True,binned=False) :
    """ 
    Does gaussian fit to input data given initial xcen,ycen
    """

    # use initial guess to get peak
    z=data[int(y0)-size:int(y0)+size+1,int(x0)-size:int(x0)+size+1]
    # refine subarray around peak
    ycen,xcen=np.unravel_index(np.argmax(z),z.shape)
    xcen+=(int(x0)-size)
    ycen+=(int(y0)-size)

    # set up input data and fit
    y,x=np.mgrid[ycen-size:ycen+size+1,xcen-size:xcen+size+1]
    z=data[ycen-size:ycen+size+1,xcen-size:xcen+size+1]

    if astropy :
        g_init=models.Gaussian2D(x_mean=xcen,y_mean=ycen,
                             x_stddev=fwhm/2.3548,y_stddev=fwhm/2.3548,
                             amplitude=data[ycen,xcen],theta=0.,
                             fixed={'theta':pafixed})+models.Const2D(0.)
        fit=fitting.LevMarLSQFitter()
        g=fit(g_init,x,y,z)
        xfwhm=g[0].x_stddev*sig2fwhm*scale
        yfwhm=g[0].y_stddev*sig2fwhm*scale
        fwhm=np.sqrt(xfwhm*yfwhm)
        theta=(g[0].theta.value % (2*np.pi)) * 180./np.pi
        print('xFWHM:{:8.2f}   yFWHM:{:8.2f}   FWHM:{:8.2f}  SCALE:{:8.2f}  PA:{:8.2f}'.format(xfwhm,yfwhm,fwhm,scale,theta))
        if plot is not None:
            xc=g[0].x_mean.value
            yc=g[0].y_mean.value
            r = np.sqrt((y-yc)**2 + (x-xc)**2)
            plots.plotp(plot,r,z,xt='R(pixels)',yt='Intensity')
            r = np.arange(0.,5*fwhm/sig2fwhm/scale,0.1)
            peak=g[0].amplitude
            plot.plot(r,peak*np.exp(-np.power(r, 2.) / (2 * np.power(g[0].x_stddev, 2.)))+g[1].amplitude)
            plot.plot(r,peak*np.exp(-np.power(r, 2.) / (2 * np.power(g[0].y_stddev, 2.)))+g[1].amplitude)
            plot.text(0.9,0.9,'x: {:7.1f} y: {:7.1f} fw: {:8.2f}'.format(xc,yc,fwhm),transform=plot.transAxes,ha='right')
            plt.draw()
        if sub :
            data[ycen-size:ycen+size+1,xcen-size:xcen+size+1]-=g[0](x,y)
       
        return g

    else :
        p0=np.array([data[ycen,xcen],xcen,ycen,1./(2*(fwhm/sig2fwhm)**2),0.,1./(2*(fwhm/sig2fwhm)**2),0.])
        if binned :
            g=curve_fit(gauss2d_binned,np.array([x,y]),z.flatten(), p0=p0)
        else :
            g=curve_fit(gauss2d,np.array([x,y]),z.flatten(), p0=p0)

        # translate parameters to xfwhm,yfwhm,theta
        amp,x0,y0,a,b,c,back=g[0]
        theta=0.5*np.arctan(b/(a-c))
        xfwhm=np.sqrt(1/(2*a*np.cos(theta)**2+2*b*np.cos(theta)*np.sin(theta)+2*c*np.sin(theta)**2))*sig2fwhm*scale
        yfwhm=np.sqrt(1/(2*a*np.sin(theta)**2-2*b*np.cos(theta)*np.sin(theta)+2*c*np.cos(theta)**2))*sig2fwhm*scale
        fwhm=np.sqrt(xfwhm*yfwhm)
        print('xFWHM:{:8.2f}   yFWHM:{:8.2f}   FWHM:{:8.2f}  SCALE:{:8.2f}  PA:{:8.2f}'.format(
               xfwhm,yfwhm,np.sqrt(xfwhm*yfwhm),scale,(theta%(2*np.pi))*180/np.pi))
        if plot is not None:
            xc=x0
            yc=y0
            xsig = xfwhm/2.355
            ysig = yfwhm/2.355
            r = np.sqrt((y-yc)**2 + (x-xc)**2)
            plots.plotp(plot,r,z,xt='R(pixels)',yt='Intensity')
            r = np.arange(0.,5*fwhm/sig2fwhm/scale,0.1)
            plot.plot(r,amp*np.exp(-np.power(r, 2.) / (2 * np.power(xsig, 2.)))+back)
            plot.plot(r,amp*np.exp(-np.power(r, 2.) / (2 * np.power(ysig, 2.)))+back)
            plot.text(0.9,0.9,'x: {:7.1f} y: {:7.1f} fw: {:8.2f}'.format(xc,yc,fwhm),transform=plot.transAxes,ha='right')
            plt.draw()
        if sub :
            data[ycen-size:ycen+size+1,xcen-size:xcen+size+1]-=gauss2d(np.array([x,y]),*g[0]).reshape(-2*size+1,2*size+1)
        return np.array([amp,x0,y0,xfwhm,yfwhm,theta,back])

def ghfit2d_thread(pars) :
    """ 
    Wrapper for ghfit2d for multithreading, with just a single argument
    """
    data,x0,y0,size,binned,nherm =  pars
    return ghfit2d(data,x0,y0,size=size,binned=binned,nherm=nherm)

def ghfit2d(data,x0,y0,size=5,fwhm=3.,nherm=6,sub=True,plot=None,fig=1,scale=1,pafixed=False,binned=False,
            p0=None, bounds=None) :
    """ 
    Does 2D Gauss-Hermite fit to input data given initial xcen,ycen
    """

    # use initial guess to get peak
    z=data[int(y0)-size:int(y0)+size+1,int(x0)-size:int(x0)+size+1]
    # refine subarray around peak
    ycen,xcen=np.unravel_index(np.argmax(z),z.shape)
    xcen+=(int(x0)-size)
    ycen+=(int(y0)-size)

    # set up input data and fit
    y,x=np.mgrid[ycen-size:ycen+size+1,xcen-size:xcen+size+1]
    z=data[ycen-size:ycen+size+1,xcen-size:xcen+size+1]

    if p0 is None :
        p0=np.array([xcen,ycen,1./(2*(fwhm/sig2fwhm)**2),0.3,1./(2*(fwhm/sig2fwhm)**2)]+[data[ycen,xcen]]+(nherm*nherm-1)*[0.]+[0.])
    if bounds is None :
        bounds=([0.,0.,0.,0.,0.,0.]+(nherm*nherm-1)*[-np.inf]+[-np.inf],
                [2048.,2048.,np.inf,np.inf,np.inf,np.inf]+(nherm*nherm-1)*[np.inf]+[np.inf])
    X = np.array([x.flatten(),y.flatten()])
    if binned :
        g=curve_fit(gh2d_binned,X,z.flatten(), p0=p0)
    else :
        try : g,cov,info,mesg,ier=curve_fit(lambda X, *params : gh2d_wrapper(X, nherm, params),X,z.flatten(), p0=p0, bounds=bounds,full_output=True)
        except : 
            g=p0
            cov=None
            info={}

    # translate parameters to xfwhm,yfwhm,theta
    x0,y0,a,b,c=g[:5]
    theta=0.5*np.arctan(b/(a-c))
    xfwhm=np.sqrt(1/(2*a*np.cos(theta)**2+2*b*np.cos(theta)*np.sin(theta)+2*c*np.sin(theta)**2))*sig2fwhm
    yfwhm=np.sqrt(1/(2*a*np.sin(theta)**2-2*b*np.cos(theta)*np.sin(theta)+2*c*np.cos(theta)**2))*sig2fwhm
    print('xFWHM:{:8.2f}   yFWHM:{:8.2f}   FWHM:{:8.2f}  SCALE:{:8.2f}  PA:{:8.2f}'.format(
           xfwhm,yfwhm,np.sqrt(xfwhm*yfwhm),scale,(theta%(2*np.pi))*180/np.pi))
    if sub :
        try :data[ycen-size:ycen+size+1,xcen-size:xcen+size+1]-=gh2d_wrapper(np.array([x,y]),nherm,g).reshape(2*size+1,2*size+1)
        except : pass
    return np.array(g),cov,info

def fit2d(X,Y,Z) :
    """ Fit 2D quadratic surface
    """
    gd=np.where(np.isfinite(Z))
    #A = np.array([X[gd]*0+1, X[gd], Y[gd], X[gd]**2, X[gd]**2*Y[gd], X[gd]**2*Y[gd]**2, Y[gd]**2, X[gd]*Y[gd]**2, X[gd]*Y[gd]]).T
    A = np.array([X[gd]*0+1, X[gd], Y[gd], X[gd]**2, X[gd]*Y[gd], Y[gd]**2]).T
    coeff, r, rank, s = np.linalg.lstsq(A, Z[gd])
    return coeff

def mk2d(X,Y,coeff) :
    """ Return 2D surface given input points and coefficients
    """
    #A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    A = np.array([X*0+1, X, Y, X**2, X*Y, Y**2]).T
    return np.dot(A,coeff)

@support_nddata
def window(data,box,header=None) :
    """
    Reduce size of image and header accordingly
    """
    if header is None :
        return data[box.ymin:box.ymax+1,box.xmin:box.xmax+1]
    else :
        header['CRVAL1'] = box.xmin
        header['CRVAL2'] = box.ymin
        header['NAXIS1'] = box.ncol()
        header['NAXIS2'] = box.nrow()
        return fits.PrimaryHDU(data[box.ymin:box.ymax+1,box.xmin:box.xmax+1],header)

def stretch(a,ncol=None,nrow=None) :
    """ 
    Stretches a 1D image into a 2D image along rows or columns 
    """
    if nrow is None and ncol is None :
        print('Must specify either nrow= or ncol=')
        return
    if nrow is not None and ncol is not None :
        print('Must specify only one of nrow= or ncol=')
        return
    if ncol is not None :
        out=np.zeros([a.shape[0],ncol])
        for i in range(ncol) :
            out[:,i]=a
    if nrow is not None :
        out=np.zeros([nrow,a.shape[0]])
        for i in range(nrow) :
            out[i,:]=a
    return out

def __get_cnpix(a) :
    """
    Gets CNPIX cards from HDU a, sets them to 1 if they don't exist
    """
    try:
        cnpix1=a.header['CNPIX1']
    except:
        a.header['CNPIX1']=1
    try:
        cnpix2=a.header['CNPIX2']
    except:
        a.header['CNPIX2']=1

    return a.header['CNPIX1'],a.header['CNPIX2']

def __get_overlap(a,b,dc=0,dr=0,box=None) :
    """
    Returns overlap coordinates from two input HDUs
    """
    
    if box is not None :
        print('need to implement box=!')
        return
 
    a_cnpix1,a_cnpix2 = __get_cnpix(a)
    b_cnpix1,b_cnpix2 = __get_cnpix(b)

    ixmin = max(a_cnpix1,b_cnpix1+dc)
    iymin = max(a_cnpix2,b_cnpix2+dr)
    ixmax = min(a_cnpix1+a.header['NAXIS1'],b_cnpix1+dc+b.header['NAXIS1'])
    iymax = min(a_cnpix2+a.header['NAXIS2'],b_cnpix2+dr+b.header['NAXIS2'])

    return (iymin-a_cnpix2,iymax-a_cnpix2,ixmin-a_cnpix1,ixmax-a_cnpix1,
           iymin-b_cnpix2-dr,iymax-b_cnpix2-dr,ixmin-b_cnpix1-dc,ixmax-b_cnpix1-dc)

def __check_hdu(a) :
    """
    Checks if input variable is an HDU
    """

    if type(a) is fits.hdu.image.PrimaryHDU :
        return True
    else:
        print('Input must be HDU type, with header and data!')
        return False

def add(a,b,dc=0,dr=0,box=None) :
    """ 
    Adds b to a, paying attention to CNPIX 
    """
    if __check_hdu(a) is False or __check_hdu(b) is False : return
    ay1,ay2,ax1,ax2,by1,by2,bx1,bx2 = __get_overlap(a,b,dr=dr,dc=dc,box=box)
    a.data[ay1:ay2,ax1:ax2] += b.data[by1:by2,bx1:bx2]

def sub(a,b,dc=0,dr=0,box=None) :
    """ 
    Subracts b from a, paying attention to CNPIX 
    """
    if __check_hdu(a) is False or __check_hdu(b) is False : return
    ay1,ay2,ax1,ax2,by1,by2,bx1,bx2 = __get_overlap(a,b,dr=dr,dc=dc,box=box)
    a.data[ay1:ay2,ax1:ax2] -= b.data[by1:by2,bx1:bx2]

def mul(a,b,dc=0,dr=0,box=None) :
    """ 
    Multiplies b by a, paying attention to CNPIX 
    """
    if __check_hdu(a) is False or __check_hdu(b) is False : return
    ay1,ay2,ax1,ax2,by1,by2,bx1,bx2 = __get_overlap(a,b,dr=dr,dc=dc,box=box)
    a.data[ay1:ay2,ax1:ax2] *= b.data[by1:by2,bx1:bx2]

def div(a,b,dc=0,dr=0,box=None) :
    """ 
    Divides a by b, paying attention to CNPIX 
    """
    if __check_hdu(a) is False or __check_hdu(b) is False : return
    ay1,ay2,ax1,ax2,by1,by2,bx1,bx2 = __get_overlap(a,b,dr=dr,dc=dc,box=box)
    a.data[ay1:ay2,ax1:ax2] /= b.data[by1:by2,bx1:bx2]

def clip(hd,min=None,max=None,vmin=None,vmax=None,box=None) :
    """
    Clipping tasks: sets all values above or below input values to specified values

    Args:
         hd : input HDU

    Keyword args:
         min=  (float) : clip values below min
         vmin= (float) : values to clip min values to. If min= is not given clips values <vmin to vmin
         max=  (float) : clip values above max
         vmax= (float) : values to clip max values to. If max= is not given clips values >vmax to vmax
    """
    if __check_hdu(hd) is False : return
   
    if box is not None :
        print('need to implement box=!')
        return
 
    if min is not None or vmin is not None :
        if vmin is None: 
            clipval=0
        else :
            clipval=vmin
        if min is None:
            min=vmin
        iy,ix=np.where(hd.data > min)
        hd.data[iy,ix]=clipval

    if max is not None or vmax is not None :
        if vmax is None: 
            clipval=0
        else :
            clipval=vmax
        if max is None:
            max=vmax
        iy,ix=np.where(hd.data > max)
        hd.data[iy,ix]=clipval

def buf(hd) :
    """
    Display information about HDU
    """ 
    if __check_hdu(hd) is False : return

    print('    SC    NC    SR    NR  Exp       Date     Name')
    cnpix1,cnpix2 = __get_cnpix(hd)
    npix1 = hd.header['NAXIS1']
    npix2 = hd.header['NAXIS2']
    print('{:6d}{:6d}{:6d}{:6d}'.format(cnpix1,npix1,cnpix2,npix2))

    #dict=globals()
    #for key in dict :
    #    if type(dict[key]) is fits.hdu.image.PrimaryHDU : print(key)


def rd(file,ext=0) :
    """
    Read file into HDU
    """
    try:
        return fits.open(file)[ext]
    except :
        print('cannot open file: ', file, ' extension: ', ext)

def create(box=None,n=None,nr=None,nc=None,sr=1,sc=1,cr=None,cc=None,const=None) :
    """
    Creates a new HDU
    """
    if box is not None:
        nr=box.nrow()
        nc=box.ncol()
        sc=box.xmin
        sr=box.ymin
    else :
        if nr is None and nc is None :
            try :
                nr=n
                nc=n
            except:
                print('You must specify either box=, n=, or nr= and nc=')
                return
        if cr is not None and cc is not None :
            sr=cr-nr/2
            sc=cc-nr/2
    try :
        im=np.zeros([nr,nc])
    except :
        print('must specify image size ')
        return
    hd=fits.PrimaryHDU(im)
    hd.header['CNPIX1'] = sc
    hd.header['CNPIX2'] = sr
    if const is not None :
        hd.data += const
    return hd

def sky(im,box=None,max=None,min=None,plot=None):
    """
    Estimate sky value in an image by fitting parabola to peak of histogram

    Args:
        im (HDU or numpy array): input image data 

    Keyword args:
        box=   : only use values within specified box (default=None)
        min=   : ignore values below min in sky computation (default=None)
        max=   : ignore values above max in sky computation (default=None)
        plot=  : matplotlib axes to view histogram and fit (default=None)
    """

    if type(im) is fits.hdu.image.PrimaryHDU :
        data = im.data
    else :
        data = im
    if box is not None :
        reg = data[box.ymin:box.ymax+1,box.xmin:box.xmax+1]
    else :
        reg = data

    if min is None: min = reg.min()
    if max is None: max = reg.max()
    if min > max :
        raise ValueError("min must be less than max")

    gd = np.where((reg >min) & (reg<max))
    if len(gd[0]) < 1 :
        raise ValueError("no pixels between min and max")

    # get median and stdev in desired region
    med = np.median(reg[gd])
    sig = reg[gd].std()
    print('initial median, sigma: ', med, sig)

    # create histogram around median and find peak
    gd = np.where((reg.flatten() > med-2*sig) & (reg.flatten() < med+2*sig))[0]
    hist,bins = np.histogram(reg.flatten()[gd],bins=np.arange(med-2*sig,med+2*sig))
    max = np.max(hist)
    imax = np.argmax(hist)

    # find half power points on either side of peak
    i1=imax
    while hist[i1] > max/2. and i1 > 0 :
        i1-=1
    i2=imax
    while hist[i2] > max/2. and i2 < len(hist) :
        i2+=1

    # fit parabola to peak, and determine location of fit max
    binwidth=bins[1]-bins[0]
    p_init=models.Polynomial1D(degree=2)
    fit=fitting.LinearLSQFitter()
    p=fit(p_init,bins[i1:i2+1]+binwidth,hist[i1:i2+1])
    sky=-p.parameters[1]/(2.*p.parameters[2])
    if plot is not None:
        plot.plot(bins[i1:i2+1]+binwidth,hist[i1:i2+1])
        plot.plot(bins[i1:i2+1]+binwidth,p(bins[i1:i2+1]+binwidth))
        plt.draw()

    return sky

def getdata(hd) :

    if isinstance(hd, (np.ndarray)) :
        data=hd
    elif isinstance(hd, (astropy.io.fits.hdu.hdulist.HDUList)) :
        data=hd[0].data
    elif isinstance(hd, (astropy.io.fits.hdu.image.PrimaryHDU)) :
        data=hd.data
    else :
        print('Unrecognized data type: ',type(hd))
    return(data)

def xcorr(a,b,lags,medfilt=0,rad=3) :
    """ Cross correlation function between two arrays, calculated at lags

        If input images have the same number of rows, then calculate a single
          cross-correlation in columns
        If first image has one row, but the second has more, then calculate 
          a cross correlation for each row of the second images

        Arguments:
            a : array_like
                reference array
            b : array_like
                array to calculate shifts for
            lags : array_like
                x-corrlation lags to use
            medfilt : int, default=0
                size of median filter for arrays 

        Returns :
            fit peak of cross-correlation (quadratic fit)
            1D cross-correlation function
    """

    # compute xcorr with starting and ending position to allow full range of lags
    xs = -lags[0]
    xs = np.max([0,xs])
    xe = b.shape[-1]-lags[-1]
    xe = np.min([xe,a.shape[-1]])
    #xe = np.min([a.shape[-1],b.shape[-1]])-lags[-1]
    atmp=np.atleast_2d(a)
    btmp=np.atleast_2d(b)

    # with medfilt parameter, subtract median filtered array from data
    if medfilt>0 :
        atmp=np.atleast_2d(atmp-scipy.signal.medfilt(a,kernel_size=[1,medfilt]))
        btmp=np.atleast_2d(btmp-scipy.signal.medfilt(b,kernel_size=[1,medfilt]))

    if atmp.shape[0] == btmp.shape[0] :
        # single cross-correlation
        shift=np.zeros([1,len(lags)])
        for i,lag in enumerate(lags) :
            shift[0,i]=np.sum(atmp[:,xs:xe]*btmp[:,xs+lag:xe+lag])
    elif atmp.shape[0] == 1 :
        # cross-correlation for each row
        shift=np.zeros([btmp.shape[0],len(lags)])
        for row in range(btmp.shape[0]) :
            print('cross correlating row: {:d}'.format(row),end='\r')
            for i,lag in enumerate(lags) :
                shift[row,i]=np.sum(atmp[0,xs:xe]*btmp[row,xs+lag:xe+lag])
    else:
        raise ValueError('input arrays must have same nrows, or first must have 1 row')
        return

    # fit the cross-correlation function to get the peak
    fitpeak=np.zeros(shift.shape[0])
    for row in range(shift.shape[0]) :
        peak=shift[row,:].argmax()
        try :
            fit=np.polyfit(range(-rad,rad+1),shift[row,peak-rad:peak+rad+1],2)
            fitpeak[row]=peak+-fit[1]/(2*fit[0])
        except TypeError :
            print('xcorr peak fit failed, row: ', row,' using peak')
            fitpeak[row]=peak

    return fitpeak,np.squeeze(np.array(shift))

def xcorr2d(a,b,lags=None,xlags=None,ylags=None) :
    """ Two-dimensional cross correlation

        Args:
            a, b : input Data frames
            lags : array (1D) of x-corrlation lags

        Returns:
            (x,y) position of cross correlation peak from quadratic fit to x-correlation
            2D cross correlation function
    """
    # do x-corrlation over section of image that fits within input lag array
    if xlags is None : xlags = lags
    if ylags is None : ylags = lags
    xs = -xlags[0]
    xe = np.min([a.shape[1],b.shape[1]])-xlags[-1]
    ys = -ylags[0]
    ye = np.min([a.shape[0],b.shape[0]])-ylags[-1]

    # compute x-correlation
    shift = np.zeros([len(ylags),len(xlags)])
    for i, xlag in enumerate(xlags) :
        for j, ylag in enumerate(ylags) :
            shift[j,i] = np.sum(a.data[ys:ye,xs:xe]*b.data[ys+ylag:ye+ylag,xs+xlag:xe+xlag])

    #y,x=np.meshgrid(ylags,xlags)
    x,y=np.meshgrid(xlags,ylags)
    yp,xp=np.unravel_index(shift.argmax(),shift.shape)
    print('yp, xp:',yp,xp,len(xlags),len(ylags))
    if xp == 0 or yp == 0 or xp > len(xlags)-2 or yp > len(ylags)-2 :
        # peak at edge of cross correlation
        peak= (xp,yp)
    else :
        # quadratic fit and determine peak
        fit=fitting.LinearLSQFitter()
        mod=models.Polynomial2D(degree=2)
        p=fit(mod,x[yp-1:yp+2,xp-1:xp+2],y[yp-1:yp+2,xp-1:xp+2],shift[yp-1:yp+2,xp-1:xp+2])
        a = np.array([ [2*p.parameters[2], p.parameters[5]], [p.parameters[5],2*p.parameters[4]] ])
        b = np.array([-p.parameters[1],-p.parameters[3]])
        peak=np.linalg.solve(a,b)+(xp,yp)+(xlags[0],ylags[0])

    return peak,shift


def zap(hd,size,nsig=3,mask=False) : 
    """ Median filter array and replace values > nsig*uncertainty
    """
    filt=scipy.signal.medfilt(hd.data,size)
    if nsig >= 0 : bd = np.where(np.atleast_2d(hd.data)-filt > nsig*hd.uncertainty.array)
    else : bd = np.where(np.atleast_2d(hd.data)-filt < nsig*hd.uncertainty.array)
    np.atleast_2d(hd.data)[bd[0],bd[1]] = np.atleast_2d(filt)[bd[0],bd[1]]
    if mask :
        if hd.mask is None : hd.mask=np.zeros(hd.data.shape,dtype=bool)
        hd.mask[bd[0],bd[1]] = True

@support_nddata
def smooth(data,size,uncertainty=None,bitmask=None) :
    """ Boxcar smooth image
    """
    npix=1
    for dim in size : npix*=dim
    data=scipy.ndimage.uniform_filter(data*npix,size=size)
    if uncertainty is not None :
        uncertainty=StdDevUncertainty(np.sqrt(scipy.ndimage.uniform_filter(uncertainty.array**2,size=size)/npix))

    return data


@support_nddata
def minmax(data,mask=None, low=3,high=10):
    """ Return min,max scaling factors for input data using median, and MAD
   
        Args:
            img : input CCDData
            low : number of MADs below median to return
            high : number of MADs above median to retunr

        Returns:
            min,max : low and high scaling factors
    """
    if mask is not None :
        gd = np.where(np.isfinite(data) & ~mask)
    else :
        gd = np.where(np.isfinite(data))
    std=np.median(np.abs(data[gd]-np.median(data[gd])))
    min = np.median(data[gd])-low*std
    max = np.median(data[gd])+high*std
    return min,max


def transform(im0,im,lines0,xlags=range(-11,12),ylags=range(-17,18),
              rad=2,scale=20,hard=None,reject=1) :
    """ Get geometric transformation between images based on point sources

    Parameters
    ----------
        im0 : Data or array-like
              Reference image
        im : Data or array-like
              Target image
        lines0 : Table
              Table with reference object positions ('x' and 'y')
        xlags, ylags : range
              Range of cross-correlation shifts to try, default is range(-11,12) and (-17,18)
        rad : float, default=2
              radius for automark
        scale : integer, default=20
              scale for quiver plots
        hard : None or char
              if char, make tranformation plots ('' to display, otherwise save to specified file name)
    """

    nr,nc=im.data.shape

    # 2D cross correlation
    peak,shift=xcorr2d(im0,im,xlags=xlags,ylags=ylags)

    # smooth cross correlation by by 3x3 kernel in case there are multiple
    # peaks and the wrong one happens to match pixel centering better
    # Just use integer peak
    kernel=np.ones([3,3])
    indices=np.unravel_index(
              scipy.signal.convolve(shift,kernel,mode='same').argmax(),shift.shape)
    dy=indices[0]+ylags[0]
    dx=indices[1]+xlags[0]
    print('xcorr shifts: ',dx,dy)
    print('automarking...',len(lines0))
    lines=stars.automark(im.data,lines0,rad=rad,dx=dx,dy=dy,
                         background=False,func='marginal_gfit')

    dx=np.nanmean(lines['x']-lines0['x'])
    dy=np.nanmean(lines['y']-lines0['y'])
    print('average shifts:',dx,dy)

    print('fitting...')
    lin=AffineTransform()
    rot=SimilarityTransform()
    gd = np.where((np.isfinite(lines0['x']))&(np.isfinite(lines['x'])))[0]
    src=np.array([lines0['x'][gd]-nc//2,lines0['y'][gd]-nr//2]).T
    dest=np.array([lines['x'][gd]-nc//2,lines['y'][gd]-nr//2]).T
    lin.estimate(src,dest)
    res=lin(src)-dest
    #rot.estimate(src,dest)
    #res=rot(src)-dest
    # reject points with >1 pixel residual
    gd=np.where((np.abs(res[:,0])<reject)&(np.abs(res[:,1])<reject))[0]
    bd=np.where((np.abs(res[:,0])>reject)|(np.abs(res[:,1])>reject))[0]
    print(len(gd),len(res))
    rot.estimate(src[gd],dest[gd])
    lin.estimate(src[gd],dest[gd])

    if hard is not None :
        fig,ax=plots.multi(4,1,figsize=(24,6),wspace=0.001)
        ax[0].quiver(src[gd,0]+nc//2,src[gd,1]+nr//2,
                     dest[gd,0]-src[gd,0],dest[gd,1]-src[gd,1],
                     scale=scale,width=0.005)
        ax[0].quiver(src[bd,0]+nc//2,src[bd,1]+nr//2,
                     dest[bd,0]-src[bd,0]-dx,dest[bd,1]-src[bd,1]-dy,
                     scale=scale,width=0.005,color='r')
        ax[1].quiver(src[gd,0]+nc//2,src[gd,1]+nr//2,
                     dest[gd,0]-src[gd,0]-dx,dest[gd,1]-src[gd,1]-dy,
                     scale=scale,width=0.005)
        ax[1].quiver(src[bd,0]+nc//2,src[bd,1]+nr//2,
                     dest[bd,0]-src[bd,0]-dx,dest[bd,1]-src[bd,1]-dy,
                     scale=scale,width=0.005,color='r')
        ax[1].set_title('dx: {:.2f} dy: {:.2f}'.format(dx,dy))
        res=rot(src)-dest
        ax[2].quiver(src[gd,0]+nc//2,src[gd,1]+nr//2,res[gd,0],res[gd,1],
                     scale=scale,width=0.005)
        ax[2].quiver(src[bd,0]+nc//2,src[bd,1]+nr//2,res[bd,0],res[bd,1],
                     scale=scale,width=0.005,color='r')
        plots.plotc(ax[2],src[gd,0]+nc//2,src[gd,1]+nr//2,
                    np.sqrt(res[gd,0]**2+res[gd,1]**2),
                    size=10,zr=[0,0.5],cmap='viridis')
        ax[2].set_title('sc: {:.6} rot: {:.2f} dx: {:.2f} dy: {:.2f} res: {:.3f}'.format(
                    rot.scale,rot.rotation*180/np.pi,*rot.translation,res.std()))
        res=lin(src)-dest
        ax[3].quiver(src[gd,0]+nc//2,src[gd,1]+nr//2,res[gd,0],res[gd,1],
                     scale=scale,width=0.005)
        ax[3].quiver(src[bd,0]+nc//2,src[bd,1]+nr//2,res[bd,0],res[bd,1],
                     scale=scale,width=0.005,color='r')
        cbar=plots.plotc(ax[3],src[gd,0]+nc//2,src[gd,1]+nr//2,
                     np.sqrt(res[gd,0]**2+res[gd,1]**2),
                     size=10,zr=[0,0.5],cmap='viridis')
        ax[3].set_title('Full affine: {:.3f}'.format(res.std()))
        for i in range(4) :
            ax[i].set_xlim(0,nc)
            ax[i].set_ylim(0,nr)
            ax[i].quiver(nc//2,250,1,0,color='g',scale=scale,width=0.005)

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        fig.colorbar(cbar, cax=cbar_ax)

        if hard != '' :
            fig.savefig(hard)
            plt.close()
        else :
            plt.draw()
            plt.show()
            pdb.set_trace()
        plt.close()

    return lin,rot
