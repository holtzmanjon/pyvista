import copy
import pdb
import numpy as np
from pyvista import simulate, tv, stars
from skimage.transform import SimilarityTransform, EuclideanTransform
from sklearn.cluster import KMeans, MeanShift
from astropy.table import Table

scale=0.258

def sim(n=5, rad=4, sky=500, back=1000., xr=[500,1500],yr=[1000,3000], counts=[5000,20000],
             rot=0.05, dx=1, dy=-1) :
    """ Simulate a slitmask image for KOSMOS

    Parameters
    ----------
    n : int, optional, default=5
        Number of slitmask holes
    rad : float, optional, default=4
        radius of holes, in arcsec
    sky: float, optional, default=500
        amplitude of image in the holes
    back: float, optional, default=1000
        background amplitude
    xr : list, optional, default[500,1500]
        x range for hole locations
    yr : list, optional, default[1000,3000]
        y range for hole locations
    counts : list, optional, default=[5000,20000]
        range for counts of artificial stars
    rot : float, optional, default=0.05
        simulated rotation in degrees
    dx : float, optional, default=1.
        simulated x offset
    dy : float, optional, default=-1.
        simulated y offset

    Returns
    -------
    a, b : array
           Data arrays of mask and star images
    """

    a=np.zeros([4096,2148],dtype=np.float)
    x=np.random.uniform(xr[0],xr[1],size=n)
    y=np.random.uniform(yr[0],yr[1],size=n)
    amp=np.random.uniform(counts[0],counts[1],size=n)
    ypix,xpix=np.mgrid[0:4096,0:2148]

    radpix=4/scale

    for xx,yy in zip(x,y) :
       gd = np.where((np.abs(xpix-xx) <= radpix)&(np.abs(ypix-yy) <= radpix) )
       a[gd] = sky

    a=np.random.poisson(a).astype(np.float)
    a+=np.random.normal(back,10,size=a.shape)

    rotrad=rot*np.pi/180.
    dxpix=dx/scale
    dypix=dy/scale

    x0=1024
    y0=2048
    xp=(x-x0)*np.cos(rotrad)-(y-y0)*np.sin(rotrad)+dxpix+x0
    yp=(x-x0)*np.sin(rotrad)+(y-y0)*np.cos(rotrad)+dypix+y0

    b=copy.deepcopy(a)
    b=simulate.gauss2d(b,np.array([amp,xp,yp]).T,fwhm=3)

    print('simulated {:d} holes, rotation: {:f} dx: {:f} dy: {:f}'.format(n,rot,dx,dy))

    return a, b

def findholes(a,thresh=1250,n=5, bandwidth=20) :
    """ Find centers of slitmask holes from flat-field image

    Parameters
    ----------
    data : array
           Flat-field image with slitmask holes illuminated
    thresh : float, optional, default=1250
           Value above which pixel is flagged to be in the hole
    n : int, optional,default=5
         number of clusters to find, should be equal to number of holes

    Returns
    -------
    Table of hole centers

    """

    c=KMeans(n_clusters=n)
    gdpix=np.where(a > thresh)
    X=np.vstack([gdpix[0],gdpix[1]]).T
    c.fit(X)
    tab=Table()
    tab['y'] = c.cluster_centers_[:,0]
    tab['x'] = c.cluster_centers_[:,1]

    c=MeanShift(bandwidth=bandwidth,bin_seeding=True,min_bin_freq=50)
    c.fit(X)
    tab=Table()
    tab['y'] = c.cluster_centers_[:,0]
    tab['x'] = c.cluster_centers_[:,1]

    return tab

def findstars(b,thresh=500,sharp=[0.4,2]) :
    """ Find locations of stars on image

    Parameters
    ----------
    data : array
           image with stars in slitmask holes 
    thresh : float, optional, default=500
           Threshold for finding star
    sharp : list, optional, default=[0.4,2])
           [min,max] sharpness values to accept as star

    Returns
    -------
    Table of star locations
    """

    out=stars.find(b,thresh=thresh)
    gd=np.where((out['sharpness'] > sharp[0]) & (out['sharpness'] < sharp[1]) )[0]

    return out[gd]

def fit(holes, locstars, ordered=False) :
    """ Given tables of hole and star locations, find offset and rotation

        Parameters
        ----------
        holes : astropy Table
                table of hole positions
        stars : astropy Table
                table of star positions
        ordered : bool, default=False
                if False, match holes to stars by closest distance.
                if True, stars and holes should match by table order
                  (and tables should have the same number of entries)

    """

    src=[]
    dest=[]
    x0=1024
    y0=2048
    if ordered :
        for hole,star in zip(holes,locstars) :
           src.append([hole['x']-x0,hole['y']-y0])
           dest.append([star['x']-x0,star['y']-y0])
        
    else :
        for hole in holes :
           d2 = (locstars['x']-hole['x'])**2 + (locstars['y']-hole['y'])**2
           j= np.argmin(d2)
           src.append([hole['x']-x0,hole['y']-y0])
           dest.append([locstars[j]['x']-x0,locstars[j]['y']-y0])
    src=np.array(src)
    dest=np.array(dest)

    trans=EuclideanTransform()
    trans.estimate(src,dest)
    print('Rotation (degrees) : ',trans.rotation*180/np.pi)
    print('Translation (arcsec) ', trans.translation*scale)
    for s,d in zip(src,dest) :
      fit=trans(s)[0]
      print(s,fit,d,np.sqrt((fit[0]-d[0])**2 +(fit[1]-d[1])**2))

    sim=SimilarityTransform()
    sim.estimate(src,dest)
    print('Rotation (degrees) : ',sim.rotation*180/np.pi)
    print('Translation (arcsec) ', sim.translation*scale)
    print('Scale ', sim.scale)
    for s,d in zip(src,dest) :
      fit=sim(s)[0]
      print(s,fit,d,np.sqrt((fit[0]-d[0])**2 +(fit[1]-d[1])**2))

    return trans.rotation*180/np.pi,trans.translation*scale

def test(n=5,rot=0.05,dx=1,dy=-1,display=None) :
    """ Run a test of slitmask routines.  Simulates image and tries to recover rotation and offset

    Parameters
    ----------
    n : int, optional, default=5
        Number of slitmask holes
    rot : float, optional, default=0.05
        simulated rotation in degrees
    dx : float, optional, default=1.
        simulated x offset
    dy : float, optional, default=-1.
        simulated y offset
    display : pyvista TV object, optional, default=None
        if specified, display image in specified object

    """
    a,b=sim(n=n,rot=rot,dx=dx,dy=dy)
    holes=findholes(a,n=n)
    print('found {:d} holes'.format(len(holes)))
    locstars=findstars(b)
    print('found {:d} stars'.format(len(locstars)))
    fit(holes,locstars)

    if display is not None :
        display.tv(b)
        stars.mark(display,holes,color='r',exit=True)
        stars.mark(display,locstars,color='g',exit=True)


