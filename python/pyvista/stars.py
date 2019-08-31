# routines to deal with stellar images

import copy
import numpy as np
import pdb
from astropy.io import fits
from astropy.table import Table
from astropy.table import Column
from astropy.nddata import support_nddata
from pyvista import mmm

# global table to share across functions
stars = Table(names=('x', 'y'), dtype=('f4', 'f4'))

def mark(tv,rad=3,new=False,color='m',auto=False):
    """ Interactive mark stars on TV, or recenter current list 

    Args : 
           tv  : TV instance from which user will mark stars
           radius= (int): radius to use for centroiding and for size of circles 
                     (default=3)
           new = (bool)  : if True, start new list, else append to current
           color= (char) : color for circles (default='m')
           auto=  (bool) : if True, recentroid from existing position
    """
    global stars

    # clear display and mark current star list( if not new)
    tv.tvclear()
    if new :
        stars.remove_rows(slice(0,len(stars)))
    else :
        if auto :
            # with auto option, recentroid
            for star in stars :
                x,y = centroid(tv.img,star['x'],star['y'],rad)
                stars['x'] = x
                stars['y'] = y
        # display stars
        for star in stars : tv.tvcirc(star['x'],star['y'],rad,color=color)

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
        stars[len(stars)-1]['x'] = x
        stars[len(stars)-1]['y'] = y
        tv.tvcirc(x,y,rad,color=color)
    return stars


@support_nddata
def photom(data,err=None,rad=[3],skyrad=None,tv=None) :
    """ Aperture photometry of input image with current star list
    """
    global stars

    if type(rad) is int : rad = [rad]
    # Add new columns to table, removing them first if they exist already
    for r in rad :
        if type(r) is not int : raise ValueError
        name='aper{:d}'.format(r)
        try : stars.remove_column(name)
        except: pass
        col = Column( np.empty(len(stars))*np.nan )
        stars.add_column(col,name=name)
        stars[name].info.format = '.1f'
    cnts=[]
    cntserr=[]

    pix = np.mgrid[0:data.shape[0],0:data.shape[1]]
    ypix = pix[0]
    xpix = pix[1]
    for istar in range(len(stars)) :
        star=stars[istar]
        dist2 = (xpix-star['x'])**2 + (ypix-star['y'])**2
        if skyrad is not None :
            gd = np.where((dist2 > skyrad[0]**2) & 
                          (dist2 < skyrad[1]**2) ) 
            sky,skysig,skyskew = mmm.mmm(data[gd[0],gd[1]].flatten())
            if tv is not None :
                tv.tvcirc(star['x'],star['y'],skyrad[0],color='g')
                tv.tvcirc(star['x'],star['y'],skyrad[1],color='g')
        else : sky =0
        for r in rad :
            name='aper{:d}'.format(r)
            gd = np.where(dist2 < r**2)
            tot =data[gd[0],gd[1]].sum()
            stars[istar][name] = tot - sky*np.pi*r**2
            if tv is not None :
                tv.tvcirc(star['x'],star['y'],r,color='b')
           
    return stars

def get(file) :
    """ Read FITS table into internal photometry list """
    global stars
    stars=Table.read(file)

def save(file) :
    """ Save internal photometry list to FITS table"""
    global stars
    stars.write(file,overwrite=True)

def centroid(data,x,y,r) :
    """ Get centroid in input data around input position, with given radius
    """
    # create arrays of pixel numbers for centroiding
    pix = np.mgrid[0:data.shape[0],0:data.shape[1]]
    ypix = pix[0]
    xpix = pix[1]

    xold=0
    yold=0
    iter=0
    while iter<10 :
        dist2 = (xpix-round(x))**2 + (ypix-round(y))**2
        gd = np.where(dist2 < r**2)
        norm=np.sum(data[gd[0],gd[1]])
        x = np.sum(data[gd[0],gd[1]]*xpix[gd[0],gd[1]]) / norm
        y = np.sum(data[gd[0],gd[1]]*ypix[gd[0],gd[1]]) / norm
        if round(x) == xold and round(y) == yold : break
        xold = round(x)
        yold = round(y)
        iter+=1
    return x,y
