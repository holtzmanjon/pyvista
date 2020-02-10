"""
Utilities for numpy structured arrays
"""

import numpy as np
import glob
import sys
import pdb
import copy
from astropy.io import fits

def pformat(file,val,iformat,fformat,sformat) :
    """ Utility routine for printing a value """
    #print('type: ', val, type(val))
    if isinstance(val,np.ndarray) :
        for v in val :
            #call pformat in case of Nd array
            pformat(file,v,iformat,fformat,sformat)
    elif isinstance(val,(float,np.float32)) :
        file.write(fformat.format(val))
    elif isinstance(val,(int,np.int32,np.int16)) :
        file.write(iformat.format(val))
    else :
        file.write(sformat.format(str(val)))

def list(s,cols=None, cond=None, ind=None, table=None, iformat='{:6d}',fformat='{:8.2f}', sformat='{:<12s}',file=None) :
    """
    List columns of requested row

    Args:
      s (numpy structured array)  : array to show

    Keyword args:
      cols : list of columns to list
      cond : tuple of (column, value); rows where column==value will be listed
      ind : list of index(es) to print
      table : use table format

    Returns:
    """
    if file is None :
       f = sys.stdout
    else :
       f = open(file,'w')

    # Use input columns if given, else all columns
    if cols is None :
        cols = s.names

    # Use condition if given, else specified index (default ind=0)
    if cond is not None :
        inds = np.where(s[cond[0]] == cond[1])[0]
    elif ind is not None :
        try :
            test=len(ind)
        except :
            ind=[ind]
        inds = np.array(ind)
    else :
        inds = np.arange([0])
   
    # if not specified, use table format for multiple entries
    if table is None :
        if len(inds) > 1 :
            table = True
        else :
            table = False

    # in table format, print column names 
    if table :
        for col in cols :
            f.write(sformat.format(col))
        f.write('\n')

    # print
    for i in inds :
        for col in cols :
            if not table :
                f.write(sformat.format(col))
            pformat(f,s[i][col],iformat,fformat,sformat)
            if not table : 
                f.write('\n')
        if table :
            f.write('\n')

def add_cols(a,b):
    """ 
    Add empty columns from recarray b to existing columns from a,
    return new recarray
    """

    # need to handle array elements properly
    newdtype = []
    names = a.dtype.names+b.dtype.names
    descrs = a.dtype.descr+b.dtype.descr
    for i in range(len(descrs)) :
        name= names[i]
        desc= descrs[i]
        print(name,desc)
        if i < len(a.dtype.names) :
            try: shape= a[name][0].shape
            except: shape= a[name].shape
        else :
            try: shape= b[name][0].shape
            except: shape= b[name].shape
        if len(desc) > 2 :
            newdtype.append((desc[0],desc[1],shape))
        else :
            newdtype.append((desc[0],desc[1]))
    # create new array
    newrecarray = np.empty(len(a), dtype = newdtype)
    # fill in all of the old columns
    #print('copying...')
    for name in a.dtype.names:
         #print(name)
         newrecarray[name] = a[name]
    return newrecarray


def append(a,b) :
    '''
    Append two structured arrays, checking for same fields, and increasing size of character fields
    as needed
    '''

    dt_a=a.dtype.descr
    dt_b=b.dtype.descr
    if len(dt_a) != len(dt_b) :
        print("structures don't have same number of fields")

    dt=copy.copy(dt_a)
    for i in range(len(dt_a)) :
        if dt_a[i][0] != dt_b[i][0] :
            print("fields don't match",i,dt_a[i],dt_b[i])
        elif dt_a[i][1].find('S') >= 0 :
            j=dt_a[i][1].find('S')
            n=len(dt_a[i][1])
            s_a=int(dt_a[i][1][j+1:n])
            n=len(dt_b[i][1])
            s_b=int(dt_b[i][1][j+1:n])
            dt[i]=(dt_a[i][0],dt_a[i][1][0:j+1]+'{:<d}'.format(max([s_a,s_b])))
            #print(dt_a[i][0],dt_a[i][1],dt_b[i][0],dt_b[i][1],dt[i][1],s_a,s_b)
        try :
            # kludge to get nD array elements done properly, instead of being converted to 1D, don't know why this is necessary
            # somehow dtype doesn't have the information, although shape gets it
            if a[dt_a[i][0]].ndim > 2 :
                dt[i]=(dt_a[i][0],dt_a[i][1],a[dt_a[i][0]].shape[1:len(a[dt_a[i][0]].shape)])
        except :
            pass
    dt=np.dtype(dt)
    # unforutantely, broadcasting of the ND array in np.append doesn't seem to work!
    return np.append(a.astype(dt),b.astype(dt)), dt

def concat(files,hdu=1,verbose=False,fixfield=False) :
    '''
    Create concatenation of structures from an input list of files files; structures must have identical tags

    Args:
        files : single file name(str) or list of files, can include wildcards (expanded using glob)
  
    Keyword args:
        hdu=  : specifies which HDU to read/concatenation (default=1)

    Returns:
        structure with concatenated records
    '''
    if type(files) == str:
        files=[files]
    allfiles=[]
    for file in files :
        allfiles.extend(glob.glob(file))
    if len(allfiles) == 0 :
        print('no files found!',file)
        return

    # first go through and determine the maximally sized dtypes to be able to store data from all files
    # append the arrays using np.append, but unfortunately, this doesn't seem to work for nD fields
    ntot=0
    for file in allfiles :
        if verbose: print(file)
        a=fits.open(file)[hdu].data
        if fixfield : 
            a = add_cols(a,np.zeros(len(a),dtype=[('ALTFIELD','S24')]))
            a['ALTFIELD'] = os.path.basename(os.path.dirname(file))

        if file == allfiles[0] :
            app=a
        else :
            app,dt=append(app,a)
        ntot+=len(a)
        if verbose: print(len(app), len(a))

    # since broadcasting in np.append doesn't seem to work, create an empty structure using the maximal
    #    dtype, and load it up line by line, and field by field
    all=np.empty(ntot,dt)

    j=0
    for file in allfiles :
        if verbose: print(file)
        a=fits.open(file)[hdu].data
        if fixfield : 
            a = add_cols(a,np.zeros(len(a),dtype=[('ALTFIELD','S24')]))
            a['ALTFIELD'] = os.path.basename(os.path.dirname(file))
        for i in range(len(a)) :
          for name in all.dtype.names :
            # strings and chararrays need to behandled properly
            try:
              all[name][j] = a[name][i]
            except:
              all[name][j] = a[name][i][0]
          j+=1
        if verbose: print(len(all), len(a))

    return all


def wrfits(a,file) :
    '''
    write input structure to FITS file
    '''
    tab=fits.BinTableHDU.from_columns(a)
    tab.writeto(file,clobber=True)


def dict2struct(d) :
    '''
    convert dictionary to structure
    '''

    # loop over keys, build up dtype
    dt=[]
    for key in sorted(d) :
        val = d[key]
        if isinstance(val,np.ndarray) :
            l = len(val)
            val = val[0]
        else :
            l = 1
        if isinstance(val,str) : 
            tt = 'S{:d}'.format(len(val))
        if isinstance(val,int) or isinstance(val,np.integer) : 
            if l == 1 :
                tt = 'i4'.format(l)
            else :
                tt = '{:d}d4'.format(l)
        if isinstance(val,float) or isinstance(val,np.floating) : 
            if l == 1 :
                tt = 'f4'.format(l)
            else :
                tt = '{:d}f4'.format(l)
        dt.append((key,tt))

    # declare and fill structured array
    rec = np.recarray(1,dtype=dt)
    for key in d :
        rec[key] = d[key]

    return(rec)

def rmfield( a, *fieldnames_to_remove ):
    """ Remove fields from structure
    """
    return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]

