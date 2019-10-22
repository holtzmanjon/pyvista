import numpy as np
import astropy
from astropy.nddata import CCDData
from astropy.nddata import NDData
from astropy.io import fits
from astropy.io import ascii
from astropy.modeling import models, fitting
from astropy.convolution import convolve, Box1DKernel

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import matplotlib.pyplot as plt
import glob
import bz2
import os
import pdb
from pyvista import image
try: 
    import pyds9
except:
    print('pyds9 is not available, proceeding')

class Reader() :
    """ Class for reading images with shorthand input, e.g., image number 

        Will use glob to search for files with name '{:s}/{:s}.{:s}.fits'.format(dir,root,formstr)
    """

    def __init__(self,dir='./',root='*',formstr='{:04d}',gain=1.,rn=0.,inst=None,verbose=True) :
        """ set default directory and format strings for reading with shorthand number ID if desired
            set gain and readout noise to add uncertainty array upon reading
        """
        self.dir=dir
        self.root=root
        self.verbose=verbose

        # we will allow for instruments to have multiple channels, so everything goes in lists
        if type(gain) is list : self.gain=gain
        else : self.gain = [gain]
        if type(rn) is list : self.rn=rn
        else : self.rn = [rn]
        if type(formstr) is list : self.formstr=formstr
        else : self.formstr=[formstr]
        if inst == 'DIS' :
            # DIS has two channels so we we read both
            self.formstr=['{:04d}b','{:04d}r']
            self.gain=[1.68,1.88]
            self.rn=[4.9,4.6]
        elif inst == 'ARCES' :
            self.formstr=['{:04d}']
            self.gain=[3.8]
            self.rn=[7]
        elif inst == 'ARCTIC' :
            self.formstr=['{:04d}']
            self.gain=[2.0]
            self.rn=[3.7]
        self.nchip = len(self.formstr)
        if self.verbose :
            for form in self.formstr :
                print('will use format:  {:s}/{:s}.{:s}.fits'.format(self.dir,self.root,form))
            print('         gain:  {}    rn: {}'.format(self.gain,self.rn))

    def rd(self,num, ext=0) :
        """ 
        Read an image
        Args :
            num (str or int) : name or number of image to read
        Returns :
            image (CCDData ) : CCDData object
        """
        out=[]
        # loop over different channels (if any)
        for form,gain,rn in zip(self.formstr,self.gain,self.rn) :
            # find the files that match the directory/format
            file=glob.glob(self.dir+'/'+self.root+form.format(num)+'.fits*')
            if len(file) > 1 : 
                if self.verbose : print('more than one match found, using first!',file)
            file=file[0]

            # read the file into a CCDData object
            if self.verbose : print('Reading file: ', file)
            im=CCDData.read(file,hdu=ext,unit='adu')
            # Add uncertainty
            im.uncertainty = np.sqrt( im.data/gain + (rn/gain)**2 )
            out.append(im)

        # return the data
        if len(out) == 1 : return out[0]
        else : return out
        

class Reducer() :
    """ Class for reducing images of a given instrument
    """
    def __init__(self,inst=None,verbose=True) :
        """  Initialize reducer with information about how to reduce
        """
        self.verbose= verbose
        if inst == 'DIS' :
            # DIS has two channels
            self.biastype = 0
            self.biasbox = [ image.BOX(xr=[2050,2096],yr=[0,2047]) ,
                             image.BOX(xr=[2050,2096],yr=[0,2047]) ]
            self.trimbox = [ image.BOX(xr=[0,2047],yr=[0,1023]) ,
                             image.BOX(xr=[0,2047],yr=[0,1023]) ]
            self.normbox = [ image.BOX(xr=[1000,1050],yr=[500,600]) ,
                             image.BOX(xr=[1000,1050],yr=[500,600]) ]
        elif inst == 'ARCES' :
            self.biastype = 0
            self.biasbox = [image.BOX(xr=[2052,2057],yr=[20,2028])]
            self.trimbox = [image.BOX(xr=[200,1850],yr=[0,2047])]
            self.normbox = [ image.BOX(xr=[1000,1050],yr=[1000,1050]) ]
            
        elif inst == 'ARCTIC' :
            self.biastype = 0
            self.biasbox = [image.BOX(xr=[2052,2090],yr=[20,2028])]
            self.trimbox = [image.BOX(xr=[0,2048],yr=[0,2048])]
            self.normbox = [ image.BOX(xr=[800,1200],yr=[800,1200]) ]

        if self.verbose :
            print('Biastype : {:d}'.format(self.biastype))
            print('Bias box: ')
            for box in self.biasbox :
                box.show()
            
    def overscan(self,im,trim=False) :
        """ Overscan subtration
        """
        if type(im) is not list : ims=[im]
        else : ims = im
        if self.biastype == 0 :    
            out=[]
            for im,biasbox in zip(ims,self.biasbox) :
                b=biasbox.mean(im.data)
                if self.verbose: print('subtracting overscan: ', b)
                im.data = im.data.astype(float)-b
                im.header.add_comment('subtracted overscan: {:f}'.format(b))
                out.append(im)
            
        #elif det.biastype == 1 :
        #    over=np.median(hdu[ext].data[:,det.biasbox.xmin:det.biasbox.xmax],axis=1)
        #    boxcar = Box1DKernel(10)
        #    over=convolve(over,boxcar,boundary='extend')
        #    over=image.stretch(over,ncol=hdu[ext].data.shape[1])
        #    hdu[ext].data -= over
        if trim :
            for im,trimbox in zip(ims,self.trimbox) :
                pdb.set_trace()
                im.data = im.data[trimbox.ymin:trimbox.ymin+trimbox.nrow(),
                                  trimbox.xmin:trimbox.xmin+trimbox.ncol()]
                im.uncertainty.array = im.uncertainty.array[trimbox.ymin:trimbox.ymin+trimbox.nrow(),
                                                            trimbox.xmin:trimbox.xmin+trimbox.ncol()]
    
    def bias(self,im,superbias=None) :
         """ Superbias subtraction
         """
         # only subtract if we are given a superbias!
         if superbias is None : return

         # work with lists so that we can handle multi-channel instruments
         if type(im) is not list : ims=[im]
         else : ims = im
         if type(superbias) is not list : superbiases=[superbias]
         else : superbiases = superbias
         for im,bias in zip(ims,superbiases) :
             if self.verbose : print('subracting superbias...')
             # subtract superbias
             im.data -= bias.data
             # adjust uncertainty
             im.uncertainty = np.sqrt(im.uncertainty.array**2 + bias.uncertainty.array**2)
             im.header.add_comment('subtracted superbias')

    def flat(self,im,superflat=None) :
         """ Flat fielding
         """
         # only flatfield if we are given a superflat!
         if superflat is None : return

         if type(im) is not list : ims=[im]
         else : ims = im
         if type(superflat) is not list : superflats=[superflat]
         else : superflats = superflat
         for im,flat in zip(ims,superflats) :
             if self.verbose : print('flat fielding...')
             im.uncertainty.array =  (im.uncertainty.array**2 / im.data**2 + flat.uncertainty.array**2 / flat.data**2 ) 
             im.data /= flat.data
             im.uncertainty.array *=  im.data**2
             im.header.add_comment('divided by superflat')

    def reduce(self,im,superbias=None,superflat=None) :
        """ Full reduction
        """
        self.overscan(im)
        self.bias(im,superbias=superbias)
        self.flat(im,superflat=superflat)

class Combiner() :
    """ Class for combining calibration data frames
    """
    def __init__(self,reader=None,reducer=None,verbose=True) :
        self.reader = reader
        self.reducer = reducer
        self.verbose = verbose

    def combine(self,ims, superbias=None, normalize=False,display=None,div=True) :
        """ Combine images from list of images 
        """
        # create list of images, reading and overscan subtracting
        allcube = []
        for im in ims :
            if self.verbose: print('im: ',im)
            data = self.reader.rd(im)
            self.reducer.overscan(data)
            if superbias is not None :
                self.reducer.bias(data,superbias)
            allcube.append(data)
        nframe = len(allcube)
        # if just one frame, put in 2D list anyway so we can use same code, allcube[nframe][nchip]
        if self.reader.nchip == 1 :
            allcube=[list(i) for i in zip(*[allcube])]

        # same for multichip
        out=[] 
        for chip in range(self.reader.nchip) :
            datacube = []
            varcube = []
            for im in range(nframe) :
                if normalize :
                    norm=self.reducer.normbox[chip].mean(allcube[im][chip].data)
                    allcube[im][chip].data /= norm
                    allcube[im][chip].uncertainty.array /= norm
                datacube.append(allcube[im][chip].data)
                varcube.append(allcube[im][chip].uncertainty.array**2)
            if self.verbose: print('median combining data....')
            med = np.median(np.array(datacube),axis=0)
            if self.verbose: print('calculating uncertainty....')
            sig = 1.253 * np.sqrt(np.mean(varcube,axis=0)/nframe)
            out.append(NDData(med,uncertainty=sig))
            if display :
                for im in range(nframe) :
                    if div :
                        display.tv(allcube[im][chip].data/med)
                    else :
                        display.tv(allcube[im][chip].data-med)
                    print('image: ',im) 
                    pdb.set_trace()

        # return the frame
        if len(out) == 1 : return out[0]
        else : return out

    def superbias(self,ims,display=None) :
        """ Driver for superbias combination (no superbias subraction no normalization)
        """
        return self.combine(ims,display=display,div=False)

    def superflat(self,ims,superbias=None,display=None) :
        """ Driver for superflat combination (with superbias if specified, normalize to normbox
        """
        return self.combine(ims,superbias=superbias,normalize=True,display=display)

    def specflat(self,ims,superbias=None,wid=100,display=None) :
        """ Spectral flat takes out variation along wavelength direction
        """
        flats = self.combine(ims,superbias=superbias,normalize=True,display=display)
        boxcar = Box1DKernel(wid)
        for iflat,flat in enumerate(flats) : 
            nrows=flats[iflat].data.shape[0]
            c=convolve(flats[iflat].data[int(nrows/2)-50:int(nrows/2)+50,:].sum(axis=0),boxcar,boundary='extend')
            for row in range(flats[iflat].data.shape[0]) :
                flats[iflat].data[row,:] /= (c/wid)
                flats[iflat].uncertainty.array[row,:] /= (c/wid)

        return flats


# old combine routine
def combine(ims,norm=False,bias=None,flat=None,trim=False,verbose=False,
            disp=None,min=None,max=None,div=False) :
    """ 
    Combines input list of images (names or numbers) by median, 
    optionally normalizes before combination

    Args:
      ims (list of int or str): list of images to combine

    Keyword args:
      norm (bool) : normalize images before combining? (default=False)
      bias (numpy array) : bias frame to subtract? (default=None)
      flat (numpy array) : flat frame to divide? (default=None)

    Returns:
      median of input data arrays, after reduction as requested
    """
    cube=[]
    for im in ims :
        print('Reading image: ', im)
        h=reduce(im,bias=bias,flat=flat,trim=trim,verbose=verbose) 
        if norm :
            b=det.normbox
            norm=np.median(h.data[b.ymin:b.ymax,b.xmin:b.xmax])
            print('Normalizing image by : ', norm)
            cube.append(h.data/norm)
        else :
            cube.append(h.data)
    print('Combining: ', ims)
    comb = np.median(cube,axis=0)
    if disp is not None :
        for im in ims :
            print(im)
            h=reduce(im,bias=bias,flat=flat,trim=trim,verbose=verbose) 
            if norm :
                b=det.normbox
                norm=np.median(h.data[b.ymin:b.ymax,b.xmin:b.xmax])
                h.data /= norm
            print('Normalizing image by : ', norm)
            if div :
                disp.tv(h.data/comb,min=min,max=max)
            else :
                disp.tv(h.data-comb,min=min,max=max)
        disp.tv(comb,min=min,max=max)
        pdb.set_trace()
    return comb
    
def specflat(flat,rows=True,indiv=False,wid=100) :
    """
    Removes spectral signature from a flat by dividing by smoothed version

    Args:
      flat: input flat fields
   
    Keyword args:
      rows (bool) : specifies if smoothing is along rows (default), otherwise columns
      indiv (bool) : specifies if smoothing is done row-by-row, or single for whole image (default)
      wid (int) : boxcar kernel width (default=100)

    Returns:
      flat with spectral signature removed
    """     
    boxcar = Box1DKernel(wid)
    smooth=flat
    if rows :
        if indiv :
            for row in range(flat.shape[0]) :
                smooth[row,:] /= convolve(flat[row,:],boxcar,boundary='extend')
        else : 
            c=convolve(flat.sum(axis=0),boxcar,boundary='extend')
            for row in range(flat.shape[0]) :
                smooth[row,:] /= c

    else :
        print('smoothing by columns not yet implemented!')
        pdb.set_trace()
      
    return smooth 

class DET() :
    """ 
    Defines detector class 
    """
    def __init__(self,inst=None,gain=0.,rn=0.,biastype=0,biasbox=None,normbox=None,trimbox=None,formstr='{:04d}') :
        self.gain = gain
        self.rn = rn
        self.biastype = biastype
        if biasbox is None : self.biasbox = image.BOX()
        else : self.biasbox = biasbox
        if normbox is None : self.normbox = image.BOX()
        else : self.normbox = normbox
        if trimbox is None : self.trimbox = image.BOX()
        else : self.trimbox = trimbox
        self.formstr = formstr
        if inst == 'ARCES' :
            # APO ARCES
            self.gain=3.8
            self.rn=7
            self.biasbox.set(2052,2057,20,2028)
            self.trimbox.set(200,1850,0,2047)
            self.biasbox = [ image.BOX(xr=[1030,1050],yr=[0,2047]) ,
                             image.BOX(xr=[1030,1050],yr=[0,2047]) ]
            self.trimbox = [ image.BOX(xr=[0,2047],yr=[0,1023]) ,
                             image.BOX(xr=[1030,1050],yr=[0,2047]) ]
        elif inst == 'DIS' :
            # DIS blue
            self.gain=[1.71,1.71]
            self.rn=[3.9,3.9]
            self.biasbox = [ image.BOX(xr=[1030,1050],yr=[0,2047]) ,
                             image.BOX(xr=[1030,1050],yr=[0,2047]) ]
            self.trimbox = [ image.BOX(xr=[0,2047],yr=[0,1023]) ,
                             image.BOX(xr=[1030,1050],yr=[0,2047]) ]
            self.formstr=['{:04d}b','{:04d}r']

def look(tv,pause=True,files=None,list=None,min=None, max=None) :
    """ 
    Displays series of files 
    """
    if files is None and list is None :
        files=glob.glob(indir+'/*.fits*')
    if list is not None :
        f=open(indir+list,'r')
        files=[]
        for line in f :
            files.extend(np.arange(int(line.split()[0]),int(line.split()[1])))
        f.close()
        pdb.set_trace()
 
    for file in files :
       hd=read(file,verbose=True,bias=False)
       disp(tv,hd,min=min,max=max)
       if pause :
           pdb.set_trace()

def getfiles(type,listfile=None,filter=None,verbose=False) :
    """ 
    Get all files of desired type from specified directory. If file is specified, read numbers from that file, else use IMAGETYP card
    """
    if listfile is None :
        list=[]
        if verbose: print('directory: ', indir)
        for file in glob.glob(indir+'/'+root+'*.fits*') :
            head=fits.open(file)[0].header
            #if verbose :
            #   print('file: ', file)
            #   print('IMAGETYPE: ', head['IMAGETYP'])

            try :
                if head['IMAGETYP'] == type :
                    if filter is None or head['FILTER'] == filter :
                        list.append(file)
            except :
                pass
    else :
        list=ascii.read(indir+listfile,Reader=ascii.NoHeader)['col1']
    return list

def disp(tv,hd,min=None,max=None,sky=False) :
    """ 
    Displays HDU or data array on specified ds9/tv device
    """

    if type(tv) is pyds9.DS9 :
        if isinstance(hd, (np.ndarray)) :
            tv.set_np2arr(hd)
            data=hd
        elif isinstance(hd, (astropy.io.fits.hdu.hdulist.HDUList)) :
            tv.set_pyfits(hd)
            data=hd[0].data
        elif isinstance(hd, (astropy.io.fits.hdu.image.PrimaryHDU)) :
            tv.set_np2arr(hd.data)
            data=hd.data
        else :
            print('Unrecognized data type for display: ',type(hd)) 
        if sky :
           skyval = mmm.mmm(data)
           min = skyval[0]-5*skyval[1]
           max = skyval[0]+20*skyval[1]
        if min is not None and max is not None :
            tv.set("scale limits {:5d} {:5d}".format(min,max))
        else :
            tv.set("scale histequ")

    else :
        if isinstance(hd, (astropy.io.fits.hdu.hdulist.HDUList)) :
            tv.tv(hd[0],min=min,max=max)
        else :
            tv.tv(hd,min=min,max=max)

