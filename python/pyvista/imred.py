import numpy as np
import astropy
import copy
from astropy import units
from astropy.nddata import CCDData, NDData, StdDevUncertainty
from astropy.nddata import NDData
from astropy.io import fits
from astropy.io import ascii
from astropy.modeling import models, fitting
from astropy.convolution import convolve, Box1DKernel, Box2DKernel, Box2DKernel
import ccdproc
import scipy.signal
import yaml

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

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/../../'

class Reducer() :
    """ Class for reducing images of a given instrument
    """
    def __init__(self,inst=None,dir='./',root='*',formstr='{04d}',gain=1,rn=0.,verbose=True,nfowler=1) :
        """  Initialize reducer with information about how to reduce
        """
        self.dir=dir
        self.root=root
        self.verbose=verbose
        self.inst=inst
        self.badpix=None
        self.scat=None
        self.mask=None
        self.display=None

        # we will allow for instruments to have multiple channels, so everything goes in lists
        self.channels=['']
        if type(gain) is list : self.gain=gain
        else : self.gain = [gain]
        if type(rn) is list : self.rn=rn
        else : self.rn = [rn]
        if type(formstr) is list : self.formstr=formstr
        else : self.formstr=[formstr]
        
        config = yaml.load(open(ROOT+'/data/'+inst+'/'+inst+'_config.yml','r'), Loader=yaml.FullLoader)
        self.channels=config['channels']
        self.formstr=config['formstr']
        self.gain=config['gain']
        self.rn=config['rn']/np.sqrt(nfowler)
        self.crbox=config['crbox']
        self.biastype=config['biastype']
        self.biasbox=[]
        for box in config['biasbox'] :
            self.biasbox.append(image.BOX(xr=box[0],yr=box[1]) )
        self.trimbox=[]
        for box in config['trimbox'] :
            self.trimbox.append(image.BOX(xr=box[0],yr=box[1]) )
        self.normbox=[]
        for box in config['normbox'] :
            self.normbox.append(image.BOX(xr=box[0],yr=box[1]) )
           
        if inst == 'DIS' :
            # DIS has two channels so we we read both
            self.channels=['blue','red']
            self.formstr=['{:04d}b','{:04d}r']
            self.gain=[1.68,1.88]
            self.rn=[4.9,4.6]
            self.crbox=[1,11]
            self.biastype = 0
            self.biasbox = [ image.BOX(xr=[2050,2096],yr=[0,2047]) ,
                             image.BOX(xr=[2050,2096],yr=[0,2047]) ]
            self.trimbox = [ image.BOX(xr=[0,2047],yr=[0,1023]) ,
                             image.BOX(xr=[0,2047],yr=[0,1023]) ]
            self.normbox = [ image.BOX(xr=[1000,1050],yr=[500,600]) ,
                             image.BOX(xr=[1000,1050],yr=[500,600]) ]


        elif inst == 'ARCES' :
            self.formstr=['{:04d}']
            self.gain=[3.8]
            self.rn=[7]
            self.crbox=[1,11]
            self.scat = 10
            self.biastype = 0
            self.biasbox = [image.BOX(xr=[2075,2125],yr=[20,2028])]
            self.trimbox = [image.BOX(xr=[200,1850],yr=[0,2047])]
            self.normbox = [ image.BOX(xr=[1000,1050],yr=[1000,1050]) ]
            self.badpix = [ [ image.BOX(yr=[0,2067],xr=[0,200]),      # left side
                              image.BOX(yr=[0,2067],xr=[1900,2127]), # right side
                              image.BOX(yr=[802,2000],xr=[787,787]),
                              image.BOX(yr=[663,2000],xr=[1682,1682]),
                              image.BOX(yr=[219,2067],xr=[101,101]),
                              image.BOX(yr=[1792,1835],xr=[1284,1284]),
                              image.BOX(yr=[1474,2067],xr=[1355,1355]),
                              image.BOX(yr=[1418,1782],xr=[1602,1602]),
                              image.BOX(yr=[1905,1943],xr=[1382,1382]),
                              image.BOX(yr=[1926,1974],xr=[1416,1416]),
                              image.BOX(yr=[1610,1890],xr=[981,981]),
                              image.BOX(yr=[1575,2067],xr=[490,490]),
                              image.BOX(yr=[1710,1722],xr=[568,568]),
                              image.BOX(yr=[1905,1981],xr=[653,654]),
                              image.BOX(yr=[1870,1925],xr=[853,853]) ] ]
            
        elif inst == 'ARCTIC' :
            self.formstr=['{:04d}']
            self.gain=[2.0]
            self.rn=[3.7]
            self.biastype = 0
            self.crbox=[11,11]
            self.biasbox = [image.BOX(xr=[2052,2090],yr=[20,2028])]
            self.trimbox = [image.BOX(xr=[0,2048],yr=[0,2048])]
            self.normbox = [ image.BOX(xr=[800,1200],yr=[800,1200]) ]

        elif inst == 'TSPEC' :
            self.formstr=['{:04d}']
            self.gain=[3.5]
            self.rn=[18]/np.sqrt(nfowler)
            self.crbox=[1,11]
            self.biastype = -1
            self.biasbox = [image.BOX(xr=[0,2048],yr=[0,1024])]
            self.trimbox = [image.BOX(xr=[0,2048],yr=[0,1024])]
            self.normbox = [ image.BOX(xr=[256,956],yr=[570,660]) ]

        try: self.mask=fits.open(os.environ['PYVISTA_DIR']+'/data/'+inst+'/'+inst+'_mask.fits')[0].data
        except: pass

        # save number of chips for convenience
        self.nchip = len(self.formstr)

        if self.verbose :
            for form in self.formstr :
                print('  will use format:  {:s}/{:s}.{:s}.fits'.format(self.dir,self.root,form))
            print('         gain:  {}    rn: {}'.format(self.gain,self.rn))
            print('  Biastype : {:d}'.format(self.biastype))
            print('  Bias box: ')
            for box in self.biasbox :
                box.show()
            print('  Trim box: ')
            for box in self.trimbox :
                box.show()
            print('  Norm box: ')
            for box in self.normbox :
                box.show()

    def rd(self,num, ext=0) :
        """ Read an image

        Args :
            num (str or int) : name or number of image to read
        Returns :
            image (CCDData ) : CCDData object
        """
        out=[]
        # loop over different channels (if any)
        idet=0 
        for form,gain,rn in zip(self.formstr,self.gain,self.rn) :
            # find the files that match the directory/format
            if type(num) is int :
                search=self.dir+'/'+self.root+form.format(num)+'.fits*'
            elif type(num) is str :
                if num.find('/') >= 0 :
                    search='*'+num+'*'
                else :
                    search=self.dir+'/*'+num+'*'
            file=glob.glob(search)
            if len(file) == 0 : 
                print('cannot find file matching: '+search)
                return
            elif len(file) > 1 : 
                if self.verbose : print('more than one match found, using first!',file)
            file=file[0]

            # read the file into a CCDData object
            if self.verbose : print('  Reading file: ', file)
            im=CCDData.read(file,hdu=ext,unit='adu')

            # Add uncertainty (will be in error if there is an overscan
            data=copy.copy(im.data)
            data[data<0] = 0.
            im.uncertainty = StdDevUncertainty(np.sqrt( data/gain + (rn/gain)**2 ))

            # Add mask
            if self.mask is not None : im.mask = self.mask
            else : im.mask = np.zeros(im.data.shape,dtype=bool)
            if self.badpix is not None :
                for badpix in self.badpix[idet] :
                    badpix.setval(im.mask,True)

            out.append(im)
            idet+=1

        # return the data
        if len(out) == 1 : return out[0]
        else : return out
            
    def overscan(self,im,trim=False) :
        """ Overscan subtration
        """
        if type(im) is not list : ims=[im]
        else : ims = im
       
        for im,gain,rn,biasbox,trimbox in zip(ims,self.gain,self.rn,self.biasbox,self.trimbox) :
            if self.display is not None : 
                self.display.clear()
                self.display.tv(im)
            if self.biastype == 0 :
                b=biasbox.mean(im.data)
                if self.verbose: print('  subtracting overscan: ', b)
                if self.display is not None : 
                    self.display.tvbox(0,0,box=biasbox)
                    self.display.plotax1.plot(np.mean(im.data[:,biasbox.xmin:biasbox.xmax],axis=1))
                    plt.draw()
                    input("  See bias box and cross section. Hit any key to continue")
                im.data = im.data.astype(float)-b
                im.header.add_comment('subtracted overscan: {:f}'.format(b))
            #elif det.biastype == 1 :
            #    over=np.median(hdu[ext].data[:,det.biasbox.xmin:det.biasbox.xmax],axis=1)
            #    boxcar = Box1DKernel(10)
            #    over=convolve(over,boxcar,boundary='extend')
            #    over=image.stretch(over,ncol=hdu[ext].data.shape[1])
            #    hdu[ext].data -= over

            # Add uncertainty (redo from scratch after overscan)
            data=copy.copy(im.data)
            data[data<0] = 0.
            im.uncertainty = StdDevUncertainty(np.sqrt( data/gain + (rn/gain)**2 ))

            # Trim if requested
            if trim:
                im.data = im.data[trimbox.ymin:trimbox.ymin+trimbox.nrow(),
                                  trimbox.xmin:trimbox.xmin+trimbox.ncol()]
                im.uncertainty.array = im.uncertainty.array[trimbox.ymin:trimbox.ymin+trimbox.nrow(),
                                                            trimbox.xmin:trimbox.xmin+trimbox.ncol()]
                if im.mask is not None :
                    im.mask = im.mask[trimbox.ymin:trimbox.ymin+trimbox.nrow(),
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
         out=[]
         for im,bias in zip(ims,superbiases) :
             if self.verbose : print('  subtracting superbias...')
             out.append(ccdproc.subtract_bias(im,bias))
         if len(out) == 1 : return out[0]
         else : return out

    def flat(self,im,superflat=None) :
         """ Flat fielding
         """
         # only flatfield if we are given a superflat!
         if superflat is None : return

         if type(im) is not list : ims=[im]
         else : ims = im
         if type(superflat) is not list : superflats=[superflat]
         else : superflats = superflat
         out=[]
         for im,flat in zip(ims,superflats) :
             if self.verbose : print('  flat fielding...')
             out.append(ccdproc.flat_correct(im,flat))
         if len(out) == 1 : return out[0]
         else : return out


    def scatter(self,im,scat=None,display=None,smooth=3) :
        """ Removal of scattered light (for multi-order/object spectrograph)
        """
        if scat is None : return

        print('  estimating scattered light ...')
        boxcar = Box1DKernel(smooth)
        points=[]
        values=[]
        nrows = im.data.shape[0]
        ncols = im.data.shape[-1]

        # find minima in each column, and save location and value
        for col in range(0,ncols,scat) :
            print('    column: {:d}'.format(col),end='\r')
            yscat = scipy.signal.find_peaks(-convolve(im.data[:,col],boxcar))[0]
            for y in yscat :
                if im.mask is None or not im.mask[y,col] :
                    points.append([y,col])
                    values.append(im.data[y,col])

        # fit surface to the minimum values
        print('    fitting surface ...')
        grid_x, grid_y = np.mgrid[0:nrows,0:ncols]

        # smooth and reject outlying points
        boxcar = Box2DKernel(31)
        grid_z=convolve(scipy.interpolate.griddata(points,values,(grid_x,grid_y),
                        method='cubic',fill_value=0.),boxcar)
        # go back and try to reject outliers
        print('    rejecting points ...')
        points_gd=[]
        values_gd=[]
        for point,value in zip(points,values) :
            if value < 1.1*grid_z[point[0],point[1]] :
                points_gd.append(point)
                values_gd.append(value)

        # refit surface
        print('    refitting surface ...')
        grid_z=convolve(scipy.interpolate.griddata(points_gd,values_gd,(grid_x,grid_y),
                        method='cubic',fill_value=0.),boxcar)

        if display is None and self.display is not None : display = self.display
        if display is not None :
            self.display.clear()
            display.tv(im)
            points=np.array(points)
            display.ax.scatter(points[:,1],points[:,0],color='r',s=1)
            points_gd=np.array(points_gd)
            display.ax.scatter(points_gd[:,1],points_gd[:,0],color='g',s=1)
            input("  See image with scattered light points. Hit any key to continue".format(im))
            display.clear()
            display.tv(im)
            display.tv(grid_z)
            col=int(im.shape[-1]/2)
            display.plotax1.cla()
            display.plotax1.plot(im.data[:,col])
            display.plotax1.plot(grid_z[:,col])
            plt.draw()
            input("  See scattered light image. Hit any key to continue".format(im))

        im.data -= grid_z

    def crrej(self,im,crbox=None,nsig=5) :
        """ CR rejection
        """
        if crbox is None: return
        if type(im) is not list : ims=[im]
        else : ims = im
        for im in ims :
            if self.verbose : print('  zapping CRs with filter [{:d},{:d}]...'.format(*crbox))
            if self.display is not None : 
                self.display.clear()
                self.display.tv(im)
            image.zap(im,crbox,nsig=nsig)
            if self.display is not None : 
                self.display.tv(im)
                input("  See CR-zapped image and original with - key. Hit any key to continue")

    def badpix_fix(self,im,val=0.) :
        """ Replace bad pixels
        """
        if val is None : return
        if type(im) is not list : ims=[im]
        else : ims = im
        for i, im in enumerate(ims) :
            if im.mask is not None :
                 bd=np.where(im.mask)
                 ims[i].data[bd[0],bd[1]] = val
                 ims[i].uncertainty.array[bd[0],bd[1]] = np.inf

    def reduce(self,num,crbox=None,superbias=None,superdark=None,superflat=None,scat=None,badpix=None,return_list=False,display=None) :
        """ Full reduction
        """
        self.display = display
        im=self.rd(num)
        self.overscan(im)
        self.crrej(im,crbox=crbox)
        im=self.bias(im,superbias=superbias)
        self.scatter(im,scat=scat)
        im=self.flat(im,superflat=superflat)
        self.badpix_fix(im,val=badpix)
        if return_list and type(im) is not list : im=[im]
        return im

    def write(self,im,name,overwrite=True) :
        """ write out image, deal with multiple channels 
        """
        if type(im) is list :
            for i,frame in enumerate(im) : frame.write(name+'_'+self.channels[i]+'.fits',overwrite=overwrite)
        else :
            im.write(name+'.fits',overwrite=overwrite)


class Combiner() :
    """ Class for combining calibration data frames
    """
    def __init__(self,reducer=None,verbose=True) :
        self.reducer = reducer
        self.verbose = verbose

    def getcube(self,ims,**kwargs) :
        """ Read images into data cube
        """
        # create list of images, reading and overscan subtracting
        allcube = []
        for im in ims :
            if type(im) is not astropy.nddata.CCDData :
                data = self.reducer.reduce(im, **kwargs)
            allcube.append(data)

        # if just one frame, put in 2D list anyway so we can use same code, allcube[nframe][nchip]
        if self.reducer.nchip == 1 :
            allcube=[list(i) for i in zip(*[allcube])]

        return allcube

    def sum(self,ims, return_list=False, **kwargs) :
        """ Coadd input images
        """
        allcube = self.getcube(ims, **kwargs)
        nframe = len(allcube)
        
        out=[]
        for chip in range(self.reducer.nchip) :
            datacube = []
            varcube = []
            for im in range(nframe) :
                datacube.append(allcube[im][chip].data)
                varcube.append(allcube[im][chip].uncertainty.array**2)
            sum = np.sum(np.array(datacube),axis=0)
            sig = np.sqrt(np.sum(np.array(varcube),axis=0))
            out.append(CCDData(sum,uncertainty=StdDevUncertainty(sig),unit='adu'))
        
        # return the frame
        if len(out) == 1 : 
           if return_list : return [out[0]]
           else : return out[0]
        else : return out

    def median(self,ims, normalize=False,display=None,div=True,return_list=False, **kwargs) :
        """ Combine images from list of images 
        """
        # create list of images, reading and overscan subtracting
        allcube = self.getcube(ims,**kwargs)
        nframe = len(allcube)

        # do the combination
        out=[] 
        for chip in range(self.reducer.nchip) :
            datacube = []
            varcube = []
            maskcube = []
            allnorm = []
            for im in range(nframe) :
                if normalize :
                    norm=self.reducer.normbox[chip].mean(allcube[im][chip].data)
                    allnorm.append(norm)
                    allcube[im][chip].data /= norm
                    allcube[im][chip].uncertainty.array /= norm
                datacube.append(allcube[im][chip].data)
                varcube.append(allcube[im][chip].uncertainty.array**2)
                maskcube.append(allcube[im][chip].mask)
            if self.verbose: print('  median combining data....')
            med = np.median(np.array(datacube),axis=0)
            if self.verbose: print('  calculating uncertainty....')
            sig = 1.253 * np.sqrt(np.mean(np.array(varcube),axis=0)/nframe)
            mask = np.any(maskcube,axis=0)
            comb=CCDData(med,uncertainty=StdDevUncertainty(sig),mask=mask,unit='adu')
            if normalize: comb.meta['MEANNORM'] = np.array(allnorm).mean()
            out.append(comb)

            # display final combined frame and individual frames relative to combined
            if display :
                display.clear()
                display.tv(med)
                input("  See final image. Hit any key to continue")
                for i,im in enumerate(ims) :
                    if div :
                        display.tv(allcube[i][chip].data/med,min=0.5,max=1.5)
                        input("    see image: {} divided by master, hit any key to continue".format(im))
                    else :
                        delta=5*self.reducer.rn[chip]
                        display.tv(allcube[i][chip].data-med,min=-delta,max=delta)
                        input("    see image: {} minus master, hit any key to continue".format(im))

        # return the frame
        if len(out) == 1 :
           if return_list : return [out[0]]
           else : return out[0]
        else : return out

    def superbias(self,ims,display=None,scat=None) :
        """ Driver for superbias combination (no superbias subraction no normalization)
        """
        return self.median(ims,display=display,div=False,scat=scat)

    def superflat(self,ims,superbias=None,scat=None, display=None) :
        """ Driver for superflat combination (with superbias if specified, normalize to normbox
        """
        return self.median(ims,superbias=superbias,normalize=True,scat=scat,display=display)

    def specflat(self,ims,superbias=None,scat=None,wid=100,display=None) :
        """ Spectral flat takes out variation along wavelength direction
        """
        flats = self.median(ims,superbias=superbias,scat=scat,normalize=True,display=display)
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

