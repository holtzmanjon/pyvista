import numpy as np
import astropy
from photutils import DAOStarFinder
import code
import copy
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from pyvista.dataclass import Data
import pyvista.dataclass
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.convolution import convolve, Box1DKernel, Box2DKernel, Box2DKernel
import ccdproc
import scipy.signal
import yaml
import subprocess
import sys
import tempfile
from pyvista import stars, image, tv, bitmask
try: from pyvista import apogee
except : pass
import pyvista.data as DATA

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import matplotlib
import matplotlib.pyplot as plt
import glob
import bz2
import os
import pdb
import importlib_resources
try: 
    import pyds9
except:
    #print('pyds9 is not available, proceeding')
    pass


class Reducer() :
    """ Class for reducing images of a given instrument

    Parameters
    ----------

    Attributes
    ----------
    dir : str
    root : str
    verbose : bool 
    inst : str
    badpix : str
    scat : int
    mask : str
    transpose : bool
    scale : float
    biastype : int
    gain : float
    rn : float
    namp : int
    crbox : list
    biasavg : int
    biasregion : 
    trimbox :
    outbox :


    """
    def __init__(self,inst=None,conf='',dir='./',root='*',formstr='{04d}',
                 gain=1,rn=0.,verbose=True,nfowler=1) :
        """  Initialize reducer with information about how to reduce
        """
        self.dir=dir
        self.root=root
        self.verbose=verbose
        self.inst=inst
        self.badpix=None
        self.scat=None
        self.bitmask=None
        self.transpose=None
        self.scale=1
        self.biastype=-1
        self.saturation=None

        # we will allow for instruments to have multiple channels, so everything goes in lists
        self.channels=['']
        if type(gain) is list : self.gain=gain
        else : self.gain = [gain]
        if type(rn) is list : self.rn=rn
        else : self.rn = [rn]
        if type(formstr) is list : self.formstr=formstr
        else : self.formstr=[formstr]
       
        # Read instrument configuation from YAML configuration file 
        if inst is not None :
            if inst.find('/') < 0 :
                config = yaml.load(open(importlib_resources.files(DATA).joinpath(inst+'/'+inst+
                              conf+'.yml'),'r'), Loader=yaml.FullLoader)
            else :
                config = yaml.load(open(inst+'.yml','r'), 
                              Loader=yaml.FullLoader)
            self.channels=config['channels']
            self.formstr=config['formstr']
            self.gain=config['gain']
            self.rn=config['rn']/np.sqrt(nfowler)
            try : self.saturation=config['saturation']
            except KeyError: self.saturation = None
            try :self.scale=config['scale']
            except KeyError: self.scale = None
            try : self.namp=config['namp']
            except KeyError: self.namp = 1
            try :self.transpose=config['transpose']
            except KeyError: self.transpose = False
            try : self.crbox=config['crbox']
            except KeyError: self.crbox=None
            self.biastype=config['biastype']
            try : self.biasavg=config['biasavg']
            except KeyError: self.biasavg=11
            if self.biasavg %2 == 0 : self.biasavg += 1
            self.biasbox=[]
            for box in config['biasbox'] :
                if self.namp == 1 :
                    self.biasbox.append(image.BOX(xr=box[0],yr=box[1]) )
                else :
                    ampbox=[]
                    for amp in box : 
                        ampbox.append(image.BOX(xr=amp[0],yr=amp[1]) )
                    self.biasbox.append(ampbox)
            self.biasregion=[]
            try :
              for box in config['biasregion'] :
                if self.namp == 1 :
                    self.biasregion.append(image.BOX(xr=box[0],yr=box[1]) )
                else :
                    ampbox=[]
                    for amp in box : 
                        ampbox.append(image.BOX(xr=amp[0],yr=amp[1]) )
                    self.biasregion.append(ampbox)
            except KeyError: 
                self.biasregion=[None]
            self.trimbox=[]
            for box in config['trimbox'] :
                if self.namp == 1 :
                    self.trimbox.append(image.BOX(xr=box[0],yr=box[1]) )
                else :
                    ampbox=[]
                    for amp in box : 
                        ampbox.append(image.BOX(xr=amp[0],yr=amp[1]) )
                    self.trimbox.append(ampbox)
            self.outbox=[]
            try :
                for box in config['outbox'] :
                    if self.namp == 1 :
                        self.outbox.append(image.BOX(xr=box[0],yr=box[1]) )
                    else :
                        ampbox=[]
                        for amp in box : 
                            ampbox.append(image.BOX(xr=amp[0],yr=amp[1]) )
                        self.outbox.append(ampbox)
            except: self.outbox=self.trimbox
         
            self.normbox=[]
            for box in config['normbox'] :
                self.normbox.append(image.BOX(xr=box[0],yr=box[1]) )
            try: self.scat=config['scat']
            except : pass
           
            # Add bad pixel mask if it exists
            try: self.bitmask=fits.open(importlib_resources.files(DATA).joinpath(inst+'/'+
                                inst+'_mask.fits'))[0].data
            except: pass

        # save number of chips for convenience
        self.nchip = len(self.formstr)

        # output setup if verbose
        if self.verbose :
            if inst is not None : 
              print('INSTRUMENT: {:s}   config: {:s}'.format(inst,conf))
            for form in self.formstr :
                print('  will use format:  {:s}/{:s}{:s}.fits*'.format(
                         self.dir,self.root,form))
            print('         gain:  {}    rn: {}'.format(self.gain,self.rn))
            print('         scale:  {}   '.format(self.scale))
            print('  Biastype : {:d}'.format(self.biastype))
            print('  Bias box: ')
            for box in self.biasbox :
                if self.namp == 1 : box.show()
                else :
                    for i,amp in enumerate(box) :
                        if i==0 : header = True
                        else: header=False
                        amp.show(header=header)
            print('  Trim box: ')
            for box in self.trimbox :
                if self.namp == 1 : box.show()
                else :
                    for i,amp in enumerate(box) :
                        if i==0 : header = True
                        else: header=False
                        amp.show(header=header)
            print('  Norm box: ')
            for box in self.normbox :
                box.show()

    def reduce(self,num,channel=None,crbox=None,bias=None,dark=None,flat=None,
               scat=None,badpix=None,solve=False,return_list=False,display=None,
               trim=True,seeing=2) :
        """ Reads data from disk, and performs reduction steps as determined from command 
            line parameters

            Parameters
            ----------
            id : int or str
                 Number or string specifying file to read. If a number, 
                 the filename will be constructed based on dir and formstr 
                 attributed of Reducer object. Without any additional 
                 command-line arguments, data will be read, overscan 
                 subtracted, and uncertainty array populated based
                 on gain and readout noise in Reducer attributes
            display : TV object, default=None
                 if specified, pyvista TV object to display data in as 
                 arious reduction steps are taken
            channel : int, default= None
                 if specified, channel to reduce if instrument is 
                 multi-channel (multi-file), otherwise all channels 
                 will be read/reduced
            bias : Data object, default= None
                 if specified, superbias frame to subtract
            dark : Data object, default= None
                 if specified, superdark frame to subtract
            flat : Data object, default= None
                 if specified, superflat frame to divide by
            crbox : list or str, default=None
                 if specified, parameter to pass to CR rejection routine, 
                 either 2-element list giving shape of box for median 
                 filter, or 'lacosmic'
            scat :
            badpix :
            trim :
            solve :
            seeing :

        """
        im=self.rd(num,dark=dark,channel=channel)
        self.overscan(im,display=display,channel=channel)
        im=self.bias(im,superbias=bias)
        im=self.dark(im,superdark=dark)
        self.scatter(im,scat=scat,display=display)
        im=self.flat(im,superflat=flat,display=display)
        self.badpix_fix(im,val=badpix)
        if trim and display is not None: display.tvclear()
        im=self.trim(im,trimimage=trim)
        im=self.crrej(im,crbox=crbox,display=display)
        if solve : 
            im=self.platesolve(im,display=display,scale=self.scale,seeing=seeing)
        if return_list and type(im) is not list : im=[im]
        return im

    def rd(self,num, ext=0, dark=None, channel=None) :
        """ Read an image

        Parameters
        ----------
            num (str or int) : name or number of image to read

        Returns 
        -------
            image (Data ) : Data object, but noise will be incorrect without overscan subtraction
        """
        out=[]
        # loop over different channels (if any)
        idet=0 
        if channel is not None : channels=[channel]
        else : channels = range(len(self.formstr))
        for chan in channels :
            form = self.formstr[chan]
            gain = self.gain[chan]
            rn = self.rn[chan]
        #for form,gain,rn in zip(self.formstr,self.gain,self.rn) :
            # find the files that match the directory/format
            if type(num) is int :
                search=self.dir+'/'+self.root+form.format(num)
            elif type(num) is str or type(num) is np.str_ :
                if num.find('/') >= 0 :
                    search=num+'*'
                else :
                    search=self.dir+'/*'+num+'*'
            else :
                print('stopping in rd... num:',num)
                pdb.set_trace()
            file=glob.glob(search)
            if len(file) == 0 : 
                raise ValueError('cannot find file matching: '+search)
                return
            elif len(file) > 1 : 
                if self.verbose : print('more than one match found, using first!',file)
            file=file[0]

            # read the file into a Data object
            if self.verbose : print('  Reading file: {:s}'.format(file)) 
            if self.inst == 'APOGEE' :
                im=apogee.cds(file,dark=dark)
            else :
                try : im=Data.read(file,hdu=ext,unit=u.dimensionless_unscaled)
                except : raise RuntimeError('Error reading file: {:s}'.format(file))
                im.data = im.data.astype(np.float32)
            im.header['FILE'] = os.path.basename(file)
            if 'OBJECT' not in im.header  or im.header['OBJECT'] == '':
                if 'OBJNAME' in im.header and im.header['OBJNAME'] != '' : 
                    im.header['OBJECT'] = im.header['OBJNAME']
                else : 
                    try : im.header['OBJECT'] = im.header['FILE']
                    except KeyError : print('No OBJECT, OBJNAME, or FILE in header')
            if 'RA' not in im.header  :
                try: im.header['RA'] = im.header['OBJCTRA']
                except : print('no RA or OBJCTRA found')
            if 'DEC' not in im.header  :
                try: im.header['DEC'] = im.header['OBJCTDEC']
                except : print('no DEC or OBJCTDEC found')

            # Add uncertainty (will be in error if there is an overscan, but redo with overscan subraction later)
            data=copy.copy(im.data)
            data[data<0] = 0.
            if isinstance(gain,list) :
              im.uncertainty = StdDevUncertainty(np.sqrt( data/gain[0] + (rn/gain[0])**2 ))
            else :
              im.uncertainty = StdDevUncertainty(np.sqrt( data/gain + (rn/gain)**2 ))

            # Add mask
            pixmask = bitmask.PixelBitMask()
            if self.bitmask is not None : im.bitmask = self.bitmask
            else : im.bitmask = np.zeros(im.data.shape,dtype=np.short)
            if self.badpix is not None :
                for badpix in self.badpix[idet] :
                    badpix.setval(im.bitmask,pixmask.getval('BADPIX'))
            if self.saturation is not None :
                bd = np.where(im.data >= self.saturation[chan])
                im.bitmask[bd] |= pixmask.getval('SATPIX')

            out.append(im)
            idet+=1

        # return the data
        if len(out) == 1 : return out[0]
        else : return out
            
    def overscan(self,im,display=None,channel=None) :
        """ Overscan subtraction
        """
        if self.biastype < 0 : return

        if type(im) is not list : ims=[im]
        else : ims = im
        if channel is not None :
            gains = [self.gain[channel]]
            rns = [self.rn[channel]]
            biasboxes = [self.biasbox[channel]]
            biasregions = [self.biasregion[channel]]
        else :
            gains = self.gain
            rns = self.rn
            biasboxes = self.biasbox
            biasregions = self.biasregion
        for ichan,(im,gain,rn,biasbox,biasregion) in enumerate(zip(ims,gains,rns,biasboxes,biasregions)) :
            if display is not None : 
                display.tv(im)
                ax=display.plotax2
                ax.cla()
            if self.namp == 1 : 
                ampboxes = [biasbox]
                databoxes = [biasregion]
            else :
                ampboxes = biasbox
                databoxes = biasregion
            im.data = im.data.astype(np.float32)
            color=['m','g','b','r','c','y']
            for ibias,(databox,ampbox) in enumerate(zip(databoxes,ampboxes)) :
              if display is not None :
                  display.tvbox(0,0,box=ampbox,color=color[ibias])
                  if type(databox) == image.BOX : 
                      display.tvbox(0,0,box=databox,color=color[ibias],ls='--',lw=2)
                      ax.plot(np.arange(databox.ymin,databox.ymax),
                          np.mean(im.data[databox.ymin:databox.ymax,
                          ampbox.xmin:ampbox.xmax], axis=1),color=color[ibias])
              if self.biastype == 0 :
                b=ampbox.mean(im.data)
                if self.verbose: print('  subtracting overscan: ', b)
                if display is not None : 
                    ax.plot([databox.ymin,databox.ymax],[b,b],color='k')
                    ax.text(0.05,0.95,'Overscan mean',transform=ax.transAxes)
                    ax.set_xlabel('Row')
                    display.fig.canvas.draw_idle()
                    plt.draw()
                if type(databox) == image.BOX :
                    im.data[databox.ymin:databox.ymax+1,
                            databox.xmin:databox.xmax+1] = \
                            im.data[databox.ymin:databox.ymax+1,
                               databox.xmin:databox.xmax+1].astype(np.float32)-b
                else : 
                    im.data = im.data.astype(np.float32)-b
                im.header.add_comment('subtracted overscan: {:f}'.format(b))
              elif self.biastype == 1 :
                over=np.median(im.data[databox.ymin:databox.ymax+1,
                               ampbox.xmin:ampbox.xmax],axis=1)
                #boxcar = Box1DKernel(self.biasavg)
                #over=convolve(over,boxcar,boundary='extend')
                over=scipy.signal.medfilt(over,kernel_size=self.biasavg)
                if display is not None : 
                    ax.plot(np.arange(databox.ymin,databox.ymax+1),
                            over,color='k')
                    ax.set_xlabel('Row')
                over=image.stretch(over,ncol=databox.ncol())
                if self.verbose: print('  subtracting overscan vector ')
                im.data[databox.ymin:databox.ymax+1,
                        databox.xmin:databox.xmax+1] = \
                    im.data[databox.ymin:databox.ymax+1,
                            databox.xmin:databox.xmax+1].astype(np.float32) - over
              # if we have separate gains, multiply by them here
              if isinstance(gain,list) :
                print('  multiplying by gain: ', gain[ibias])
                if type(databox) == image.BOX :
                    im.data[databox.ymin:databox.ymax+1,
                            databox.xmin:databox.xmax+1] = \
                            im.data[databox.ymin:databox.ymax+1,
                               databox.xmin:databox.xmax+1].astype(np.float32)*gain[ibias]
                else : 
                    im.data = im.data.astype(np.float32)*gain[ibias]

            if display is not None :
                display.tv(im)
                getinput("  See bias box and cross section. ",display)

            # Add uncertainty (redo from scratch after overscan)
            data=copy.copy(im.data)
            data[data<0] = 0.
            if isinstance(gain,list) :
              im.uncertainty = StdDevUncertainty(np.sqrt( data + rn**2 ))
            else :
              im.uncertainty = StdDevUncertainty(np.sqrt( data/gain + (rn/gain)**2 ))

    def trim(self,inim,trimimage=False) :
        """ Trim image by masking non-trimmed pixels
            May need to preserve image size to match reference/calibration frames, etc.
        """
        if type(inim) is not list : ims=[inim]
        else : ims = inim

        outim = []
        for  im,trimbox,outbox in zip(ims,self.trimbox,self.outbox) :
            if self.namp == 1 : 
                boxes = [trimbox]
                outboxes = [outbox]
            else : 
                boxes = trimbox
                outboxes = outbox
            if im.bitmask is None :
                print('adding a bitmask...')
                im.add_bitmask(np.zeros(im.data.shape,dtype=np.uintc))
            pixmask=bitmask.PixelBitMask()
            for box in boxes :
                box.setbit(im.bitmask,pixmask.getval('INACTIVE_PIXEL'))
            if trimimage :
                xmax=0 
                ymax=0 
                for box,outbox in zip(boxes,outboxes) :
                    xmax = np.max([xmax,outbox.xmax])
                    ymax = np.max([ymax,outbox.ymax])
                z=np.zeros([ymax+1,xmax+1]) 
                out = Data(data=z,uncertainty=z,bitmask=z.astype(np.uintc),
                           unit=u.dimensionless_unscaled,header=im.header)
                for box,outbox in zip(boxes,outboxes) :
                    out.data[outbox.ymin:outbox.ymax+1,outbox.xmin:outbox.xmax+1] =  \
                            im.data[box.ymin:box.ymax+1,box.xmin:box.xmax+1]
                    out.uncertainty.array[outbox.ymin:outbox.ymax+1,outbox.xmin:outbox.xmax+1] = \
                            im.uncertainty.array[box.ymin:box.ymax+1,box.xmin:box.xmax+1]
                    out.bitmask[outbox.ymin:outbox.ymax+1,outbox.xmin:outbox.xmax+1] = \
                            im.bitmask[box.ymin:box.ymax+1,box.xmin:box.xmax+1]
                outim.append(out)
        if trimimage: 
            if len(outim) == 1 : return outim[0]
            else : return outim
        else : return inim
       
    def bias(self,im,superbias=None) :
         """ Superbias subtraction
         """
         # only subtract if we are given a superbias!
         if superbias is None : return im

         # work with lists so that we can handle multi-channel instruments
         if type(im) is not list : ims=[im]
         else : ims = im
         if type(superbias) is not list : superbiases=[superbias]
         else : superbiases = superbias
         out=[]
         for im,bias in zip(ims,superbiases) :
             if self.verbose : print('  subtracting bias...')
             #out.append(ccdproc.subtract_bias(im,bias))
             corr = copy.deepcopy(im)
             corr.data -= bias.data
             corr.uncertainty.array = np.sqrt(corr.uncertainty.array**2+
                                              bias.uncertainty.array**2)
             out.append(corr)
         if len(out) == 1 : return out[0]
         else : return out

    def dark(self,im,superdark=None) :
         """ Superdark subtraction
         """
         # only subtract if we are given a superdark!
         if superdark is None : return im
         if self.inst == 'APOGEE' : return im

         # work with lists so that we can handle multi-channel instruments
         if type(im) is not list : ims=[im]
         else : ims = im
         if type(superdark) is not list : superdarks=[superdark]
         else : superdarks = superdark
         out=[]
         for im,dark in zip(ims,superdarks) :
             if self.verbose : print('  subtracting dark...')
             #out.append(ccdproc.subtract_dark(im,dark,exposure_time='EXPTIME',exposure_unit=u.s))
             corr = copy.deepcopy(im)
             exptime = corr.header['EXPTIME']
             corr.data -= dark.data*exptime
             corr.uncertainty.array = np.sqrt(corr.uncertainty.array**2+
                                              exptime**2*dark.uncertainty.array**2)
             out.append(corr)
         if len(out) == 1 : return out[0]
         else : return out

    def flat(self,im,superflat=None,display=None) :
         """ Flat fielding
         """
         # only flatfield if we are given a superflat!
         if superflat is None : return im

         if type(im) is not list : ims=[im]
         else : ims = im
         if type(superflat) is not list : superflats=[superflat]
         else : superflats = superflat
         out=[]
         for im,flat in zip(ims,superflats) :
             if self.verbose : print('  flat fielding...')
             if display is not None : 
                 display.tv(im)
             #corr = ccdproc.flat_correct(im,flat)
             corr = copy.deepcopy(im)
             corr.data /= flat.data
             corr.uncertainty.array /= flat.data
             out.append(corr)
             if display is not None : 
                 display.tv(corr)
                 #plot central crossections
                 display.plotax2.cla()
                 dim=corr.data.shape
                 col = int(dim[1]/2)
                 row = corr.data[:,col]
                 display.plotax2.plot(row)
                 min,max=tv.minmax(row,low=5,high=5)
                 display.plotax2.set_ylim(min,max)
                 display.plotax2.set_xlabel('row')
                 display.plotax2.text(0.05,0.95,'Column {:d}'.format(col),
                     transform=display.plotax2.transAxes)
                 #display.plotax2.cla()
                 row = int(dim[0]/2)
                 col = corr.data[row,:]
                 min,max=tv.minmax(col,low=10,high=10)
                 display.plotax2.plot(col)
                 display.plotax2.set_xlabel('col')
                 display.plotax2.text(0.05,0.95,'Row {:d}'.format(row),
                     transform=display.plotax2.transAxes)
                 display.plotax2.set_ylim(min,max)
                 getinput("  See flat-fielded image and original with - (minus) key.",display)
         if len(out) == 1 : return out[0]
         else : return out


    def scatter(self,inim,scat=None,display=None,smooth=3,smooth2d=31,transpose=False) :
        """ Removal of scattered light (for multi-order/object spectrograph)

            Remove scattered light by looking for valleys in cross-sections across traces
            and fitting a 2D surface to the low points. Cross-sections are smoothed before
            finding the valleys, and interpolated surface is also smoothed. Some attempt
            is made to reject outliers before fitting the final surface, which is subtracted
            from the image.
 
            Parameters
            ----------
            im : Data object
                 input image to correct
            transpose : bool, default=False
                 set to true if spectra run along columns
            scat : integer, default=None
                 get scattered light measurements every scat pixels. If None, no correction
            display : TV object, default=None
                 if set, show the scattered light measurements
            smooth : integer, default=3
                 boxcar width for smoothing profile perpendicular to traces before looking
                 for valleys
            smooth2d : integer, default=31
                 boxcar width for smoothing interpolated scattered light surface
        """
        if scat is None : return
        if transpose :
            im = Data(data=inim.data.T)
        else :
            im = inim

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
        boxcar = Box2DKernel(smooth2d)
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

        if display is not None :
            display.clear()
            display.tv(im)
            points=np.array(points)
            display.ax.scatter(points[:,1],points[:,0],color='r',s=3)
            points_gd=np.array(points_gd)
            display.ax.scatter(points_gd[:,1],points_gd[:,0],color='g',s=3)
            getinput("  See image with scattered light points",display)
            display.clear()
            display.tv(im)
            display.tv(grid_z)
            col=int(im.shape[-1]/2)
            display.plotax2.cla()
            display.plotax2.plot(im.data[:,col])
            display.plotax2.plot(grid_z[:,col])
            plt.draw()
            getinput("  See scattered light image",display)

        if transpose :
            im.data -= grid_z.T
        else :
            im.data -= grid_z

    def crrej(self,im,crbox=None,nsig=5,display=None,
              objlim=5.,fsmode='median',inbkg=None) :
        """ Cosmic ray rejection using spatial median filter or lacosmic. 

            If crbox is given as a 2-element list, then a box of this shape is
            run over the image. At each location, the median in the box is determined.
            For each pixel in the box, if the value is larger than nsig*uncertainty
            (where uncertainty is taken from the input.uncertainty.array), the pixel
            is replaced by the median.  The pixel is also flagged in input.bitmask

            If crbox='lacosmic', the LA Cosmic routine, as implemented in ccdproc
            (using astroscrappy) is run on the image, with default options. 
            Other keywords are ignored.

            Parameters
            ----------
            crbox : list, int shape of box to use for median filters, or 'lacosmic'
            nsig  : float, default 5, threshold for CR rejection if using spatial 
                    median filter
            display : None for no display, pyvista TV object to display
        """

        if crbox is None: return im
        if type(im) is not list : ims=[im]
        else : ims = im
        out=[]
        for i,(im,gain,rn) in enumerate(zip(ims,self.gain,self.rn)) :
            if display is not None : 
                display.clear()
                min,max=tv.minmax(im,low=5,high=30)
                display.tv(im.uncertainty.array,min=min,max=max)
                display.tv(im,min=min,max=max)
            if crbox == 'lacosmic':
                if self.verbose : print('  zapping CRs with ccdproc.cosmicray_lacosmic')
                if isinstance(gain,list) : g=1.
                else : g=gain
                outim= ccdproc.cosmicray_lacosmic(im,gain_apply=False,
                         objlim=objlim,fsmode=fsmode,inbkg=inbkg,
                         gain=g*u.dimensionless_unscaled,
                         readnoise=rn*u.dimensionless_unscaled)
                outim.add_bitmask(im.bitmask)
                outim.add_wave(im.wave)
            else :
                if self.verbose : print('  zapping CRs with filter [{:d},{:d}]...'.format(*crbox))
                if crbox[0]%2 == 0 or crbox[1]%2 == 0 :
                    raise ValueError('cosmic ray rejection box dimensions must be odd numbers...')
                if crbox[0]*crbox[1] > 49 :
                    print('WARNING: large rejection box may take a long time to complete!')
                    tmp=input(" Hit c to continue anyway, else quit")
                    if tmp != 'c' : return
                outim=copy.deepcopy(im)
                image.zap(outim,crbox,nsig=nsig)
            if display is not None : 
                display.tv(outim,min=min,max=max)
                display.tv(im.subtract(outim),min=min,max=max)
                getinput("  See CRs and CR-zapped image and original using - key",display)
            crpix = np.where(~np.isclose(im.subtract(outim),0.))
            pixmask = bitmask.PixelBitMask()
            outim.bitmask[crpix] |= pixmask.getval('CRPIX')
            out.append(outim)
        if len(out) == 1 : return out[0]
        else : return out

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

    def platesolve(self,im,scale=0.46,seeing=2,display=None) :
        """ try to get plate solution with imwcs
        """
        if self.verbose : print('  plate solving with local astrometry.net....')

        # find stars
        mad=np.nanmedian(np.abs(im-np.nanmedian(im)))
        daofind=DAOStarFinder(fwhm=seeing/scale,threshold=10*mad)
        objs=daofind(im.data-np.nanmedian(im.data))
        if len(objs) == 0 :
            raise RuntimeError('no stars detected. Maybe try setting seeing?')

        try: objs.sort(['mag'])
        except: pdb.set_trace()
        gd=np.where((objs['xcentroid']>50)&(objs['ycentroid']>50)&
                    (objs['xcentroid']<im.data.shape[1]-50)&
                    (objs['ycentroid']<im.data.shape[0]-50))[0]
        if display is not None :
            display.tv(im)
            objs['x'] = objs['xcentroid']
            objs['y'] = objs['ycentroid']
            stars.mark(display,objs[gd],exit=True)
        tmpfile=tempfile.mkstemp(dir='./')
        objs.write(os.path.basename(tmpfile[1])+'xy.fits')

        # solve with astrometry.net routines
        ra=im.header['RA'].replace(' ',':')
        dec=im.header['DEC'].replace(' ',':')
        rad=15*(float(ra.split(':')[0])+float(ra.split(':')[1])/60.+float(ra.split(':')[2])/3600.)
        decd=(float(dec.split(':')[0])+float(dec.split(':')[1])/60.+float(dec.split(':')[2])/3600.)
        cmd=('/usr/local/astrometry/bin/solve-field'+
            ' --scale-units arcsecperpix --scale-low {:f} --scale-high {:f}'+
            ' -X xcentroid -Y ycentroid -w 4800 -e 3000 --overwrite'+
            ' --ra {:f} --dec {:f} --radius 3 {:s}xy.fits').format(
              .9*scale,1.1*scale,rad,decd,os.path.basename(tmpfile[1]))
        print(cmd)
        ret = subprocess.call(cmd.split())


        """
        cmd='/usr/local/astrometry/bin/new-wcs -i {:s}.fits -w {:s}xy.wcs -o {:s}w.fits'.format(tmpfile[1],os.path.basename(tmpfile[1]),os.path.basename(tmpfile[1]))
        ret = subprocess.call(cmd.split())
        pdb.set_trace()

        im.write(tmpfile[1]+'.fits')
        if flip : 
            arg=''
            objs['xcentroid'] = im.data.shape[1]-1-objs['xcentroid']
        else : arg=''
        objs['xcentroid','ycentroid','mag'][gd].write(
                    tmpfile[1]+'.txt',format='ascii.fast_commented_header')
        cmd=("imcat -c ua2 {:s}.fits".format(tmpfile[1])).split()
        ret = subprocess.check_output(cmd)
        def parse(ret) :
            tmp= ret.split(b'\n')
            x=[]
            y=[]
            for t in tmp[0:-1] :
              l=t.split()
              x.append(float(l[6]))
              y.append(float(l[7]))
            return x,y
        x,y=parse(ret)
        for xx,yy in zip(x,y) :
            display.tvcirc(xx,yy,rad=5,color='r')
        cmd=('imwcs -vw {:s} -d {:s}.txt -c ua2 -j {:s} {:s} -p {:.2f} {:s}.fits'.
                format(arg,tmpfile[1],ra,dec,scale,tmpfile[1])).split()
        cmd=('imwcs -vw {:s} -h 200 -d {:s}.txt -c ua2 -j {:s} {:s} {:s}.fits'.
                format(arg,tmpfile[1],ra,dec,tmpfile[1])).split()
        ret = subprocess.call(cmd)
        print(cmd)
        header=fits.open(os.path.basename(tmpfile[1])+'w.fits')[0].header
        if flip :
            header['CD1_1'] *= -1
            header['CD2_1'] *= -1
        w=WCS(header)
        im.wcs=w
        pdb.set_trace() 
        cmd=("imcat -c ua2 {:s}w.fits".format(os.path.basename(tmpfile[1]))).split()
        ret = subprocess.check_output(cmd)
        x,y=parse(ret)
        for xx,yy in zip(x,y) :
            display.tvcirc(xx,yy,rad=5,color='g')

        """
        if display is not None :
            getinput("  See plate solve stars",display)
            display.tvclear()
        # get WCS
        try:
            header=fits.open(os.path.basename(tmpfile[1])+'xy.wcs')[0].header
            w=WCS(header)
            im.wcs=w
        except :
            for f in glob.glob(os.path.basename(tmpfile[1])+'*') : os.remove(f)
            raise RuntimeError('plate solve FAILED')

        for f in glob.glob(os.path.basename(tmpfile[1])+'*') : os.remove(f)
        return im

    def noise(self,pairs,rows=None,cols=None,nbox=200,display=None) :
        """ Noise characterization from image pairs
        """
        mean=[]
        std=[]
        for pair in pairs :
            a=self.reduce(pair[0])
            b=self.reduce(pair[1])
            diff=a.data-b.data
            avg=(a.data+b.data)/2
            if display != None :
                display.tv(avg)
                display.tv(diff)
            if rows is None : rows=np.arange(0,a.shape[0],nbox)
            if cols is None : cols=np.arange(0,a.shape[1],nbox)
            for irow,r0 in enumerate(rows[0:-1]) :
                for icol,c0 in enumerate(cols[0:-1]) :
                    if display != None :
                        box = image.BOX(xr=[cols[icol],cols[icol+1]],
                                yr=[rows[irow],rows[irow+1]]) 
                        display.tvbox(0,0,box=box)
                    print(r0,c0,box.median(avg),box.stdev(diff))
                    mean.append(box.median(avg))
                    std.append(box.stdev(diff))
        mean=np.array(mean)
        std=np.array(std)
        plt.figure()
        plt.plot(mean,std**2,'ro')


    def display(self,display,id) :

        im = self.reduce(id)
        if type(im) is not list : ims=[im]
        else : ims = im
        for i, im in enumerate(ims) :
            display.tv(im)

    def write(self,im,name,overwrite=True,png=False,wave=None) :
        """ write out image, deal with multiple channels 
        """

        if type(im) is not list : ims=[im]
        else : ims = im
        for i,frame in enumerate(ims) : 
            if self.nchip > 1 : outname = name.replace('.fits','')+'_'+self.channels[i]+'.fits'
            else : outname = name
            frame.write(outname,overwrite=overwrite)
            if png :
                #backend=matplotlib.get_backend()
                #matplotlib.use('Agg')
                if wave is not None :
                    fig=plt.figure(figsize=(18,6))
                    for row in frame.data.size[0] :
                        plt.plot(wave[row],frame.data[row])
                        plt.xlabel('Wavelength')
                        plt.ylabel('Flux')
                else :
                    fig=plt.figure(figsize=(12,9))
                    vmin,vmax=tv.minmax(frame.data)
                    plt.imshow(frame.data,vmin=vmin,vmax=vmax,
                           cmap='Greys_r',interpolation='nearest',origin='lower')
                    plt.colorbar(shrink=0.8)
                    plt.axis('off')
                fig.tight_layout()
                fig.savefig(name.replace('.fits','.png'))
                plt.close()
                #matplotlib.use(backend)
 

    def getcube(self,ims,**kwargs) :
        """ Read images into data cube
        """
        # create list of images, reading and overscan subtracting
        allcube = []
        for im in ims :
            if not isinstance(im,pyvista.dataclass.Data) :
                data = self.reduce(im, **kwargs)
            else :
                data = im
            allcube.append(data)

        # if just one frame, put in 2D list anyway so we can use same code, allcube[nframe][nchip]
        if self.nchip == 1 :
            allcube=[list(i) for i in zip(*[allcube])]

        return allcube

    def sum(self,ims, return_list=False, **kwargs) :
        """ Coadd input images
        """

        return self.combine(ims,type='sum',return_list=return_list,**kwargs)

    def combine(self,ims, normalize=False,display=None,div=True,
                return_list=False, type='median',sigreject=5,**kwargs) :
        """ Combine images from list of images 
        """
        # create list of images, reading and overscan subtracting
        allcube = self.getcube(ims,**kwargs)
        nframe = len(allcube)

        # do the combination
        out=[] 
        for chip in range(self.nchip) :
            datacube = []
            varcube = []
            maskcube = []
            allnorm = []
            for im in range(nframe) :
                if normalize :
                    norm=self.normbox[chip].mean(allcube[im][chip].data)
                    allnorm.append(norm)
                    allcube[im][chip].data /= norm
                    allcube[im][chip].uncertainty.array /= norm
                datacube.append(allcube[im][chip].data)
                varcube.append(allcube[im][chip].uncertainty.array**2)
                maskcube.append(allcube[im][chip].bitmask)
            if type == 'median' :
                if self.verbose: print('  combining data with median....')
                med = np.median(np.array(datacube),axis=0)
                sig = 1.253 * np.sqrt(np.mean(np.array(varcube),axis=0)/nframe)
            elif type == 'mean' :
                if self.verbose: print('  combining data with mean....')
                med = np.mean(np.array(datacube),axis=0)
                sig = np.sqrt(np.sum(np.array(varcube),axis=0)/nframe)
            elif type == 'sum' :
                if self.verbose: print('  combining data with sum....')
                med = np.sum(np.array(datacube),axis=0)
                sig = np.sqrt(np.mean(np.array(varcube),axis=0)/nframe)
            elif type == 'reject' :
                datacube = np.array(datacube)
                if self.verbose: print('  combining data with rejection....')
                med = np.median(datacube,axis=0)
                bd=np.where(datacube>med+sigreject*np.sqrt(np.array(varcube)))
                datacube[bd]=np.nan
                med = np.nanmean(datacube,axis=0)
                sig = np.sqrt(np.nanmean(np.array(varcube),axis=0)/nframe)
            else :
                raise ValueError('no combination type: {:s}'.format(type))
            if self.verbose: print('  calculating uncertainty....')
            #mask = np.any(maskcube,axis=0)
            mask = np.bitwise_or.reduce(maskcube,axis=0)
            comb=Data(med.astype(np.float32),header=allcube[im][chip].header,
                         uncertainty=StdDevUncertainty(sig.astype(np.float32)),
                         bitmask=mask,unit=u.dimensionless_unscaled)
            if normalize: comb.meta['MEANNORM'] = np.array(allnorm).mean()
            out.append(comb)

            # display final combined frame and individual frames relative to combined
            if display :
                display.clear()
                display.tv(comb,sn=True)
                display.tv(comb)
                if comb.mask is not None :
                    gd=np.where(comb.mask == False)
                elif comb.bitmask is not None :
                    pixmask=bitmask.PixelBitMask()
                    gd=np.where((comb.bitmask&pixmask.badval())==0)
                else :
                    gd=np.where(med>0)
 
                min,max=tv.minmax(med[gd[0],gd[1]],low=10,high=10)
                display.plotax2.hist(med[gd[0],gd[1]],bins=np.linspace(min,max,100),histtype='step')
                display.fig.canvas.draw_idle()
                getinput("  See final image, use - key for S/N image.",display)
                for i,im in enumerate(ims) :
                    min,max=tv.minmax(med[gd[0],gd[1]],low=5,high=5)
                    display.fig.canvas.draw_idle()
                    if div :
                        display.plotax2.hist((allcube[i][chip].data/med)[gd[0],gd[1]],
                                            bins=np.linspace(0.5,1.5,100),histtype='step')
                        display.tv(allcube[i][chip].data/med,min=0.5,max=1.5)
                        getinput("    see image: {} divided by master".format(im),display)
                    else :
                        delta=5*self.rn[chip]
                        display.plotax2.hist((allcube[i][chip].data-med)[gd[0],gd[1]],
                                            bins=np.linspace(-delta,delta,100),histtype='step')
                        display.tv(allcube[i][chip].data-med,min=-delta,max=delta)
                        getinput("    see image: {} minus master".format(im),display)

        # return the frame
        if len(out) == 1 :
           if return_list : return [out[0]]
           else : return out[0]
        else : return out

    def mkbias(self,ims,display=None,scat=None,type='median',sigreject=5,
               trim=False) :
        """ Driver for superbias combination (no superbias subtraction no normalization)
        """
        return self.combine(ims,display=display,div=False,scat=scat,trim=trim,
                            type=type,sigreject=sigreject)

    def mkdark(self,ims,bias=None,display=None,scat=None,trim=False,
               type='median',sigreject=5,clip=None) :
        """ Driver for superdark combination (no normalization)
        """
        dark= self.combine(ims,bias=bias,display=display,trim=trim,
                            div=False,scat=scat,type=type,sigreject=sigreject)
        if clip != None:
            low = np.where(dark.data < clip*dark.uncertainty.array)
            dark.data[low] = 0.
            dark.data[low] = 0.
        return dark

    def mkflat(self,ims,bias=None,dark=None,scat=None,display=None,trim=False,
               type='median',sigreject=5,spec=False,width=101) :
        """ Driver for superflat combination 
             (with superbias if specified, normalize to normbox

            Parameters
            ----------
            ims : list of frames to combine
            display : TV object, default= None
                      if specified, displays flat and individual frames/flat for inspection
            bias : CCDData object, default=None
                  if specified, superbias to subtract before combining flats
            dark : CCDData object, default=None
                  if specified, superdark to subtract before combining flats
            scat : 
            type : str, default='median'
                  combine method
            sigreject : float
                  rejection threshold for combine type='reject'
            spec : bool, default=False
                  if True, creates "spectral" flat by taking out wavelength
                  shape
            width : int, default=101
                  window width for removing spectral shape for spec=True

        Returns
        -------
            CCDData object with combined flat
        """
        flat= self.combine(ims,bias=bias,dark=dark,normalize=True,trim=trim,
                 scat=scat,display=display,type=type,sigreject=sigreject)
        if spec :
            return self.mkspecflat(flat,width=width,display=display)
        else :
            return flat

    def mkspecflat(self,flats,width=101,display=None) :
        """ Spectral flat takes out variation along wavelength direction
        """

        if type(flats) is not list : flats=[flats]
        boxcar = Box1DKernel(width)

        sflat=[]
        for flat in flats :
            if self.transpose :
                tmp = image.transpose(flat)
            else :
                tmp = copy.deepcopy(flat)
            nrows=tmp.data.shape[0]
            # limit region for spectral shape to high S/N area (slit width)
            snmed = np.nanmedian(tmp.data/tmp.uncertainty.array,axis=1)
            gdrows = np.where(snmed>50)[0]
            med = convolve(np.nanmedian(tmp[gdrows,:],axis=0),
                           boxcar,boundary='extend')
            if display is not None :
                display.tv(tmp)
                display.plotax2.cla()
                display.plotax2.plot(med)
            for row in range(tmp.data.shape[0]) :
                tmp.data[row,:] /= med
                tmp.uncertainty.array[row,:] /= med
            if display is not None :
                display.tv(tmp,min=0.7,max=1.3)
            if self.transpose :
                sflat.append(image.transpose(tmp))
            else :
                sflat.append(tmp)

        if len(sflat) == 1 :return sflat[0]
        else : return sflat

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

def mkmask(inst=None) :

    if inst == 'ARCES' :
        nrow=2068
        ncol=2128
        badpix = [ image.BOX(yr=[0,2067],xr=[0,130]),      # left side
                   image.BOX(yr=[0,2067],xr=[2000,2127]), # right side
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
                   image.BOX(yr=[1870,1925],xr=[853,853]) ] 
            

    elif inst == 'DIS' :
        nrow=1078
        ncol=2098
        badpix = [ image.BOX(yr=[474,1077],xr=[803,803]),
                   image.BOX(yr=[0,1077],xr=[1196,1196]),
                   image.BOX(yr=[0,1077],xr=[0,0]) ]

    mask = np.zeros([nrow,ncol],dtype=np.int16)
    for box in badpix :
        box.setval(mask,True)

    hdulist=fits.HDUList()
    hdulist.append(fits.PrimaryHDU(mask))
    hdulist.writeto(inst+'_mask.fits',overwrite=True)

    return mask

def getinput(text,display) :
    """ print text, get a key input from display
    """
    print(text)
    print('   To continue, hit space in display window (p for debug) ')
    get = display.tvmark()[0]
    if get == 'i' : code.interact(local=globals())
    elif get == 'p' :
        pdb.set_trace()
    return get
