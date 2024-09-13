import numpy as np
import shutil
import astropy
from photutils import DAOStarFinder
import code
import copy
from astropy import units as u
from astropy.table import Table
from astropy.nddata import StdDevUncertainty
from pyvista.dataclass import Data
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.convolution import convolve, Box1DKernel, Box2DKernel, Box2DKernel
from holtztools import html, plots
import astroscrappy
import scipy.signal
from scipy.optimize import curve_fit
import yaml
import subprocess
import sys
import tempfile
from pyvista import stars, image, tv, bitmask, dataclass, spectra
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
    def __init__(self,inst=None,conf='',dir='./',root='*',formstr='{:04d}.f*',
                 gain=1,rn=0.,saturation=2**32,verbose=True,nfowler=1,
                 trim=False) :
        """  Initialize reducer with information about how to reduce

        Parameters
        ----------
        inst : str, default=None
               configuration file name to read for instrument
        conf : str, default=''
               configuration suffix to add to configuration file name
        dir : str,default='./'
              default directory to get images from
        trim : bool, default=False
               if True, trim calibration products
        formstr : str, default='{04d}.f*'
              sets format string to find images given integer ID, 
              if not reading from configuration file
        gain : float, default=1
              gain if not reading from configuration file
        rn : float, default=0
              rn if not reading from configuration file
        saturation : int, default=2**32
              saturation value if not reading from configuration file
        verbose : bool, default=False
              turn on verbose output
        nfowler : integer, default=1
              nfowler value if not reading from configuration file
        """
        self.dir=dir
        self.root=root
        self.verbose=verbose
        self.badpix=None
        self.scat=None
        self.bitmask=None
        self.transpose=None
        self.trim=trim
        self.scale=1
        self.biastype=-1
        self.biasbox=[]
        self.trimbox=[]
        self.normbox=[]
        self.outbox=[]
        self.headerbox=False
        self.ext='fits'
        self.inst='generic'

        # we will allow for instruments to have multiple channels, so everything goes in lists
        self.channels=['']
        if type(gain) is list : self.gain=gain
        else : self.gain = [gain]
        if type(rn) is list : self.rn=rn
        else : self.rn = [rn]
        if type(saturation) is list : self.saturation=saturation
        else : self.saturation = [saturation]
        if type(formstr) is list : self.formstr=formstr
        else : self.formstr=[formstr]
       
        # Read instrument configuation from YAML configuration file 
        if inst is not None :
            self.inst=inst
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
            for card in ['cols','ext','scale','crbox','headerbox'] :
                try : setattr(self,card,config[card])
                except : setattr(self,card,None)
            try : self.saturation=config['saturation']
            except KeyError: 
                self.saturation = []
                for chan in self.channels: self.saturation.append(2**32)
            try : self.namp=config['namp']
            except KeyError: self.namp = 1
            try :self.transpose=config['transpose']
            except KeyError: self.transpose = False
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


    def log(self,files=None,htmlfile=None,ext=None,hdu=0,channel='', 
            cols=None, display=None,hexdump=False) :
        """ Create chronological image log from file headers in default
            directory.

            If any .csv file exists in the directory, its contents are
            added as a table to htmlfile

        Parameters
        ----------
        htmlfile : str, default=None
                   if specified, write HTML log to htmlfile
        ext : override default extension to search
        cols : array-like, str, specifies which FITS header cards to output,
               default=None, which will use cards
               as defined in the Reducer object, if they have been set
               by configuration file, or, otherwise will use
               ['DATE-OBS','OBJNAME','RA','DEC','EXPTIME']
        display : if not None, specifies tv tool in which to 
               display each image and make png thumbnail, to include
               in htmlfile

        Returns
        -------
        astropy table from FITS headers

        """
        if cols is None :
            if self.cols is not None :
                cols = self.cols
            else :
                cols=['DATE-OBS','OBJNAME','RA','DEC','EXPTIME']
        if ext is None :
            if self.ext is not None :
                ext = self.ext
            else :
                ext='fit*'

        # get list of files from default formstr or as requested by keyword
        if files == None :
            files=glob.glob(self.dir+'/*{:s}*.'.format(channel)+ext)
        elif  num.find('/') >= 0 :
            files=glob.glob('{:s}'.format(files))
        else :
            files=glob.glob(self.dir+'/{:s}'.format(files))

        if len(files) == 0 :
            print('no files found matching: ',
                  self.dir+'/*{:s}*.'.format(channel)+ext)

        date=[]
        objs=[]
        for file in files :
          try :
              if hexdump : header=_get_meta(file,keys=cols)
              else : header=fits.open(file)[hdu].header
          except FileNotFound :
              print('error opening file: {:s}, hdu: {:d}'.format(file,hdu))
              return
          try :
              date.append(header['DATE-OBS'])
          except KeyError :
              print('file {:s} does not have DATE-OBS'.format(file))
              date.append('')
          for col in cols :
              try: 
                  if 'OBJ' in col : objs.append(header[col])
              except: 
                  objs.append('?')
        date=np.array(date)
        sort=np.argsort(date)

        if htmlfile is not None :
            fp=html.head(htmlfile+'.html')
            fp.write('Objects observed: ')
            for obj in set(objs) : fp.write('{:s} '.format(obj))
            fp.write('\n<p>\n')
            fp.write('new objects in light blue, new filter in light green') 
            fp.write('<TABLE BORDER=2>\n')
            if display is not None :
                fp.write('<br> <a href={:s}_thumb.html> Thumbnails  page</a>'.format(
                         os.path.basename(htmlfile)))
                fd=html.head(htmlfile+'_thumb.html')
                fd.write('<TABLE BORDER=2>\n')

        # get names and dtypes for table, based on which requested cards are in last header
        names=['FILE']
        dtypes=['S24']
        if htmlfile is not None : fp.write('<TR style="background-color:lightred"><TD>FILE\n')
        for col in cols :
            names.append(col)
            dtypes.append('S16')
            if htmlfile is not None :
                fp.write('<TD>{:s}\n'.format(col))
        tab=Table(names=names,dtype=dtypes)

        # set up style for rows with new object
        newobj= ''
        newfilt= ''
        for col in cols :
            if 'OBJ' in col :
                newobj='style="background-color:lightgreen"'
            if 'FILT' in col :
                newfilt='style="background-color:lightblue"'

        oldobj = ''
        oldfilt = ''
        style = ''
        for i in sort :
          if hexdump : header=_get_meta(file,keys=cols)
          else : header=fits.open(files[i])[hdu].header
          # fix for RA/DEC for MaximDL headers
          if 'RA' not in header  :
              try: header['RA'] = header['OBJCTRA'].replace(' ',':')
              except : pass
          if 'DEC' not in header  :
              try: header['DEC'] = header['OBJCTDEC'].replace(' ',':')
              except : pass
          # if we have OBJECT card, we can color rows for new object
          for col in cols :
            if 'OBJ' in col :
                try :
                    if header[col] != oldobj : 
                        style=newobj
                        oldobj=header[col]
                    else : style=''
                except : pass
          # if we have FILTER card, we can color rows for new filter (if not new object)
          for col in cols :
            if 'FILT' in col :
                try :
                    if header[col] != oldfilt :
                        oldfilt=header[col]
                        if style == '' : style=newfilt
                except : pass
          if htmlfile is not None :
              fp.write('<TR {:s}><TD>{:s}\n'.format(style,os.path.basename(files[i])))  
          row=[os.path.basename(files[i])]
          for col in cols :
            try:
              row.append(str(header[col]))
              if htmlfile is not None:
                  fp.write('<TD>{:s}\n'.format(str(header[col])))
            except: 
              row.append('')
              if htmlfile is not None: fp.write('<TD>\n')
          tab.add_row(row)
          if display is not None : 
              if not os.path.exists(files[i]+'.png') :
                  a=fits.open(files[i])[hdu].data
                  display.tv(a)
                  display.savefig(files[i]+'.png')
              fp.write('<TD><A HREF="{:s}">png image</A>\n'.
                      format(
                      os.path.basename(files[i]+'.png')))
              fd.write('<TR><TD>{:s}<TD><A HREF="{:s}"><IMG SRC="{:s}" WIDTH=400></A>\n'.
                      format(os.path.basename(files[i]),
                      os.path.basename(files[i]+'.png'),
                      os.path.basename(files[i]+'.png')))
        if htmlfile is not None :
            fp.write('</TABLE>\n')
            if display is not None : fd.write('</TABLE>')

        files=glob.glob(self.dir+'/*csv')
        for file in files :
            log=ascii.read(file)
            if htmlfile is not None :
                fp.write('<h3>{:s}'.format(file))
                fp.write('<TABLE BORDER=2>\n')
                for row in log :
                  fp.write('<TR>')
                  for col in row :
                    fp.write('<TD>')
                    fp.write(str(col))
                fp.write('</TABLE>\n')
        
        if htmlfile is not None : html.tail(fp)

        return tab

    def movie(self,ims,display=None,out='movie.gif',channel=0,min=None, max=None) :
        """
        Create animated gif of images

        Parameters
        ----------
        ims : list of image numbers.
        display : pyvista TV object
        out : output file name
        """
        if display == None :
            raise ValueError('you must specify a pyvista TV object with display=')

        files = []
        for im in ims :
            display.clear()
            a=self.rd(im,channel=channel)
            display.tv(a,min=min,max=max,draw=False)
            y,x=a.data.shape
            try: 
                display.tvtext(x//2,y*3//4,'{:d} {:f}'.format(im,a.header['EXPTIME']),color='r')
                display.savefig('tmpimage{:d}.png'.format(im))
                files.append('tmpimage{:d}.png'.format(im))
            except:
                print('error at file: ', im,' stopping there')
                continue

        import imageio
        with imageio.get_writer(out,mode='I') as writer :
            for filename in files :
                image = imageio.imread(filename)
                os.remove(filename)
                writer.append_data(image)

    def reduce(self,num,channel=None,ext=0,
               crbox=None,crsig=5,objlim=5,sigfrac=0.3,
               bias=None,dark=None,flat=None,
               scat=None,badpix=None,solve=False,return_list=False,display=None,
               trim=True,seeing=2,utr=False) :
        """ Reads data from disk, and performs reduction steps 
            as determined from command line parameters

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
             various reduction steps are taken
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
        scat : integer, default=None
             if specified, do scattered light correction, getting
             estimate every scat pixels
        badpix : int, default=None
             if specified, set masked pixels to specified value
        trim : bool, default=True
             trim image after calibration, irrelevant if red.trimg=True
        solve : bool, default=False
             attempt to plate-solve image after reduction, requires 
             local astrometry.net
        seeing : float, default=2
             seeing used to find stars if solve=True

        """
        im=self.rd(num,dark=dark,channel=channel,utr=utr,ext=ext)
        self.overscan(im,display=display,channel=channel)
        if self.trim : im=self.trimimage(im,trimimage=self.trim)
        im=self.bias(im,superbias=bias)
        im=self.dark(im,superdark=dark)
        im=self.crrej(im,crbox=crbox,crsig=crsig,objlim=objlim,sigfrac=sigfrac,
                      display=display)
        self.scatter(im,scat=scat,display=display)
        im=self.flat(im,superflat=flat,display=display)
        self.badpix_fix(im,val=badpix)
        if trim and display is not None: display.tvclear()
        if trim and not self.trim : im=self.trimimage(im,trimimage=trim)
        if solve : 
            im=self.platesolve(im,display=display,scale=self.scale,seeing=seeing)
        if return_list and type(im) is not list : im=[im]
        return im

    def rd(self,num, ext=0, dark=None, channel=None, utr=False) :
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

            # find the files that match the directory/format
            if isinstance(num,(int,np.int64)) :
                search=self.dir+'/'+self.root+form.format(num)
            elif type(num) is str or type(num) is np.str_ :
                if num.find('/') >= 0 :
                    search=num
                else :
                    search=self.dir+'/'+num
            else :
                print('stopping in rd... num:',num)
                pdb.set_trace()
            file=glob.glob(search)
            if len(file) == 0 : 
                file=glob.glob(search+'.gz')
                if len(file) == 0 :
                    raise ValueError('cannot find file matching: '+search,num)
                    return
            elif len(file) > 1 : 
                if self.verbose : print('more than one match found, using first!',file)
            file=file[0]

            # read the file into a Data object
            if self.verbose : print('  Reading file: {:s}'.format(file)) 
            if 'APOGEE' in self.inst :
                if utr : im=apogee.utr(file,dark=dark)
                else : im=apogee.cds(file,dark=dark)
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
            # fix for RA/DEC for MaximDL headers
            if 'RA' not in im.header  :
                try: im.header['RA'] = im.header['OBJCTRA']
                except : print('no RA or OBJCTRA found in {:s}'.format(file))
            if 'DEC' not in im.header  :
                try: im.header['DEC'] = im.header['OBJCTDEC']
                except : print('no DEC or OBJCTDEC found in {:s}'.format(file))

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
        if self.biastype < 0 : return im

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
        if self.headerbox : self.boxfromheader(im,2,2)

        for ichan,(im,gain,rn,biasbox,biasregion) in \
                enumerate(zip(ims,gains,rns,biasboxes,biasregions)) :
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
                over2d=image.stretch(over,ncol=databox.ncol())
                if self.verbose: print('  subtracting overscan vector ')
                im.data[databox.ymin:databox.ymax+1,
                        databox.xmin:databox.xmax+1] = \
                    im.data[databox.ymin:databox.ymax+1,
                            databox.xmin:databox.xmax+1].astype(np.float32) - over2d
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
                getinput("  See bias box (solid outlines applied to dashed regions of the same color), and cross section. ",display)

            # Add uncertainty (redo from scratch after overscan)
            data=copy.copy(im.data)
            data[data<0] = 0.
            if isinstance(gain,list) :
              im.uncertainty = StdDevUncertainty(np.sqrt( data + rn**2 ))
            else :
              im.uncertainty = StdDevUncertainty(np.sqrt( data/gain + (rn/gain)**2 ))

    def trimimage(self,inim,trimimage=False) :
        """ Trim image by masking non-trimmed pixels
            May need to preserve image size to match reference/calibration frames, etc.
        """
        if type(inim) is not list : ims=[inim]
        else : ims = inim

        if self.headerbox : self.boxfromheader(inim,2,2)
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
       
    def boxfromheader(self,im,nx,ny) :
        """ Set boxes from image header
        """
        i=0
        xs=0
        for ix in range(nx) :
            ys=0
            for iy in range(ny) :
                dsec=im.header['BSEC{:d}{:d}'.format(ix+1,iy+1)]
                out=dsec.strip('[').strip(']').replace(',',' ').replace(':',' ').split() 
                xmin,xmax,ymin,ymax=[int(i) for i in out]
                self.biasbox[0][i].set(xmin-1,xmax-1,ymin-1,ymax-1)

                dsec=im.header['DSEC{:d}{:d}'.format(ix+1,iy+1)]
                out=dsec.strip('[').strip(']').replace(',',' ').replace(':',' ').split() 
                xmin,xmax,ymin,ymax=[int(i) for i in out]
                self.biasregion[0][i].set(xmin-1,xmax-1,ymin-1,ymax-1)
                self.trimbox[0][i].set(xmin-1,xmax-1,ymin-1,ymax-1)

                self.outbox[0][i].set(xs,xs+xmax-xmin,ys,ys+ymax-ymin)
                ys+=(ymax-ymin+1)
                i+=1
            xs+=(xmax-xmin+1)

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
             corr = copy.deepcopy(im)
             exptime = corr.header['EXPTIME']
             dark_exptime = dark.header['EXPTIME']
             corr.data -= dark.data/dark_exptime*exptime
             corr.uncertainty.array = np.sqrt(corr.uncertainty.array**2+
                          exptime**2*(dark.uncertainty.array/dark_exptime)**2)
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
             corr = copy.deepcopy(im)
             corr.data /= flat.data
             corr.uncertainty.array /= flat.data
             out.append(corr)
             if display is not None : 
                 display.tv(corr,same=True)
                 #plot central crossections
                 display.plotax2.cla()
                 dim=corr.data.shape
                 col = int(dim[1]/2)
                 row = corr.data[:,col]
                 display.plotax2.plot(row)
                 min,max=image.minmax(row,low=5,high=5)
                 display.plotax2.set_ylim(min,max)
                 display.plotax2.set_xlabel('row')
                 display.plotax2.text(0.05,0.95,'Column {:d}'.format(col),
                     transform=display.plotax2.transAxes)
                 #display.plotax2.cla()
                 row = int(dim[0]/2)
                 col = corr.data[row,:]
                 min,max=image.minmax(col,low=10,high=10)
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

    def crrej(self,im,crbox=None,crsig=5,display=None,
              objlim=5.,sigfrac=0.3,fsmode='median',inbkg=None) :
        """ Cosmic ray rejection using spatial median filter or lacosmic. 

        If crbox is given as a 2-element list, then a box of this shape is
        run over the image. At each location, the median in the box is determined.
        For each pixel in the box, if the value is larger than crsig*uncertainty
        (where uncertainty is taken from the input.uncertainty.array), the pixel
        is replaced by the median.  If crsig is a list, then multiple passes
        are done with successive values of crsig (which should then be decreasing),
        but only neighbors of previously flagged CRs are tested. 
        Affected pixels are flagged in input.bitmask

        If crbox='lacosmic', the LA Cosmic routine, as implemented in 
        astroscrappy is run on the image, with default options,
        but objlim, fsmode, and inbkg can be specified.

        Parameters
        ----------
        crbox : list, int shape of box to use for median filters, or 'lacosmic'
        crsig  : list/float, default 5, threshold for CR rejection if using spatial 
                median filter; if list, do multiple passes, with all passes after
                first only on neighbors of previously flagged CRs
        objlim  : for LAcosmic, default=5
        fsmod  : for LAcosmic, default='median'
        inbkg  : for LAcosmic, default=None
        display : None for no display, pyvista TV object to display
        """

        if crbox is None: return im
        if type(im) is not list : ims=[im]
        else : ims = im
        if type(crsig) is not list : nsigs=[crsig]
        else : nsigs = crsig
        out=[]
        for i,(im,gain,rn,sat) in enumerate(zip(ims,self.gain,self.rn,self.saturation)) :
          print('  starting CR rejection, may take some time ....')
          for iter,nsig in enumerate(nsigs) : 
            if display is not None : 
                display.clear()
                min,max=image.minmax(im,low=5,high=30)
                display.tv(im.uncertainty.array,min=min,max=max)
                display.tv(im,min=min,max=max)
            if crbox == 'lacosmic':
                if self.verbose : 
                    print('  zapping CRs with astroscrappy detect_cosmics')
                if isinstance(gain,list) : g=1.
                else : g=gain

                outim =copy.deepcopy(im)
                crmask,outim.data =astroscrappy.detect_cosmics(im.data, 
                          sigclip=crsig, sigfrac=sigfrac, objlim=objlim,
                          gain=g, readnoise=rn, satlevel=sat,
                          fsmode=fsmode) 
 
            else :
                if self.verbose : 
                    print('  Iteration {:d}, zapping CRs with filter [{:d},{:d}]...'.format(iter, *crbox))
                if crbox[0]%2 == 0 or crbox[1]%2 == 0 :
                    raise ValueError('cosmic ray rejection box dimensions must be odd numbers...')
                if crbox[0]*crbox[1] > 49 :
                    print('WARNING: large rejection box may take a long time to complete!')
                    tmp=input(" Hit c to continue anyway, else quit")
                    if tmp != 'c' : return
                if iter == 0 : outim=copy.deepcopy(im)
                image.zap(outim,crbox,nsig=nsig)
                if iter > 0 :
                    # if not first iteration, only allow changes to neighbors of CRs
                    mask = np.where( image.smooth
                            (outim.bitmask&pixmask.getval('CRPIX'),[3,3]) == 0)
                    outim.data[mask[0],mask[1]] = im.data[mask[0],mask[1]]
            if display is not None : 
                display.tv(outim,min=min,max=max)
                display.tv(im.subtract(outim),min=min,max=max)
                getinput("  See CRs and CR-zapped image and original using - key",display)
            crpix = np.where(~np.isclose(im.data-outim.data,0.))
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

    def platesolve(self,im,scale=0.46,seeing=2,display=None,thresh=10) :
        """ try to get plate solution with astrometry.net
        """
        if self.verbose : print('  plate solving with local astrometry.net....')

        # find stars
        mad=np.nanmedian(np.abs(im-np.nanmedian(im)))
        daofind=DAOStarFinder(fwhm=seeing/scale,threshold=thresh*mad)
        objs=daofind(im.data-np.nanmedian(im.data))
        if len(objs) == 0 :
            raise RuntimeError('no stars detected. Maybe try setting seeing?')
        else : print('found ',len(objs),' objects ')

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
        cmdname=shutil.which('solve-field')
        if cmdname is None :
            cmdname=shutil.which('/usr/local/astrometry/bin/solve-field')
        if cmdname is None :
            print('cannot find local astrometry.net solve-field routine')
            pdb.set_trace()

        cmd=(cmdname+
            ' --scale-units arcsecperpix --scale-low {:f} --scale-high {:f}'+
            ' -X xcentroid -Y ycentroid -w 4800 -e 3000 --overwrite'+
            ' --ra {:f} --dec {:f} --radius 3 {:s}xy.fits').format(
              .9*scale,1.1*scale,rad,decd,os.path.basename(tmpfile[1]))
        print(cmd)
        ret = subprocess.call(cmd.split())

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

    def noise(self,pairs,rows=None,cols=None,nbox=200,display=None,channel=None,levels=None,skip=1) :
        """ Noise characterization from image pairs
        """
        title=''
        fig,ax=plots.multi(1,3,hspace=0.001,sharex=True)
        colors=['r','g','b','c','m','y','k']
        for icolor,pair in enumerate(pairs) :
            mean=[]
            std=[]
            n=[]
            title+='[{:s},{:s}]'.format(str(pair[0]),str(pair[1]))
            a=self.reduce(pair[0],channel=channel)
            b=self.reduce(pair[1],channel=channel)
            diff=a.data[::skip,::skip]-b.data[::skip,::skip]
            avg=(a.data[::skip,::skip]+b.data[::skip,::skip])/2
            if display != None :
                display.tv(avg)
                display.tv(diff)
            if levels is not None :
                for i,level in enumerate(levels[0:-1]) :
                    j=np.where((avg.flatten() > level) & (avg.flatten() <= levels[i+1]))[0]
                    if len(j) > 100 :
                        mean.append((level+levels[i+1])/2.)
                        std0 = diff.flatten()[j].std()
                        gd = np.where(np.abs(diff.flatten()[j]) < 5*std0)[0]
                        std.append(diff.flatten()[j[gd]].std())
                        n.append(len(j))
                        print((level+levels[i+1])/2.,diff.flatten()[j].std(),len(j))
            else :
              if rows is None : rows=np.arange(0,a.shape[0],nbox)
              if cols is None : cols=np.arange(0,a.shape[1],nbox)
              for irow,r0 in enumerate(rows[0:-1]) :
                for icol,c0 in enumerate(cols[0:-1]) :
                    box = image.BOX(xr=[cols[icol],cols[icol+1]],
                            yr=[rows[irow],rows[irow+1]]) 
                    if display != None :
                        display.tvbox(0,0,box=box)
                    print(r0,c0,box.median(avg),box.stdev(diff))
                    mean.append(box.median(avg))
                    std.append(box.stdev(diff))
                    n.append((cols[icol+1]-cols[icol])*(rows[irow+1]-rows[irow]))
            mean=np.array(mean)
            std=np.array(std)
            n=np.array(n)
            plots.plotp(ax[0],mean,std**2,yt='$\sigma^2$',size=30,color=colors[icolor])
            plots.plotp(ax[1],mean,2*mean/std**2,yt='G = 2 C / $\sigma^2$',size=20,color=colors[icolor])
            plots.plotp(ax[2],mean,np.log10(n),xt='counts (C)',yt='log(Npix)',size=20,color=colors[icolor])
        fig.suptitle(title+' channel: {:d}'.format(channel))

        pdb.set_trace()


    def display(self,display,id) :
        """ Reduce and display image
        """
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
                    vmin,vmax=image.minmax(frame.data)
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
            if isinstance(im,dataclass.Data) :
                data = im
            elif isinstance(im,list) :
                #multi-channel instrument
                data = im
            else :
                data = self.reduce(im, **kwargs)
            allcube.append(data)

        # if just one frame, put in 2D list anyway so we can use same code, allcube[nframe][nchip]
        if self.nchip == 1 :
            allcube=[list(i) for i in zip(*[allcube])]

        return allcube

    def sum(self,ims, return_list=False, **kwargs) :
        """ Coadd input images
        """

        return self.combine(ims,type='sum',return_list=return_list,**kwargs)

    def combine(self,ims,normalize=False,display=None,div=True,
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
                    try:
                        norm=self.normbox[chip].mean(allcube[im][chip].data)
                    except:
                        norm=np.nanmean(allcube[im][chip].data)
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
                for i,f in enumerate(comb) :
                    comb.header['OBJECT'] = 'Combined frame'
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
 
                min,max=image.minmax(med[gd[0],gd[1]],low=10,high=10)
                display.plotax2.hist(med[gd[0],gd[1]],
                       bins=np.linspace(min,max,100),histtype='step')
                display.fig.canvas.draw_idle()
                getinput("  See final image, use - key for S/N image.",display)
                for i,im in enumerate(ims) :
                    min,max=image.minmax(med[gd[0],gd[1]],low=5,high=5)
                    display.fig.canvas.draw_idle()
                    if div :
                        display.plotax2.hist((allcube[i][chip].data/med)[gd[0],gd[1]],
                                            bins=np.linspace(0.8,1.2,100),histtype='step')
                        display.tv(allcube[i][chip].data/med,min=0.8,max=1.2,
                                   object='{} / master'.format(im))
                        getinput("    see image: {} divided by master".format(allcube[i][chip].header['FILE']),display)
                    else :
                        delta=5*self.rn[chip]
                        display.plotax2.hist((allcube[i][chip].data-med)[gd[0],gd[1]],
                                            bins=np.linspace(-delta,delta,100),histtype='step')
                        display.tv(allcube[i][chip].data-med,min=-delta,max=delta,
                                   object='{} - master'.format(im))
                        getinput("    see image: {} minus master".format(allcube[i][chip].header['FILE']),display)

        # return the frame
        if len(out) == 1 :
           if return_list : return [out[0]]
           else : return out[0]
        else : return out

    def mkbias(self,ims,display=None,scat=None,type='median',sigreject=5,
               trim=False) :
        """ Driver for superbias combination (no superbias subtraction no normalization)

        ims : list of frames to combine
        display : TV object, default= None
                  if specified, displays bias and individual frames-bias for inspection
        type : str, default='median'
              combine method
        sigreject : float
              rejection threshold for combine type='reject', otherwise ignored
        """

        bias= self.combine(ims,display=display,div=False,scat=scat,trim=trim,
                            type=type,sigreject=sigreject)
        for i,f in enumerate(bias) :
            bias[i].header['OBJECT'] = 'Combined bias'
        return bias

    def mkdark(self,ims,ext=0,bias=None,display=None,scat=None,trim=False,
               type='median',sigreject=5,clip=None) :
        """ Driver for superdark combination (no normalization)

        Parameters
        ----------
        ims : list of frames to combine
        display : TV object, default= None
                  if specified, displays dark and individual frames-dark for inspection
        bias : Data object, default=None
              if specified, superbias to subtract before combining darks
        type : str, default='median'
              combine method
        sigreject : float
              rejection threshold for combine type='reject', otherwise ignored
        clip : float, default=None
              if specified, set all values in output dark < clip*uncertainty to zero in master dark
        """

        dark= self.combine(ims,ext=ext,bias=bias,display=display,trim=trim,
                            div=False,scat=scat,type=type,sigreject=sigreject)
        for i,f in enumerate(dark) :
            dark[i].header['OBJECT'] = 'Combined dark'
        if clip != None:
            low = np.where(dark.data < clip*dark.uncertainty.array)
            dark.data[low] = 0.
           
        return dark

    def mkflat(self,ims,bias=None,dark=None,scat=None,display=None,trim=False,ext=0,
               type='median',sigreject=5,spec=False,width=101,littrow=False,normalize=True,
               snmin=50,clip=None) :
        """ Driver for superflat combination, with superbias if specified, normalize to normbox

        Parameters
        ----------
        ims : list of frames to combine
        display : TV object, default= None
                  if specified, displays flat and individual frames/flat for inspection
        bias : Data object, default=None
              if specified, superbias to subtract before combining flats
        dark : Data object, default=None
              if specified, superdark to subtract before combining flats
        scat : 
        type : str, default='median'
              combine method
        sigreject : float
              rejection threshold for combine type='reject', otherwise ignored
        spec : bool, default=False
              if True, creates "spectral" flat by taking out wavelength
              shape
        littrow : bool, default=False
              if True, attempts to fit and remove Littrow ghost from flat,
              LITTROW_GHOST bit must be set in bitmask first to identify 
              ghost location. Ignored if spec==False
        width : int, default=101
              window width for removing spectral shape for spec=True

        Returns
        -------
            Data object with combined flat
        """
        flat= self.combine(ims,ext=ext,bias=bias,dark=dark,normalize=normalize,trim=trim,
                 scat=scat,display=display,type=type,sigreject=sigreject)
        for i,f in enumerate(flat) :
            flat[i].header['OBJECT'] = 'Combined flat'
        if spec :
            flat = self.mkspecflat(flat,width=width,display=display,
                                   littrow=littrow,snmin=snmin)
        if clip is not None :
            low = np.where(flat.data < clip*flat.uncertainty.array)
            flat.data[low] = 1.
            flat.uncertainty.array[low] = 1.

        return flat

    def mkspecflat(self,flats,width=101,display=None,littrow=False,snmin=50) :
        """ Spectral flat takes out variation along wavelength direction
        """

        if type(flats) is not list : flats=[flats]
        boxcar = Box1DKernel(width)

        sflat=[]
        for flat in flats :
            if self.transpose :
                tmp = dataclass.transpose(flat)
            else :
                tmp = copy.deepcopy(flat)
            nrows=tmp.data.shape[0]
            # subtract Littrow ghost
            if littrow :
                print('   fitting/subtracting Littrow ghost')
                pixmask=bitmask.PixelBitMask()
                nmed2=10
                fixed = copy.deepcopy(tmp.data)
                for i in np.arange(nmed2,tmp.shape[0]-nmed2) :
                    try :
                        pix=np.where((tmp.bitmask[i]&pixmask.getval('LITTROW_GHOST')) > 0)[0]
                        yy=np.median(tmp.data[i-nmed2:i+nmed2,pix.min()-10:pix.max()+10],axis=0)
                        xx=np.arange(yy.size)
                        p0 = [.05,(pix.max()-pix.min())/2+10,(pix.max()-pix.min())/4.,1.,0.]
                        coeffs,var = curve_fit(spectra.gauss,xx,yy,p0)
                        xx=np.arange(tmp.data.shape[1])
                        if coeffs[3] > 0.5 :
                            coeffs[3] = 0.
                            coeffs[4] = 0.
                            coeffs[1] += pix.min()-10
                            fixed[i] -= spectra.gauss(xx,*coeffs)
                    except: pass
                tmp.data = fixed

            # limit region for spectral shape to high S/N area (slit width)
            snmed = np.nanpercentile(tmp.data/tmp.uncertainty.array,[90],axis=1)
            gdrows = np.where(snmed[0]>snmin)[0]
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
                sflat.append(dataclass.transpose(tmp))
            else :
                sflat.append(tmp)

        if len(sflat) == 1 :return sflat[0]
        else : return sflat

class DET() :
    """ 
    Defines detector class 
    """
    def __init__(self,inst=None,gain=0.,rn=0.,biastype=0,biasbox=None,
                 normbox=None,trimbox=None,formstr='{:04d}') :
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


def _parse_hexdump_headers(output, keys, default=""):
    """ Parse character string from file into keyword,value pairs
    """
    meta = [default] * len(keys)
    for line in output:
        try:
            key, value = line.split("=", 2)
        except ValueError: # grep'd something in the data
            continue

        key = key.strip()
        if key in keys:
            index = keys.index(key)
            if "/" in value:
                # could be comment
                *parts, comment = value.split("/")
                value = "/".join(parts)
            value = value.strip("' ")
            meta[index] = value.strip()
    return meta

def _get_meta(path, keys=['DATE-OBS'], head=20_000):
    """ Read fits headers faster!!
    """
    keys_str = "|".join(keys)
    if '.gz' in path : 
        commands = " | ".join([
            'zcat {path}','hexdump -n {head} -e \'80/1 "%_p" "\\n"\'' ,
            'egrep "{keys_str}"'
        ]).format(head=head, path=path, keys_str=keys_str)
    else : 
        commands = " | ".join([
            'hexdump -n {head} -e \'80/1 "%_p" "\\n"\' {path}' ,
            'egrep "{keys_str}"'
        ]).format(head=head, path=path, keys_str=keys_str)
        
    outputs = subprocess.check_output(commands, shell=True, text=True)
    outputs = outputs.strip().split("\n")
    values = _parse_hexdump_headers(outputs, keys)
    headers = dict()
    headers.update(dict(zip(keys, values)))
    return headers

