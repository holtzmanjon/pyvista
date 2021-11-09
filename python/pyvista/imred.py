import numpy as np
import astropy
from photutils import DAOStarFinder
import code
import copy
from astropy import units as u
from astropy.nddata import CCDData, NDData, StdDevUncertainty
from astropy.nddata import NDData
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
from pyvista import stars

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import matplotlib
import matplotlib.pyplot as plt
import glob
import bz2
import os
import pdb
from pyvista import image
from pyvista import tv
try: 
    import pyds9
except:
    print('pyds9 is not available, proceeding')

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/../../'



class Reducer() :
    """ Class for reducing images of a given instrument
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
        self.mask=None
        self.transpose=None

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
                config = yaml.load(open(ROOT+'/data/'+inst+'/'+inst+
                              conf+'.yml','r'), Loader=yaml.FullLoader)
            else :
                config = yaml.load(open(inst+'.yml','r'), 
                              Loader=yaml.FullLoader)
            self.channels=config['channels']
            self.formstr=config['formstr']
            self.gain=config['gain']
            self.rn=config['rn']/np.sqrt(nfowler)
            try :self.scale=config['scale']
            except : self.scale = None
            try : self.flip=config['flip']
            except : self.flip = None
            try : self.namp=config['namp']
            except : self.namp = 1
            try :self.transpose=config['transpose']
            except : self.scale = False
            try : self.crbox=config['crbox']
            except : self.crbox=None
            self.biastype=config['biastype']
            try : self.biasavg=config['biasavg']
            except : self.biasavg=11
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
            except: self.biasregion=[None]
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
            try: self.mask=fits.open(ROOT+'/data/'+inst+'/'+
                                inst+'_mask.fits')[0].data.astype(bool)
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
            print('         scale:  {}   flip: {}'.format(self.scale,self.flip))
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
                search=self.dir+'/'+self.root+form.format(num)+'.f*'
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
                print('cannot find file matching: '+search)
                return
            elif len(file) > 1 : 
                if self.verbose : print('more than one match found, using first!',file)
            file=file[0]

            # read the file into a CCDData object
            if self.verbose : print('  Reading file: {:s}'.format(file)) 
            try : im=CCDData.read(file,hdu=ext,unit=u.dimensionless_unscaled)
            except : raise RuntimeError('Error reading file: {:s}'.format(file))
            im.header['FILE'] = os.path.basename(file)
            if 'OBJECT' not in im.header :
                try: im.header['OBJECT'] = im.header['OBJNAME']
                except KeyError : im.header['OBJECT'] = im.header['FILE']

            # Add uncertainty (will be in error if there is an overscan, but redo with overscan subraction later)
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
            
    def overscan(self,im,display=None) :
        """ Overscan subtraction
        """
        if self.biastype < 0 : return

        if type(im) is not list : ims=[im]
        else : ims = im
        for ichan,(im,gain,rn,biasbox,biasregion) in enumerate(zip(ims,self.gain,self.rn,self.biasbox,self.biasregion)) :
            if display is not None : 
                display.tv(im)
                if ichan %2 == 0 : ax=display.plotax1
                else : ax=display.plotax2
                ax.cla()
            if self.namp == 1 : 
                ampboxes = [biasbox]
                databoxes = [biasregion]
            else :
                ampboxes = biasbox
                databoxes = biasregion
            im.data = im.data.astype(float)
            for databox,ampbox in zip(databoxes,ampboxes) :
              if display is not None :
                  display.tvbox(0,0,box=ampbox)
                  if type(databox) == image.BOX : 
                      display.tvbox(0,0,box=databox,color='g')
                      ax.plot(np.arange(databox.ymin,databox.ymax),
                          np.mean(im.data[databox.ymin:databox.ymax,
                          ampbox.xmin:ampbox.xmax], axis=1))
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
                                    databox.xmin:databox.xmax+1].astype(float)-b
                else : 
                    im.data = im.data.astype(float)-b
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
                over=image.stretch(over,ncol=databox.ncol())
                if self.verbose: print('  subtracting overscan vector ')
                im.data[databox.ymin:databox.ymax+1,
                        databox.xmin:databox.xmax+1] = \
                    im.data[databox.ymin:databox.ymax+1,
                            databox.xmin:databox.xmax+1].astype(float) - over
            if display is not None :
                display.tv(im)
                get=input("  See bias box and cross section. Hit any key to continue")
                if get == 'i' : code.interact(local=locals())
                elif get == 'q' : sys.exit()
                elif get == 'p' : pdb.set_trace()

            # Add uncertainty (redo from scratch after overscan)
            data=copy.copy(im.data)
            data[data<0] = 0.
            im.uncertainty = StdDevUncertainty(np.sqrt( data/gain + (rn/gain)**2 ))

    def imtranspose(self,im) :
        """ Transpose a CCDData object
        """
        return CCDData(im.data.T,header=im.header,
                       uncertainty=StdDevUncertainty(im.uncertainty.array.T),
                       mask=im.mask.T,unit=u.dimensionless_unscaled)

    def trim(self,im,trimimage=False) :
        """ Trim image by masking non-trimmed pixels
            May need to preserve image size to match reference/calibration frames, etc.
        """
        if type(im) is not list : ims=[im]
        else : ims = im

        outim = []
        for  im,trimbox,outbox in zip(ims,self.trimbox,self.outbox) :
            if self.namp == 1 : 
                boxes = [trimbox]
                outboxes = [outbox]
            else : 
                boxes = trimbox
                outboxes = outbox
            tmp = np.ones(im.mask.shape,dtype=bool)
            for box in boxes :
                box.setval(tmp,False)
            im.mask = np.logical_or(im.mask,tmp)
            if trimimage :
                xmax=0 
                ymax=0 
                for box,outbox in zip(boxes,outboxes) :
                    xmax = np.max([xmax,outbox.xmax])
                    ymax = np.max([ymax,outbox.ymax])
                z=np.zeros([ymax+1,xmax+1]) 
                out = CCDData(z,z,z,unit=u.dimensionless_unscaled,header=im.header)
                for box,outbox in zip(boxes,outboxes) :
                    out.data[outbox.ymin:outbox.ymax+1,outbox.xmin:outbox.xmax+1] =  \
                            im.data[box.ymin:box.ymax+1,box.xmin:box.xmax+1]
                    out.uncertainty.array[outbox.ymin:outbox.ymax+1,outbox.xmin:outbox.xmax+1] = \
                            im.uncertainty.array[box.ymin:box.ymax+1,box.xmin:box.xmax+1]
                    out.mask[outbox.ymin:outbox.ymax+1,outbox.xmin:outbox.xmax+1] = \
                            im.mask[box.ymin:box.ymax+1,box.xmin:box.xmax+1]
                outim.append(out)
        if trimimage: 
            if len(outim) == 1 : return outim[0]
            else : return outim
        else : return im
       
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
             out.append(ccdproc.subtract_bias(im,bias))
         if len(out) == 1 : return out[0]
         else : return out

    def dark(self,im,superdark=None) :
         """ Superdark subtraction
         """
         # only subtract if we are given a superdark!
         if superdark is None : return im

         # work with lists so that we can handle multi-channel instruments
         if type(im) is not list : ims=[im]
         else : ims = im
         if type(superdark) is not list : superdarks=[superdark]
         else : superdarks = superdark
         out=[]
         for im,dark in zip(ims,superdarks) :
             if self.verbose : print('  subtracting dark...')
             out.append(ccdproc.subtract_dark(im,dark,exposure_time='EXPTIME',exposure_unit=u.s))
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
             corr = ccdproc.flat_correct(im,flat)
             out.append(corr)
             if display is not None : 
                 display.tv(corr)
                 #plot central crossections
                 display.plotax1.cla()
                 dim=corr.data.shape
                 col = int(dim[1]/2)
                 row = corr.data[:,col]
                 display.plotax1.plot(row)
                 min,max=tv.minmax(row,low=5,high=5)
                 display.plotax1.set_ylim(min,max)
                 display.plotax1.set_xlabel('row')
                 display.plotax1.text(0.05,0.95,'Column {:d}'.format(col),
                     transform=display.plotax1.transAxes)
                 display.plotax2.cla()
                 row = int(dim[0]/2)
                 col = corr.data[row,:]
                 min,max=tv.minmax(col,low=10,high=10)
                 display.plotax2.plot(col)
                 display.plotax2.set_xlabel('col')
                 display.plotax2.text(0.05,0.95,'Row {:d}'.format(row),
                     transform=display.plotax2.transAxes)
                 display.plotax2.set_ylim(min,max)
                 input("  See flat-fielded image and original with - key. Hit any key to continue")
         if len(out) == 1 : return out[0]
         else : return out


    def scatter(self,im,scat=None,display=None,smooth=3,smooth2d=31) :
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

    def crrej(self,im,crbox=None,nsig=5,display=None) :
        """ CR rejection
        """
        if crbox is None: return im
        if type(im) is not list : ims=[im]
        else : ims = im
        out=[]
        for i,(im,gain,rn) in enumerate(zip(ims,self.gain,self.rn)) :
            if display is not None : 
                display.tv(im)
            if crbox == 'lacosmic':
                if self.verbose : print('  zapping CRs with ccdproc.cosmicray_lacosmic')
                im= ccdproc.cosmicray_lacosmic(im,gain_apply=False,
                       gain=gain*u.dimensionless_unscaled,readnoise=rn*u.dimensionless_unscaled)
            else :
                if self.verbose : print('  zapping CRs with filter [{:d},{:d}]...'.format(*crbox))
                image.zap(im,crbox,nsig=nsig)
            if display is not None : 
                display.tv(im)
                input("  See CR-zapped image and original with - key. Hit any key to continue")
            out.append(im)
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

    def platesolve(self,im,scale=0.46,seeing=2,display=None,flip=True) :
        """ try to get plate solution with imwcs
        """
        if self.verbose : print('  plate solving ....')

        # find stars
        mad=np.nanmedian(np.abs(im-np.nanmedian(im)))
        daofind=DAOStarFinder(fwhm=seeing/scale,threshold=10*mad)
        objs=daofind(im.data-np.nanmedian(im.data))
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

        # get WCS
        header=fits.open(os.path.basename(tmpfile[1])+'xy.wcs')[0].header
        w=WCS(header)
        im.wcs=w

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
        for f in glob.glob(os.path.basename(tmpfile[1])+'*') :
            os.remove(f)
        if display is not None :
            input("  See plate solve stars. Hit any key to continue")
            display.tvclear()
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

    def reduce(self,num,crbox=None,bias=None,dark=None,flat=None,
               scat=None,badpix=None,solve=False,return_list=False,display=None,trim=False,seeing=2) :
        """ Full reduction
        """
        im=self.rd(num)
        self.overscan(im,display=display)
        im=self.bias(im,superbias=bias)
        im=self.dark(im,superdark=dark)
        self.scatter(im,scat=scat,display=display)
        im=self.flat(im,superflat=flat,display=display)
        self.badpix_fix(im,val=badpix)
        if trim and display is not None: display.tvclear()
        if trim : im=self.trim(im,trimimage=trim)
        im=self.crrej(im,crbox=crbox,display=display)
        if solve : 
            im=self.platesolve(im,display=display,scale=self.scale,flip=self.flip,seeing=seeing)
        if return_list and type(im) is not list : im=[im]
        return im

    def write(self,im,name,overwrite=True,png=False) :
        """ write out image, deal with multiple channels 
        """

        if type(im) is not list : ims=[im]
        else : ims = im
        for i,frame in enumerate(ims) : 
            if self.nchip > 1 : outname = name.replace('.fits','')+'_'+self.channels[i]+'.fits'
            else : outname = name
            frame.write(outname,overwrite=overwrite)
            if png :
                backend=matplotlib.get_backend()
                matplotlib.use('Agg')
                fig=plt.figure(figsize=(12,9))
                vmin,vmax=tv.minmax(frame.data)
                plt.imshow(frame.data,vmin=vmin,vmax=vmax,
                           cmap='Greys_r',interpolation='nearest',origin='lower')
                plt.colorbar(shrink=0.8)
                plt.axis('off')
                fig.tight_layout()
                fig.savefig(name+'.png')
                plt.close()
                matplotlib.use(backend)
 

    def getcube(self,ims,**kwargs) :
        """ Read images into data cube
        """
        # create list of images, reading and overscan subtracting
        allcube = []
        for im in ims :
            if type(im) is not astropy.nddata.CCDData :
                data = self.reduce(im, **kwargs)
            allcube.append(data)

        # if just one frame, put in 2D list anyway so we can use same code, allcube[nframe][nchip]
        if self.nchip == 1 :
            allcube=[list(i) for i in zip(*[allcube])]

        return allcube

    def sum(self,ims, return_list=False, **kwargs) :
        """ Coadd input images
        """
        allcube = self.getcube(ims, **kwargs)
        nframe = len(allcube)
        
        out=[]
        for chip in range(self.nchip) :
            datacube = []
            varcube = []
            maskcube = []
            for im in range(nframe) :
                datacube.append(allcube[im][chip].data)
                varcube.append(allcube[im][chip].uncertainty.array**2)
                maskcube.append(allcube[im][chip].mask)
            sum = np.sum(np.array(datacube),axis=0)
            sig = np.sqrt(np.sum(np.array(varcube),axis=0))
            mask = np.any(maskcube,axis=0)
            out.append(CCDData(sum,header=allcube[0][chip].header,
                       uncertainty=StdDevUncertainty(sig),
                       mask=mask,unit=u.dimensionless_unscaled))
        
        # return the frame
        if len(out) == 1 : 
           if return_list : return [out[0]]
           else : return out[0]
        else : return out

    def combine(self,ims, normalize=False,display=None,div=True,return_list=False, type='median',sigreject=5,**kwargs) :
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
                maskcube.append(allcube[im][chip].mask)
            if self.verbose: print('  median combining data....')
            if type == 'median' :
                med = np.median(np.array(datacube),axis=0)
                sig = 1.253 * np.sqrt(np.mean(np.array(varcube),axis=0)/nframe)
            elif type == 'mean' :
                med = np.mean(np.array(datacube),axis=0)
                sig = np.sqrt(np.mean(np.array(varcube),axis=0)/nframe)
            elif type == 'reject' :
                med = np.median(np.array(datacube),axis=0)
                gd=np.where(np.array(datacube)<med+sigreject*np.sqrt(np.array(varcube)))
		med = np.mean(np.array(datacube[gd]),axis=0)
            if self.verbose: print('  calculating uncertainty....')
            mask = np.any(maskcube,axis=0)
            comb=CCDData(med,header=allcube[im][chip].header,uncertainty=StdDevUncertainty(sig),
                         mask=mask,unit=u.dimensionless_unscaled)
            if normalize: comb.meta['MEANNORM'] = np.array(allnorm).mean()
            out.append(comb)

            # display final combined frame and individual frames relative to combined
            if display :
                display.clear()
                display.tv(comb,sn=True)
                display.tv(comb)
                gd=np.where(comb.mask == False)
                min,max=tv.minmax(med[gd[0],gd[1]],low=10,high=10)
                display.plotax1.hist(med[gd[0],gd[1]],bins=np.linspace(min,max,100),histtype='step')
                display.fig.canvas.draw_idle()
                get = input("  See final image, use - key for S/N image. Hit any key to continue")
                if get == 'i' : code.interact(local=locals())
                elif get == 'q' : sys.exit()
                elif get == 'p' : pdb.set_trace()
                for i,im in enumerate(ims) :
                    min,max=tv.minmax(med[gd[0],gd[1]],low=5,high=5)
                    display.fig.canvas.draw_idle()
                    if div :
                        display.plotax2.hist((allcube[i][chip].data/med)[gd[0],gd[1]],bins=np.linspace(0.5,1.5,100),histtype='step')
                        display.tv(allcube[i][chip].data/med,min=0.5,max=1.5)
                        input("    see image: {} divided by master, hit any key to continue".format(im))
                    else :
                        delta=5*self.rn[chip]
                        display.plotax2.hist((allcube[i][chip].data-med)[gd[0],gd[1]],bins=np.linspace(-delta,delta,100),histtype='step')
                        display.tv(allcube[i][chip].data-med,min=-delta,max=delta)
                        input("    see image: {} minus master, hit any key to continue".format(im))

        # return the frame
        if len(out) == 1 :
           if return_list : return [out[0]]
           else : return out[0]
        else : return out

    def mkbias(self,ims,display=None,scat=None) :
        """ Driver for superbias combination (no superbias subtraction no normalization)
        """
        return self.combine(ims,display=display,div=False,scat=scat)

    def mkdark(self,ims,bias=None,display=None,scat=None) :
        """ Driver for superdark combination (no normalization)
        """
        return self.combine(ims,bias=bias,display=display,div=False,scat=scat)

    def mkflat(self,ims,bias=None,dark=None,scat=None,display=None) :
        """ Driver for superflat combination (with superbias if specified, normalize to normbox
        """
        return self.combine(ims,bias=bias,dark=dark,normalize=True,scat=scat,display=display)

    def mkspecflat(self,flats,wid=101) :
        """ Spectral flat takes out variation along wavelength direction
        """
        boxcar = Box1DKernel(wid)
        for iflat,flat in enumerate(flats) : 
            nrows=flats[iflat].data.shape[0]
            med = convolve(np.median(flat,axis=0),boxcar,boundary='extend')
            for row in range(flats[iflat].data.shape[0]) :
                flats[iflat].data[row,:] /= med
                flats[iflat].uncertainty.array[row,:] /= med

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

def getinput(str) :
    get = input(str)
    if get == 'i' : code.interact(local=globals())
    elif get == 'p' :
        pdb.set_trace()
    return get

class Data(object) :
    """ Experimental data class to cinclude wavelength array
    """
    def __init__(self,data,wave=None) :
        if type(data) is str :
            hdulist=fits.open(data)
            self.meta = hdulist[0].header
            self.attr_list = []
            for i in range(1,len(hdulist) ) :
                try : 
                    attr=hdulist[i].header['ATTRIBUT']
                except KeyError :
                    if i == 1 : attr='data'
                    elif i == 2 : attr='uncertainty'
                    elif i == 3 : attr='mask'
                    elif i == 4 : attr='wave'
                print('attr: {:s}'.format(attr))
                self.attr_list.append(attr)
                setattr(self,attr,hdulist[i].data) 
        elif type(data) is CCDData :
            self.unit = data.unit
            self.meta = data.meta
            self.data = data.data
            self.uncertainty = data.uncertainty
            self.mask = data.mask
            self.wave = wave
        else :
            print('Input must be a filename or CCDData object')

    def write(self,file,overwrite=True) :
        hdulist=fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=self.meta))
        for attr in self.attr_list :
            hdulist.append(fits.ImageHDU(getattr(self,attr)))
        hdulist.writeto(file,overwrite=overwrite)
