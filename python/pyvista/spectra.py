import matplotlib
from importlib_resources import files
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import pdb
import pickle
import copy
import scipy.signal
import scipy.interpolate
from scipy.optimize import curve_fit
import numpy as np
from astropy.modeling import models, fitting
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.io import ascii, fits
from astropy.convolution import convolve, Box1DKernel, Box2DKernel
from astropy.table import Table
import pyvista
import pyvista.data
from pyvista import image, tv
from tools import plots

class SpecData(CCDData) :
    """ Class to include a wavelength array on top of CCDData, with simple read/write/plot methods

    Parameters
    ----------

    Attributes
    ----------

    """
    def __init__(self,data,wave=None) :
        if type(data) is str :
            hdulist=fits.open(data)
            self.meta = hdulist[0].header
            self.unit = hdulist[0].header['BUNIT']
            self.data = hdulist[1].data
            self.uncertainty = StdDevUncertainty(hdulist[2].data)
            self.mask = hdulist[3].data
            self.wave = hdulist[4].data
        elif type(data) is CCDData :
            self.unit = data.unit
            self.meta = data.meta
            self.data = data.data
            self.uncertainty = data.uncertainty
            self.mask = data.mask
            self.wave = wave
        else :
            print('Input must be a filename or CCDData object')

    def write(self,file,overwrite=True,png=False) :
        """  Write SpecData to file
        """
        hdulist=fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=self.meta))
        hdulist.append(fits.ImageHDU(self.data))
        hdulist.append(fits.ImageHDU(self.uncertainty.array))
        hdulist.append(fits.ImageHDU(self.mask.astype(np.int16)))
        hdulist.append(fits.ImageHDU(self.wave))
        hdulist.writeto(file,overwrite=overwrite)
        if png :
            #backend=matplotlib.get_backend()
            #matplotlib.use('Agg')
            fig,ax=plots.multi(1,1,figsize=(18,6))
            self.plot(ax)
            fig.savefig(file+'.png')
            #matplotlib.use(backend)
            plt.close()

    def plot(self,ax,**kwargs) :
        if self.data.ndim == 1 :
            gd = np.where(self.mask == False)[0]
            plots.plotl(ax,self.wave[gd],self.data[gd],**kwargs)
        else :
            for row in range(self.wave.shape[0]) :
                gd = np.where(self.mask[row,:] == False)[0]
                plots.plotl(ax,self.wave[row,gd],self.data[row,gd],**kwargs)
        gd = np.where(self.mask == False)[0]
        med=np.nanmedian(self.data[gd])
        ax.set_ylim(0,2*med)
        
class WaveCal() :
    """ Class for wavelength solutions

    Parameters
    ----------
    file : str, optional
        filename for FITS file with WaveCal attributes
    type : str, optional, default='chebyshev'
        astropy model function
    degree : int, optional, default=2
        polynomial degree for wavelength direction
    ydegree : int, optional, default=2
        polynomial degree for spatial or cross-dispersed direction
    pix0 : int, optional, default=0
        reference pixel

    Attributes
    ----------
    type : str
        astropy model function for wavelength solution 
    degree : int
        polynomial degree for wavelength direction
    ydegree : int
        polynomial degree for spatial or cross-dispersed direction
    pix0 : int
        reference pixel
    orders : array_like
        spectral order for each row in spatially resolved or cross-dispersed
    pix : array_like
        pixel positions of identified lines in spectrum
    waves : array_like
        wavelength positions of identified lines in spectrum
    weights : array_like
        weights for fitting of identified lines in spectrum
    y : array_like
        spatial or cross-dispersed array position
    spectrum : array_like
        reference spectrum to be used to identify lines
    model : astropy model, or list of models
        Model(s) for the wavelength solution

    """

    def __init__ (self,file=None, type='chebyshev',degree=2,ydegree=2,
                  pix0=0) :
        if file is not None :
            if str(file)[0] == '.' or str(file)[0] == '/' :
                tab=Table.read(file)
            else :
                tab=Table.read(files(pyvista.data).joinpath(file))
            for tag in ['type','degree','ydegree','waves',
                        'waves_order','orders',
                        'pix0','pix','y','spectrum','weights'] :
                setattr(self,tag,tab[tag][0])
            self.orders=np.atleast_1d(self.orders)
            # make the initial models from the saved data
            self.model = None
            self.fit()
        else :
            self.type = type
            self.degree = degree
            self.ydegree = ydegree
            self.pix0 = pix0
            self.waves = None
            self.x = None
            self.y = None
            self.weights = None
            self.model = None
            self.ax = None
            self.spectrum = None
            self.orders = [1]

    def write(self,file,append=False) :
        """ Save object to file

            Parameters 
            ----------
            file : str, name of output file to write to, FITS format
            append : bool, append to existing file (untested)
        """
        tab=Table()
        for tag in ['type','degree','ydegree','waves','waves_order',
                    'orders','pix0','pix','y','weights','spectrum'] :
            tab[tag] = [getattr(self,tag)]
        if append :
            tab.write(file,append=True)
        else :
            tab.write(file,overwrite=True)
        return tab

    def wave(self,pixels=None,image=None) :
        """ Wavelength from pixel using wavelength solution model

        With pixels=[pixels,rows] keyword, return wavelengths for input set of pixels/rows
        With image=(nrow,ncol), returns wavelengths in an image
        

        Parameters
        ----------
        pix : array_like, optional
           input pixel positions [x] or [y,x]
        image : tuple, optional
           for input image size [nrows,ncols], return wavelengths at all pixels

        Returns 
        -------
        wavelength

        """
        if pixels is not None :
            out=np.zeros(len(pixels[0]))
            for i,pixel in enumerate(pixels[0]) :
                order=self.orders[pixels[1][i]]
                if self.type.find('2D') > 0 :
                    out[i]=self.model(pixel-self.pix0,pixels[1][i])/order
                else :
                    out[i]=self.model(pixel-self.pix0)/order
            return out
        else :
            out=np.zeros(image)
            cols=np.arange(out.shape[-1])
            if out.ndim == 2 :
                for row in range(out.shape[0]) : 
                    rows=np.zeros(len(cols))+row
                    order = self.orders[row]
                    if '2D' in self.type :
                        out[row,:] = self.model(cols-self.pix0,rows)/order
                    else :
                        out[row,:] = self.model(cols-self.pix0)/order
            else :
                out= self.model(cols-self.pix0)/self.orders[0]
            return out

    def getmod(self) :
        """ Return model for current attributes
        """

        if self.type == 'Polynomial1D' :
            mod=models.Polynomial1D(degree=self.degree)
        elif self.type == 'chebyshev' :
            mod=models.Chebyshev1D(degree=self.degree)
        elif self.type == 'chebyshev2D' :
            sz=self.spectrum.shape
            mod=models.Chebyshev2D(x_degree=self.degree,y_degree=self.ydegree,
                                   x_domain=[0,sz[1]],y_domain=[0,sz[0]])
        else :
            raise ValueError('unknown fitting type: '+self.type)
            return
        return mod

    def fit(self,degree=None,reject=3) :
        """ do a wavelength fit 

            If a figure has been set in identify, will show fit graphically and
            allow for manual removal of lines in 1D case.  In 2D case, outliers
            are detected and removed
 
            Parameters
            ----------
            degree : int, optional
              degree of polynomial in wavelength, else as previously set in object
        """
        print("doing wavelength fit")
        # set up fitter and model
        twod='2D' in self.type
        fitter=fitting.LinearLSQFitter()
        if degree is not None : self.degree=degree
        if self.model is None : self.model = self.getmod()
        mod = self.model

        if not hasattr(self,'ax') : self.ax = None
        if twod :
            # for 2D fit, we just use automatic line rejections
            nold=-1
            nbd=0
            while nbd != nold :
                # iterate as long as new points have been removbed
                nold=nbd
                self.model=fitter(mod,self.pix-self.pix0,self.y,
                           self.waves*self.waves_order,weights=self.weights)
                diff=self.waves-self.wave(pixels=[self.pix,self.y])
                gd = np.where(self.weights > 0)[0]
                print('  rms: {:8.3f}'.format(diff[gd].std()))
                bd = np.where(abs(diff) > reject*diff.std())[0]
                nbd = len(bd)
                print('rejecting {:d} points from {:d} total: '.format(
                      nbd,len(self.waves)))
                self.weights[bd] = 0.

            # plot the results
            if self.ax is not None : 
                self.ax[1].cla()
                scat=self.ax[1].scatter(self.waves,diff,marker='o',c=self.y,s=5)
                plots.plotp(self.ax[1],self.waves[bd],diff[bd],
                            marker='o',color='r',size=5)

                xlim=self.ax[1].get_xlim()
                self.ax[1].set_ylim(diff.min()-0.5,diff.max()+0.5)
                self.ax[1].plot(xlim,[0,0],linestyle=':')
                self.ax[1].text(0.1,0.9,'rms: {:8.3f}'.format(
                                diff[gd].std()),transform=self.ax[1].transAxes)
                cb_ax = self.fig.add_axes([0.94,0.05,0.02,0.4])
                cb = self.fig.colorbar(scat,cax=cb_ax)
                cb.ax.set_ylabel('Row')
                plt.draw()
                try: self.fig.canvas.draw_idle()
                except: pass
                print('  See 2D wavecal fit. Enter space in plot window to continue')
                get=plots.mark(self.fig)

        else :
          # 1D fit, loop over all rows in which lines have been identified
          nmod = len(set(self.y))
          models = []
          for row in set(self.y) :
            irow = np.where(self.y == row)[0]
            self.model=fitter(mod,self.pix[irow]-self.pix0,
                             self.waves[irow]*self.waves_order[irow],
                             weights=self.weights[irow])
            diff=self.waves-self.wave(pixels=[self.pix,self.y])
            print('  rms: {:8.3f} Angstroms'.format(diff[irow].std()))
            if self.ax is not None :
                # iterate allowing for interactive removal of points
                done = False
                ymax = self.ax[0].get_ylim()[1]
                print('  Input in plot window: ')
                print('       l : to remove all lines to left of cursor')
                print('       r : to remove all lines to right of cursor')
                print('       n : to remove line nearest cursor x position')
                print('       anything else : finish and return')
                while not done :
                    # do fit
                    gd=np.where(self.weights[irow]>0.)[0]
                    gd=irow[gd]
                    bd=np.where(self.weights[irow]<=0.)[0]
                    bd=irow[bd]
                    self.model=fitter(mod,self.pix[gd]-self.pix0,
                                      self.waves[gd]*self.waves_order[gd],
                                      weights=self.weights[gd])
                    diff=self.waves-self.wave(pixels=[self.pix,self.y])
                    print('  rms: {:8.3f} Anstroms'.format(diff[gd].std()))

                    # replot spectrum with new fit wavelength scale
                    self.ax[0].cla()
                    self.ax[0].plot(self.wave(image=self.spectrum.shape[1]),
                                    self.spectrum[0,:])
                    # plot residuals
                    self.ax[1].cla()
                    self.ax[1].plot(self.waves[gd],diff[gd],'go')
                    self.ax[1].text(0.1,0.9,'rms: {:8.3f} Angstroms'.format(
                               diff[gd].std()),transform=self.ax[1].transAxes)
                    self.ax[1].set_xlabel('Wavelength')
                    self.ax[1].set_ylabel('obs wave - fit wave')
                    if len(bd) > 0 : 
                        self.ax[1].plot(self.waves[bd],diff[bd],'ro')
                    self.ax[1].set_ylim(diff[gd].min()-0.5,diff[gd].max()+0.5)
                    plots._data_x = self.waves[irow][np.isfinite(self.waves[irow])]
                    plots._data_y = diff[irow][np.isfinite(diff[irow])]
                    for i in range(len(self.pix[irow])) :
                        j = irow[i]
                        self.ax[1].text(self.waves[j],diff[j],'{:2d}'.format(
                                        i),va='top',ha='center')
                        if self.weights[i] > 0 :
                            self.ax[0].plot([self.waves[j],self.waves[j]],
                                            [0,ymax],'g')
                        else :
                            self.ax[0].plot([self.waves[j],self.waves[j]],
                                            [0,ymax],'r')
                    plt.draw()

                    # get input from user on lines to remove
                    i = getinput('  input from plot window...', 
                                 self.fig,index=True)
                    if i == '' :
                        done = True
                    elif i[2] == 'l' :
                        bd=np.where(self.waves[irow]<i[0])[0]
                        self.weights[irow[bd]] = 0.
                    elif i[2] == 'r' :
                        bd=np.where(self.waves[irow]>i[0])[0]
                        self.weights[irow[bd]] = 0.
                    elif i[2] == 'n' :
                        #bd=np.argmin(np.abs(self.waves[irow]-i[0]))
                        bd=i[3]
                        self.weights[irow[bd]] = 0.
                    elif i == 'O' :
                        print('  current degree of fit: {:d}'.format(
                              self.degree))
                        self.degree = int(getinput(
                              '  enter new degree of fit: ',self.fig))
                        mod = self.getmod()
                    else : done = True
            models.append(self.model)
          if nmod == 1 : self.model = models[0]
          else : self.model = models       
        if self.ax is not None :
          plt.close(self.fig)
          self.fig=None
          self.ax=None

    def set_spectrum(self,spectrum) :
        """ Set spectrum used to derive fit
        """
        self.spectrum = np.atleast_2d(spectrum)

    def get_spectrum(self) :
        """ Set spectrum used to derive fit
        """
        return self.spectrum 

    def identify(self,spectrum,file=None,wav=None,wref=None,inter=False,
                 orders=None,verbose=False,rad=5,thresh=100, fit=True,
                 disp=None,display=None,plot=None,pixplot=False,
                 xmin=None,xmax=None,lags=range(-300,300), nskip=1) :
        """ Given some estimate of wavelength solution and file with lines,
            identify peaks and centroid, via methods:

            1. if input wav array/image is specified, use this to identify lines
            2. if WaveCal object as associated spectrum, use cross correlation
               to identify shift of input spectrum, then use previous solution
               to create a wavelength array. Cross correlation lags to try
               are specified by lags=range(dx1,dx2), default range(-300,300)
            3. if inter==True, prompt user to identify 2 lines
            4. use header cards DISPDW and DISPWC for dispersion and wave center
               or as specified by input disp=[dispersion] and wref=[lambda,pix]

            Given wavelength guess array, identify lines from input file, or,
               if no file given, lines saved in the WaveCal structure

            Lines are identified by looking for peaks within rad pixels of
               initial guess

            After line identification, fit() is called, unless fit=False

            With plot=True, plot of spectrum is shown, with initial wavelength
               guess. With pixplot=True, plot is shown as function of pixel
        """

        sz=spectrum.data.shape
        if len(sz) == 1 : 
            spectrum.data = np.atleast_2d(spectrum.data)
            spectrum.uncertainty.array = np.atleast_2d(spectrum.uncertainty.array)
            sz=spectrum.data.shape
        if xmin is None : xmin=0
        if xmax is None : xmax=sz[-1]
        nrow=sz[0]
        if orders is not None : self.orders = orders

        # get initial reference wavelengths if not given
        if wav is None :
            pix=np.arange(sz[-1])
            if self.spectrum is not None :
                # cross correlate with reference image to get pixel shift
                print('  cross correlating with reference spectrum using lags: ', lags)
                fitpeak,shift = image.xcorr(self.spectrum,spectrum.data,lags)
                if shift.ndim == 1 :
                    pixshift=(fitpeak+lags[0])[0]
                    print('  Derived pixel shift from input wcal: ',fitpeak+lags[0])
                    if display is not None :
                        display.plotax1.cla()
                        display.plotax1.text(0.05,0.95,'spectrum and reference',transform=display.plotax1.transAxes)
                        for row in range(spectrum.data.shape[0]) :
                            display.plotax1.plot(spectrum.data[row,:],color='m')
                            display.plotax1.plot(self.spectrum[row,:],color='g')
                        display.plotax1.set_xlabel('Pixel')
                        display.histclick=False
                        display.plotax2.cla()
                        display.plotax2.text(0.05,0.95,
                               'cross correlation: {:8.3f}'.format(pixshift),
                               transform=display.plotax2.transAxes)
                        display.plotax2.plot(lags,shift)
                        display.plotax2.set_xlabel('Lag')
                        plt.draw()
                        print("  See spectrum and template spectrum (top), cross correlation(bottom)",display.fig)
                    # single shift for all pixels
                    self.pix0 = self.pix0+fitpeak+lags[0]
                    wav=np.atleast_2d(self.wave(image=np.array(sz)))
                    #wav=np.atleast_2d(self.wave(image=sz))
                else :
                    # different shift for each row
                    wav=np.zeros(sz)
                    cols = np.arange(sz[-1])
                    orders=[]
                    for row in range(wav.shape[0]) : 
                        print('  Derived pixel shift from input wcal for row: {:d} {:d}'.format
                               (row,shift[row,:].argmax()+lags[0]),end='\r')
                        rows=np.zeros(len(cols))+row
                        try : order = self.orders[row]
                        except : order=self.orders[0]
                        orders.append(order)
                        pix0 = self.pix0+fitpeak[row]+lags[0]
                        wav[row,:] = self.model(cols-pix0)/order
                    # ensure we have 2D fit
                    self.type = 'chebyshev2D'
                    self.model = None
                    self.orders = orders
                    print("")
            elif inter :
                f,a=plots.multi(1,1)
                a.plot(spectrum.data[0,:])
                for i in range(2) :
                    print('mark location of known line with m key')
                    ret=plots.mark(f)
                    w=input('wavelength of line: ')
                    if i==0 :
                        w0=float(w)
                        pix0=ret[0]
                    else :
                        disp = (float(w)-w0)/(ret[0]-pix0)
                print(w0,pix0,disp)
                wav=np.atleast_2d(w0+(pix-pix0)*disp)
            else :
                # get dispersion guess from header cards if not given in disp
                if disp is None: disp=spectrum.header['DISPDW']
                if wref is not None :
                    w0=wref[0]
                    pix0=wref[1]
                else:
                    w0=spectrum.header['DISPWC']
                    pix0=sz[1]/2 
                wav=np.atleast_2d(w0+(pix-pix0)*disp)

        # open file with wavelengths and read
        if file is not None :
            if file.find('/') < 0 :
                f=open(files(pyvista.data).joinpath('lamps/'+file),'r')
            else :
                f=open(file,'r')
            lines=[]
            for line in f :
                if line[0] != '#' :
                    w=float(line.split()[0])
                    # if we have microns, convert to Angstroms
                    if w<10 : w*=10000
                    if w > np.nanmin(wav) and w < np.nanmax(wav) : 
                        lines.append(w)
            lines=np.array(lines)
            f.close()
        else :
            lines = self.waves
            weights = self.weights
            gd = np.where(weights >0)[0]
            lines = set(lines[gd])

        # get centroid around expected lines
        x=[]
        y=[]
        fwhm=[]
        waves=[]
        waves_order=[]
        weight=[]
        diff=[]
        if display is not None and  isinstance(display,pyvista.tv.TV) :
            display.ax.cla()
            display.ax.axis('off')
            display.tv(spectrum.data)
        if plot is None :
            self.ax = None
        else :
            if type(plot) is matplotlib.figure.Figure :
                plot.clf()
                plt.draw()
                ax1=plot.add_subplot(2,1,1) 
                ax2=plot.add_subplot(2,1,2,sharex=ax1) 
                plot.subplots_adjust(left=0.05,right=0.92, hspace=1.05)
                ax=[ax1,ax2]
                self.fig = plot
                self.ax = ax
            else :
                fig,ax = plt.subplots(2,1,sharex=True,figsize=(10,5))
                fig.subplots_adjust(hspace=1.05)
                self.fig = fig
                self.ax = ax

        if plot is not None : ax[0].cla()
        for row in range(0,nrow,nskip) :
            print('  identifying lines in row: ', row,end='\r')
            if plot is not None :
                # next line for pixel plot
                if pixplot : ax[0].plot(spectrum.data[row,:])
                else : ax[0].plot(wav[row,:],spectrum.data[row,:])
                #ax[0].set_yscale('log')
                ax[0].set_ylim(1.,ax[0].get_ylim()[1])
                ax[0].text(0.1,0.9,'row: {:d}'.format(row),transform=ax[0].transAxes)
                ax[0].set_xlabel('Rough wavelength')
                ax[0].set_ylabel('Intensity')
            for line in lines :
                # initial guess from input wavelengths
                peak=np.nanargmin(abs(line-wav[row,:]))
                if ( (peak > xmin+rad) and (peak < xmax-rad)) :
                  # set peak to highest nearby pixel
                  peak=(spectrum.data[row,peak-rad:peak+rad+1]).argmax()+peak-rad
                  if ( (peak < xmin+rad) or (peak > xmax-rad)) : continue
                  if isinstance(display,pyvista.tv.TV) :
                      display.ax.scatter(peak,row,marker='o',color='r',s=2)
                  # S/N threshold
                  if (spectrum.data[row,peak-rad:peak+rad+1]/
                      spectrum.uncertainty.array[row,peak-rad:peak+rad+1]).max() > thresh:

                    oldpeak = 0
                    niter=0
                    xx = np.arange(peak-rad,peak+rad+1)
                    yy = spectrum.data[row,peak-rad:peak+rad+1]
                    try :  
                      while peak != oldpeak and  niter<10:
                        p0 = [spectrum.data[row,peak],peak,rad,0.]
                        coeff, var_matrix = curve_fit(gauss, xx, yy, p0=p0)
                        cent = coeff[1]
                        oldpeak = peak
                        peak = int(cent)
                        niter+=1
                      if niter == 10 : continue
                    except : continue
                    if verbose : print(line,peak,*coeff)
                    if display is not None and  isinstance(display,pyvista.tv.TV) :
                        display.ax.scatter(cent,row,marker='o',color='g',s=2)
                    if plot is not None :
                        if pixplot :ax[0].plot([cent,cent],ax[0].get_ylim(),color='r')
                        else : ax[0].text(line,1.,'{:7.1f}'.format(line),rotation='vertical',va='top',ha='center')
                    x.append(cent)
                    y.append(row)
                    fwhm.append(np.abs(coeff[2]*2.354))
                    # we will fit for wavelength*order
                    waves.append(line)
                    try: order = self.orders[row]
                    except: order=self.orders[0]
                    waves_order.append(order)
                    weight.append(1.)
        if plot is not None : 
            self.fig.tight_layout()
            print('  See identified lines.')
        self.pix=np.array(x)
        self.y=np.array(y)
        self.fwhm=np.array(fwhm)
        self.waves=np.array(waves)
        self.waves_order=np.array(waves_order)
        self.weights=np.array(weight)
        self.spectrum = spectrum.data

        if fit: self.fit()
        print('')

    def scomb(self,hd,wav,average=True,usemask=True) :
        """ Resample onto input wavelength grid

        Uses current wavelength solution, linearly interpolates to specified
          wavelengths, on a row-by-row basis. Allows for order overlap.
       

        Parameters
        ----------
        hd : array or CCDData
          input image to resample
        wav : array_like
          new wavelengths to interpolate to
        average : bool, optional, default=True
          if overlapping orders, average if True, otherwise sum
        usemask : bool, optional, default=True
          if True, skip input masked pixels for interpolation

        """
        #output grid
        out=np.zeros(len(wav))
        sig=np.zeros(len(wav))
        mask=np.zeros(len(wav),dtype=bool)
        # raw wavelengths
        w=self.wave(image=np.array(np.atleast_2d(hd.data).shape))
        for i in range(np.atleast_2d(hd).shape[0]) :
            sort=np.argsort(w[i,:])
            if usemask : 
                gd = np.where(~hd.mask[i,sort])
                sort= sort[gd]
            if len(gd[0]) == 0 : continue
            wmin=w[i,sort].min()
            wmax=w[i,sort].max()
            w2=np.abs(wav-wmin).argmin()
            w1=np.abs(wav-wmax).argmin()
            if average :
                out[w2:w1] += ( np.interp(wav[w2:w1],w[i,sort],
                                np.atleast_2d(hd.data)[i,sort]) /
                                np.interp(wav[w2:w1],w[i,sort],
                                np.atleast_2d(hd.uncertainty.array)[i,sort])**2 )
                sig[w2:w1] += 1./np.interp(wav[w2:w1],w[i,sort],
                                np.atleast_2d(hd.uncertainty.array)[i,sort])**2 
            else :
                out[w2:w1] += np.interp(wav[w2:w1],w[i,sort],
                                np.atleast_2d(hd.data)[i,sort])
                sig[w2:w1] += np.interp(wav[w2:w1],w[i,sort],
                                np.atleast_2d(hd.uncertainty.array**2)[i,sort])
        if average :
            out = out / sig
            sig = np.sqrt(1./sig)
        else :
            sig = np.sqrt(sig)
        return CCDData(out,uncertainty=StdDevUncertainty(sig),
                       mask=mask,header=hd.header,unit='adu')


    def correct(self,hd,wav) :
        """ Resample input image to desired wavelength scale

        Uses current wavelength solution, linearly interpolates to specified
          wavelengths, on a row-by-row basis.

        Parameters
        ----------
        hd : CCDData, input image to resample
        wav : array_like, new wavelengths to interpolate to

        """

        out=np.zeros([hd.data.shape[0],len(wav)])
        sig=np.zeros_like(out)
        w=self.wave(image=hd.data.shape)
        for i in range(len(out)) :
            sort=np.argsort(w[i,:])
            wmin=w[i,sort].min()
            wmax=w[i,sort].max()
            w2=np.abs(wav-wmin).argmin()
            w1=np.abs(wav-wmax).argmin()
            out[i,w2:w1] += np.interp(wav[w2:w1],w[i,sort],hd.data[i,sort])
            sig[i,w2:w1] += np.sqrt(
                            np.interp(wav[w2:w1],w[i,sort],hd.uncertainty.array[i,sort]**2))

        return CCDData(out,StdDevUncertainty(sig),unit='adu'))

class Trace() :
    """ Class for spectral traces

    Attributes 
    ----------
    type : str
        type of astropy model to use
    degree : int
        polynomial degree to use
    sc0 : int
        starting column for trace, will work in both directions from here
    pix0 : int
        derived shift of current image relative to reference image
    spectrum : array_like
        reference spatial slice at sc0, used to determine object location
    rad : int
        radius in pixels to use for calculating centroid
    lags : array_like
        range of lags to use to try to find object locations

    Parameters
    ----------
    file : str, optional
        filename for FITS file with WaveCal attributes

    """

    def __init__ (self,file=None,inst=None, type='Polynomial1D',degree=2,
                  pix0=0,rad=5, spectrum=None,model=None,sc0=None,rows=None,
                  transpose=False,lags=None,channel=None) :

        if file is not None :
            """ Initialize object from FITS file
            """
            try:
                if str(file)[0] == '.' or str(file)[0] == '/' :
                    tab=Table.read(file)
                else :
                    tab=Table.read(files(pyvista.data).joinpath(file))
            except FileNotFoundError :
                raise ValueError("can't find file {:s}",file)

            for tag in ['type','degree','sc0','pix0',
                        'spectrum','rad','lags','transpose'] :
                setattr(self,tag, tab[tag][0])
            for tag in ['rows','index'] :
                try : setattr(self,tag,tab[tag][0])
                except KeyError : 
                    print('no attribute: ', tag)
                    setattr(self,tag,None)
            coeffs = tab['coeffs'][0]
            # use saved coefficients to instantiate model
            self.model = []
            for row in coeffs :
                if self.type == 'Polynomial1D' :
                    kwargs={}
                    for i,c in enumerate(row) :
                        name='c{:d}'.format(i)
                        kwargs[name] = c
                    self.model.append(
                        models.Polynomial1D(degree=self.degree,**kwargs))
                else :
                    raise ValueError('Only Polynomial1D currently implemented')
            return

        self.type = type
        self.degree = degree
        self.pix0 = pix0
        self.spectrum = spectrum
        self.rad = rad
        self.transpose = transpose
        if inst == 'TSPEC' :
            self.degree = 3
            self.rows = [[135,235],[295,395],[435,535],[560,660],[735,830]]
            self.lags = range(-75,75) 
        elif inst == 'DIS' :
            if channel == 0 : self.rows=[215,915]
            elif channel == 1 : self.rows=[100,800]
            else : raise ValueError('need to specify channel')
            self.lags = range(-300,300) 
        elif inst == 'KOSMOS' :
            self.rows=[550,1450]
            self.lags = range(-300,300) 
            self.transpose = True
        elif inst == 'ARCES' :
            self.lags = range(-10,10) 
        if rows is not None : self.rows=rows
        if lags is not None : self.lags=lags
        if model is not None : self.model=model
        if sc0 is not None : self.sc0=sc0

    def write(self,file,append=False) :
        """ Write trace information to FITS file
        """
        tab=Table()
        for tag in ['type','degree','sc0','pix0',
                    'spectrum','rad','lags','transpose'] :
            tab[tag] = [getattr(self,tag)]
        for tag in ['rows','index'] :
            try : 
                if getattr(self,tag) is not None : tab[tag] = [np.array(getattr(self,tag))]
            except AttributeError : print('no attribute: ', tag)
        coeffs = []
        if self.type == 'Polynomial1D' :
            # load model coefficients
            for m in self.model :
                row=[]
                for i in range(self.degree+1) :
                    name='c{:d}'.format(i)
                    row.append(getattr(m,name).value)
                coeffs.append(row)
        tab['coeffs'] = [np.array(coeffs)]
        if append :
            tab.write(file,append=True)
        else :
            tab.write(file,overwrite=True)
        return tab


    def trace(self,im,srows,sc0=None,plot=None,display=None,
              thresh=20,index=None,skip=1) :
        """ Trace a spectrum from starting position
        """

        if plot == None and display != None : plot = display

        fitter=fitting.LinearLSQFitter()
        if self.type == 'Polynomial1D' :
            mod=models.Polynomial1D(degree=self.degree)
        else :
            raise ValueError('unknown fitting type: '+self.type)
            return

        if index is not None and len(index) != len(srows) :
            raise ValueError('length of index and srows must be the same')

        if self.transpose :
            hd = image.transpose(im)
        else :
            hd = im

        nrow = hd.data.shape[0]
        ncol = hd.data.shape[1]
        if sc0 is None : self.sc0 = int(ncol/2)
        else : self.sc0 = sc0
        self.spectrum = hd.data[:,self.sc0]
        self.spectrum[self.spectrum<0] = 0.
        rows = np.arange(nrow)
        ypos = np.zeros(ncol)
        ysum = np.zeros(ncol)
        yvar = np.zeros(ncol)
        ymask = np.ones(ncol,dtype=bool)

        # we want to handle multiple traces, so make sure srows is iterable
        if type(srows ) is int or type(srows) is float : srows=[srows]
        self.model=[]
        if plot : 
            plot.clear()
            plot.tv(hd)

        rad = self.rad-1
        for irow,srow in enumerate(srows) :
            try:
                print('  Tracing row: {:d}'.format(int(srow)),end='\r')
            except: 
                pdb.set_trace()
                continue
            sr=copy.copy(srow)
            sr=int(round(sr))
            sr=hd.data[sr-rad:sr+rad+1,self.sc0].argmax()+sr-rad
            # march left from center
            for col in range(self.sc0,0,-skip) :
                # centroid
                cr=sr-rad+hd.data[sr-rad:sr+rad+1,col].argmax()
                ysum[col] = np.sum(hd.data[cr-rad:cr+rad+1,col]) 
                ypos[col] = np.sum(rows[cr-rad:cr+rad+1]*hd.data[cr-rad:cr+rad+1,col]) / ysum[col]
                yvar[col] = np.sum(hd.uncertainty.array[cr-rad:cr+rad+1,col]**2) 
                ymask[col] = np.any(hd.mask[cr-rad:cr+rad+1,col]) 
                # if centroid is too far from starting guess, mask as bad
                if np.abs(ypos[col]-sr) > np.max([rad/2.,0.75]) : ymask[col] = True
                # use this position as starting center for next if above threshold S/N
                if (not ymask[col]) & np.isfinite(ysum[col]) & (ysum[col]/np.sqrt(yvar[col]) > thresh)  : sr=int(round(ypos[col]))
            sr=copy.copy(srow)
            sr=int(round(sr))
            sr=hd.data[sr-rad:sr+rad+1,self.sc0].argmax()+sr-rad
            # march right from center
            for col in range(self.sc0+1,ncol,skip) :
                # centroid
                cr=sr-rad+hd.data[sr-rad:sr+rad+1,col].argmax()
                ysum[col] = np.sum(hd.data[cr-rad:cr+rad+1,col]) 
                ypos[col] = np.sum(rows[cr-rad:cr+rad+1]*hd.data[cr-rad:cr+rad+1,col]) / ysum[col]
                yvar[col] = np.sum(hd.uncertainty.array[cr-rad:cr+rad+1,col]**2) 
                ymask[col] = np.any(hd.mask[cr-rad:cr+rad+1,col]) 
                # use this position as starting center for next if above threshold S/N
                if np.abs(ypos[col]-sr) > np.max([rad/2.,0.75]) : ymask[col] = True
                if (not ymask[col]) & np.isfinite(ysum[col]) & (ysum[col]/np.sqrt(yvar[col]) > thresh)  : sr=int(round(ypos[col]))

            cols=np.arange(ncol)
            gd = np.where((~ymask) & (ysum/np.sqrt(yvar)>thresh) )[0]
            model=(fitter(mod,cols[gd],ypos[gd]))

            # reject outlier points (>1 pixel) and refit
            res = model(cols)-ypos
            gd = np.where((~ymask) & (ysum/np.sqrt(yvar)>thresh) & (np.abs(res)<1))[0]
            model=(fitter(mod,cols[gd],ypos[gd]))
            if len(gd) < 10 : 
                print('  failed trace for row: {:d}'.format(irow))
                #model=copy.copy(oldmodel[irow])
            self.model.append(model)

            if plot : 
                plot.ax.scatter(cols,ypos,marker='o',color='r',s=4) 
                plot.ax.scatter(cols[gd],ypos[gd],marker='o',color='g',s=10) 
                plot.ax.plot(cols,model(cols),color='m')
                plot.plotax2.cla()
                plot.plotax2.plot(cols,model(cols),color='m')
                plot.plotax2.text(0.05,0.95,'Derived trace',
                       transform=plot.plotax2.transAxes)
                #plt.pause(0.05)

        self.pix0=0
        if index is not None : self.index = index
        else : self.index=np.arange(len(self.model))
        print("")
        if plot : 
            while getinput('  See trace. Hit space bar to continue....',plot.fig)[2] != ' ' :
                pass

    def retrace(self,hd,plot=None,display=None,thresh=20) :
        """ Retrace starting with existing model
        """
        if plot == None and display != None : plot = display
        self.find(hd)
        srows = []
        for row in range(len(self.model)) :
            print("Using shift: ",self.pix0)
            srows.append(self.model[row](self.sc0)+self.pix0)
        self.trace(hd,srows,plot=plot,thresh=thresh)
    
    def findpeak(self,hd,width=100,thresh=5,plot=False) :
        """ Find peaks in spatial profile for subsequent tracing

            Parameters
            ----------
            hd : CCDData object
                 Input image
            width : int, default=100
                 width of window around central wavelength to median to give spatial profile
            thresh : float, default = 5
                 threshold for finding objects, as a factor to be multiplied by the median uncertainty

            Returns
            -------
            list of peak locations

        """
        if self.transpose :
            im = image.transpose(hd)
        else :
            im = copy.deepcopy(hd)

        print('looking for peaks using {:d} pixels around {:d}, threshhold of {:f}'.
              format(2*width,self.sc0,thresh))

        back =np.median(im.data[self.rows[0]:self.rows[1],
                                self.sc0-width:self.sc0+width])
        sig =np.median(im.uncertainty.array[self.rows[0]:self.rows[1],
                                self.sc0-width:self.sc0+width])

        if plot :
            plt.figure()
            plt.plot(self.rows[0]:self.rows[1],np.median(im.data[self.rows[0]:self.rows[1],
                                       self.sc0-width:self.sc0+width],axis=1)-back)
            plt.xlabel('Spatial pixel')
            plt.ylabel('Median flux')
        peaks,fiber = findpeak(np.median(im.data[self.rows[0]:self.rows[1],
                                         self.sc0-width:self.sc0+width],axis=1)-back,
                         thresh=thresh*sig)
        return peaks+self.rows[0], fiber

 
    def find(self,hd,lags=None,plot=None,display=None) :
        """ Determine shift from existing trace to input frame
        """
        if lags is None : lags = self.lags
        if plot == None and display != None : plot = display

        if self.transpose :
            im = image.transpose(hd)
        else :
            im = copy.deepcopy(hd)
      
        # if we have a window, zero array outside of window
        spec=im.data[:,self.sc0-50:self.sc0+50].sum(axis=1)
        try:
            spec[:self.rows[0]] = 0.  
            spec[self.rows[1]:] = 0.  
        except: pass
        fitpeak,shift = image.xcorr(self.spectrum,spec,lags)
        pixshift=(fitpeak+lags[0])[0]
        if plot is not None :
            plot.clear()
            plot.tv(im)
            plot.plotax1.cla()
            plot.plotax1.text(0.05,0.95,'obj and ref cross-section',transform=plot.plotax1.transAxes)
            plot.plotax1.plot(self.spectrum/self.spectrum.max())
            plot.plotax1.plot(im.data[:,self.sc0]/im.data[:,self.sc0].max())
            plot.plotax1.set_xlabel('row')
            plot.histclick=False
            plot.plotax2.cla()
            plot.plotax2.text(0.05,0.95,'cross correlation {:8.3f}'.format(pixshift),
                              transform=plot.plotax2.transAxes)
            plot.plotax2.plot(lags,shift)
            plot.plotax2.set_xlabel('lag')
            plt.draw()
            getinput('  See spectra and cross-correlation. Hit any key in display window to continue....',plot.fig)
        self.pix0=fitpeak+lags[0]
        self.pix0=pixshift
        return fitpeak+lags[0]
 
    def extract(self,im,rad=None,back=[],scat=False,
                display=None,plot=None,medfilt=None,nout=None,threads=0) :
        """ Extract spectrum given trace(s)
        """
        if plot == None and display != None : plot = display
        if self.transpose :
            hd = image.transpose(im)
        else :
            hd = im

        if rad is None : rad=self.rad
        if len(back) > 0 :
            for bk in back:
                try :
                    if len(bk) != 2 or not isinstance(bk[0],int) or not isinstance(bk[1],int) :
                        raise ValueError('back must be list of [backlo,backhi] integer pairs')
                except :
                    raise ValueError('back must be list of [backlo,backhi] integer pairs')
        nrows=hd.data.shape[0]
        ncols=hd.data.shape[-1]
        if nout is not None :
            spec = np.zeros([nout,hd.data.shape[1]])
            sig = np.zeros([nout,hd.data.shape[1]])
            mask = np.zeros([nout,hd.data.shape[1]],dtype=bool)
        else :
            spec = np.zeros([len(self.model),hd.data.shape[1]])
            sig = np.zeros([len(self.model),hd.data.shape[1]])
            mask = np.zeros([len(self.model),hd.data.shape[1]],dtype=bool)

        pars=[]
        if threads == 0 : 
            skip=1
            npars=ncols
        else : 
            skip=ncols//threads
            npars=threads
        for col in range(npars) :
            if col == threads-1 : ec=ncols
            else : ec=col*skip+skip
            pars.append((hd.data[:,col*skip:ec],
                         hd.uncertainty.array[:,col*skip:ec],
                         hd.mask[:,col*skip:ec],
                         np.arange(col*skip,ec),self.model,rad,self.pix0,back))
        if threads > 0 :
            pool = mp.Pool(threads)
            output = pool.map_async(extract_col, pars).get()
            pool.close()
            pool.join()
            col=0
            for out in output :
                nc=out[0].shape[1]
                spec[self.index,col:col+nc] = out[0]
                sig[self.index,col:col+nc] = out[1]
                mask[self.index,col:col+nc] = out[2]
                col+=skip
        else :
            col=0
            for par in pars :
                out=extract_col(par)
                spec[self.index,col:col+skip] = out[0]
                sig[self.index,col:col+skip] = out[1]
                mask[self.index,col:col+skip] = out[2]
                col+=skip

        if plot is not None:
            plot.clear()
            plot.tv(hd)

        for j,model in enumerate(self.model) :

            i=self.index[j]
            if medfilt is not None :
                boxcar = Box1DKernel(medfilt)
                median = convolve(spec[i,:],boxcar,boundary='extend')
                spec[i,:]/=median
                sig[i,:]/=median

            if plot is not None :
                cr=model(np.arange(ncols))+self.pix0
                if i%2 == 0 : color='b'
                else : color='m'
                plot.ax.plot(range(ncols),cr,color='g',linewidth=3)
                plot.ax.plot(range(ncols),cr-rad,color=color,linewidth=1)
                plot.ax.plot(range(ncols),cr+rad,color=color,linewidth=1)
                if len(back) > 0 :
                    for bk in back:
                        plot.ax.plot(range(ncols),cr+bk[0],color='r',linewidth=1)
                        plot.ax.plot(range(ncols),cr+bk[1],color='r',linewidth=1)
                plot.plotax2.cla()
                plot.plotax2.plot(range(ncols),spec[i],color=color,linewidth=1)
                plot.plotax2.text(0.05,0.95,'Extracted spectrum',
                       transform=plot.plotax2.transAxes)
                plt.draw()
        if plot is not None : 
            while getinput('  See extraction window(s). Hit space bar to continue....',plot.fig)[2] != ' ' :
                pass
        print("")
        return CCDData(spec,uncertainty=StdDevUncertainty(sig),mask=mask,header=hd.header,unit='adu')

  
    def extract2d(self,im,rows=None,plot=None) :
        """  Extract 2D spectrum given trace(s)

             Assumes all requests row uses same trace, just offset, not a 2D model for traces. Linear interpolation is used.
        """
        if self.transpose :
            hd = image.transpose(im)
        else :
            hd = im
        nrows=hd.data.shape[0]
        ncols=hd.data.shape[-1]
        out=[]
        if plot is not None:
            plot.clear()
            plot.tv(hd)
        if rows != None : self.rows = rows

        for model in self.model :
            if plot is not None :
                plot.ax.plot([0,ncols],[self.rows[0],self.rows[0]],color='g')
                plot.ax.plot([0,ncols],[self.rows[1],self.rows[1]],color='g')
                plt.draw()
            outrows=np.arange(self.rows[0],self.rows[1])
            noutrows=len(range(self.rows[0],self.rows[1]))
            spec=np.zeros([noutrows,ncols])
            sig=np.zeros([noutrows,ncols])
            cr=model(np.arange(ncols))
            cr-=cr[self.sc0]
            for col in range(ncols) :
                spec[:,col] = np.interp(outrows+cr[col],np.arange(nrows),
                                        hd.data[:,col])
                sig[:,col] = np.sqrt(np.interp(outrows+cr[col],np.arange(nrows),
                                        hd.uncertainty.array[:,col]**2))
            out.append(CCDData(spec,StdDevUncertainty(sig),unit='adu'))
        if plot is not None: getinput(
                          '  enter something to continue....',plot.fig)

        if len(out) == 1 : return out[0]
        else : return out

def gfit(data,x0,rad=10,sig=3,back=None) :
    """ Fit 1D gaussian
    """ 
    xx = np.arange(x0-rad,x0+rad+1)
    yy = data[x0-rad:x0+rad+1]
    peak=yy.argmax()+x0-rad
    xx = np.arange(peak-rad,peak+rad+1)
    yy = data[peak-rad:peak+rad+1]
    if back == None : p0 = [data[peak],peak,sig]
    else : p0 = [data[peak],peak,sig,back]
    
    coeff, var_matrix = curve_fit(gauss, xx, yy, p0=p0)
    fit = gauss(xx,*coeff)
    print(coeff)
    return coeff

def gauss(x, *p):
    """ Gaussian function
    """
    if len(p) == 3 : 
        A, mu, sigma = p
        back = 0.
    elif len(p) == 4 : 
        A, mu, sigma, back = p
    elif len(p) == 7 : 
        A, mu, sigma, B, Bmu, Bsigma, back = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))+B*np.exp(-(x-Bmu)**2/2.*Bsigma**2)+back

    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+back

def findpeak(x,thresh,diff=10000,bundle=10000) :
    """ Find peaks in vector x above input threshold
        attempts to associate an index with each depending on spacing
    """
    j=[]
    fiber=[]
    f=0
    for i in range(len(x)) :
        if i>0 and i < len(x)-1 and x[i]>x[i-1] and x[i]>x[i+1] and x[i]>thresh :
            #print(i,f)
            j.append(i)
            if len(j)>1 and j[-1]-j[-2] > diff and f%bundle != 0: 
                #print(j[-1],j[-2],f)
                f=f+1
            fiber.append(f)
            f=f+1
          
    return j,fiber
 
def getinput(prompt,fig=None,index=False) :
    """  Get input from terminal or matplotlib figure
    """
    if fig == None : return '','',input(prompt)
    print(prompt)
    get = plots.mark(fig,index=index)
    return get

def extract_col(pars) :
    """ Extract a single column, using boxcar extraction for multiple traces
    """
    data,err,mask,cols,models,rad,pix0,back = pars
    spec = np.zeros([len(models),len(cols)])
    sig = np.zeros([len(models),len(cols)])
    mask = np.zeros([len(models),len(cols)],dtype=bool)
    for i,model in enumerate(models) :
      for j,col in enumerate(cols) :
        cr=model(col)+pix0
        icr=np.round(cr).astype(int)
        rfrac=cr-icr+0.5   # add 0.5 because we rounded
        r1=icr-rad
        r2=icr+rad
        try :
            if r1>=0 and r2<data.size :
                # sum inner pixels directly
                spec[i,j]=np.sum(data[r1+1:r2,j])
                sig[i,j]=np.sum(err[r1+1:r2,j]**2)
                # outer pixels depending on fractional pixel location of trace
                spec[i,j]+=data[r1,j]*(1-rfrac)
                sig[i,j]+=err[r1,j]**2*(1-rfrac)
                spec[i,j]+=data[r2,j]*rfrac
                sig[i,j]+=err[r2,j]**2*rfrac
                sig[i,j]=np.sqrt(sig[i,j])
                mask[i,j] = np.any(mask[r1:r2+1,j]) 
            if len(back) > 0 :
                bpix = np.array([])
                bvar = np.array([])
                for bk in back :
                    bpix=np.append(bpix,data[icr+bk[0]:icr+bk[1],j])
                    bvar=np.append(bvar,err[icr+bk[0]:icr+bk[1],j]**2)
                spec[i,j] -= np.median(bpix)*(r2-r1)
                sig[i,j] = np.sqrt(sig[i,j]**2+np.sum(bvar)/(len(bvar)-1))
          
        except : 
            print('      extraction failed',i,j,col)
            mask[i,j] = True
    return spec,sig, mask
