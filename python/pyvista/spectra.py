import matplotlib
import glob
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
from scipy.ndimage import median_filter, gaussian_filter1d, gaussian_filter
from scipy.linalg import solve_banded
import numpy as np
from numpy.polynomial.chebyshev import chebvander, chebvander2d
import astropy
from astropy.modeling import models, fitting, Parameter
from astropy.nddata import StdDevUncertainty
from pyvista.dataclass import Data
from astropy.io import ascii, fits
from astropy.convolution import convolve, Box1DKernel, Box2DKernel
from astropy.table import Table
import pyvista
import pyvista.data
from pyvista import image, tv, skycalc, bitmask, dataclass
from holtztools import plots
import erfa
from numpy.polynomial import Polynomial


sig2fwhm = 2*np.sqrt(2*np.log(2))

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
                  pix0=0,index=0,hdu=1,orders=None) :
        if file is not None :
            if file == '?' :
                out=glob.glob(
                    str(files(pyvista.data).joinpath('*/*wave*.fits')))
                print('available predefined WaveCals: ')
                for o in out :
                   print(o.split('/')[-2]+'/'+o.split('/')[-1])
                return
            if isinstance(file,astropy.io.fits.fitsrec.FITS_rec) :
                tab=Table(file)
            elif str(file)[0] == '.' or str(file)[0] == '/' :
                tab=Table.read(file,hdu=hdu)
            else :
                tab=Table.read(files(pyvista.data).joinpath(file),hdu=hdu)
            for tag in ['type','degree','ydegree','waves',
                        'waves_order','orders','index',
                        'pix0','pix','y','spectrum','weights','apers','coeff'] :
                if tag in tab.keys() :
                    setattr(self,tag,tab[tag][0])
                else :
                    setattr(self,tag,None)
            self.orders=np.atleast_1d(self.orders)
            # make the initial models from the saved data
            self.model = self.getmod()
            # if we've saved an astropy model, need to redo the fit here
            if self.type != 'chebyshev2D_shift' : self.fit()
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
            if orders is not None :
                self.orders = orders
            else :
                self.orders = [1]
            self.index = None

    def write(self,file,append=False) :
        """ Save object to file

            Parameters 
            ----------
            file : str, name of output file to write to, FITS format
            append : bool, append to existing file (untested)
        """
        tab=Table()
        for tag in ['type','degree','ydegree','waves','waves_order',
                    'orders','index','pix0','pix','y','weights','spectrum','apers','coeff'] :
            try: 
                if getattr(self,tag) is not None : tab[tag] = [getattr(self,tag)]
            except: pass
        if append :
            tab.write(file,append=True)
        else :
            tab.write(file,overwrite=True)
        return tab

    def wave(self,pixels=None,image=None,domain=False) :
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
            if '2D' in self.type :
                out=self.model(pixels[0]-self.pix0,pixels[1])/np.array(self.orders)[pixels[1]]
            else :
                out=self.model(pixels[0]-self.pix0)/self.orders[pixels[1]]
            #out=np.zeros(len(pixels[0]))
            #for i,pixel in enumerate(pixels[0]) :
            #    order=self.orders[pixels[1][i]]
            #    if self.type.find('2D') > 0 :
            #        out[i]=self.model(pixel-self.pix0,pixels[1][i])/order
            #    else :
            #        out[i]=self.model(pixel-self.pix0)/order
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
                        if domain :
                            bd=np.where(((cols-self.pix0) < self.model.domain[0]-10) |
                                    ((cols-self.pix0) > self.model.domain[1]+10) )[0]
                            out[row,bd] = np.nan
            else :
                out= self.model(cols-self.pix0)/self.orders[0]
                if domain :
                    bd=np.where(((cols-self.pix0) < self.model.domain[0]-10) |
                                ((cols-self.pix0) > self.model.domain[1]+10) )[0]
                    out[bd] = np.nan
            return out

    def add_wave(self, hd) :
        """ Add wavelength attribute to input image using current wavelength solution

            Parameters 
            ----------
            hd : Data object
                 Image to add wavelength array to
        """

        hd.add_wave(self.wave(image=np.array(np.atleast_2d(hd.data).shape)))

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
        elif self.type == 'chebyshev2D_shift' :
            mod = Model(func=chebyshev2D_shift,xdegree=self.degree,ydegree=self.ydegree,xdomain=[0,1],ydomain=[0,1],
                        apers=self.apers,coeff=self.coeff)
        else :
            raise ValueError('unknown fitting type: '+self.type)
            return
        return mod

    def plot(self,hard=None) :
        """ Plot current solution
        """

        if self.ax is None :
            fig,ax = plt.subplots(2,1,sharex=True,figsize=(10,5))
            fig.subplots_adjust(hspace=1.05)
            self.fig = fig
            self.ax = ax

        #plot spectrum with current wavelength solution
        self.ax[0].cla()
        wav = self.wave(image=self.spectrum.shape)
        if len(self.spectrum) > 1 :
            row = self.spectrum.shape[0] // 2
            self.ax[0].plot(wav[row,:],self.spectrum[row,:])
        else :
            self.ax[0].plot(wav[0],self.spectrum[0,:])
        for line in self.waves :
            self.ax[0].text(line,1.,'{:7.1f}'.format(line),
                            rotation='vertical',va='top',ha='center')

        # plot residuals
        diff=self.waves-self.wave(pixels=[self.pix,self.y])
        gd = np.where(self.weights > 0.001)[0]
        bd = np.where(self.weights <= 0.)[0]

        self.ax[1].cla()
        if len(self.spectrum) == 1 :
            self.ax[1].plot(self.waves[gd],diff[gd],'go')
        else :
            scat=self.ax[1].scatter(self.waves,diff,marker='o',c=self.y,s=5,cmap='viridis')
            cb_ax = self.fig.add_axes([0.94,0.05,0.02,0.4])
            cb = self.fig.colorbar(scat,cax=cb_ax)
            cb.ax.set_ylabel('Row')

        xlim=self.ax[1].get_xlim()
        self.ax[1].set_ylim(diff.min()-0.5,diff.max()+0.5)
        self.ax[1].plot(xlim,[0,0],linestyle=':')
        self.ax[1].text(0.1,0.9,'rms: {:8.3f} Angstroms'.format(
                                diff[gd].std()),transform=self.ax[1].transAxes)
        self.ax[1].set_xlabel('Wavelength')
        self.ax[1].set_ylabel('obs wave - fit wave')
        if len(bd) > 0 : 
            self.ax[1].scatter(self.waves[bd],diff[bd],c='r',s=5)
        self.ax[1].set_ylim(diff[gd].min()-0.5,diff[gd].max()+0.5)

        self.fig.tight_layout()
        plt.draw()
        if hard is not None :
            self.fig.savefig(hard+'.png')

    def dispersion(self) :
        """ approximate dispersion from 1st order term"
        """
        return self.model.c1.value/(self.model._domain[1]-self.model._domain[0])*2.

    def fit(self,degree=None,reject=3,inter=True) :
        """ do a wavelength fit 

            If a figure has been set in identify, will show fit graphically and
            allow for manual removal of lines in 1D case.  In 2D case, outliers
            are detected and removed
 
            Parameters
            ----------
            degree : int, optional
              degree of polynomial in wavelength, else as previously set in object
        """
        # set up fitter and model
        twod='2D' in self.type
        fitter=fitting.LinearLSQFitter()
        if degree is not None : 
            self.degree=degree
            self.model = self.getmod()
        mod = self.model
        redo = False

        if not hasattr(self,'ax') : self.ax = None
        if twod :
            # for 2D fit, we just use automatic line rejections
            nold=-1
            nbd=0
            while nbd != nold :
                # iterate as long as new points have been removed
                nold=nbd

                if self.type == 'chebyshev2D_shift' :
                    # don't use astropy routines, use curve_fit
                    gd=np.where(self.weights > 0)[0]
                    # quick initial linear fit without shifts
                    design = chebvander2d(self.pix[gd]-self.pix0,self.y[gd],[self.degree,self.ydegree])
                    rhs = self.waves[gd]*self.waves_order[gd]
                    fit=np.linalg.solve(np.dot(design.T,design),np.dot(design.T,rhs))
                    apers = list(set(self.y[gd]))
                    model = Model(func=chebyshev2D_shift,xdegree=self.degree,ydegree=self.ydegree,
                                  xdomain=[0,1],ydomain=[0,1],apers=apers)
                    cfit=curve_fit(model.evaluate,[self.pix[gd]-self.pix0,self.y[gd]],self.waves[gd],
                                   p0=np.append(fit,np.zeros(len(apers)-1)),full_output=True)

                    # note that model will only be defined for apertures used for fit
                    self.coeff = cfit[0]
                    self.apers = apers
                    self.model = Model(func=chebyshev2D_shift,coeff=cfit[0],
                                       xdegree=self.degree,ydegree=self.ydegree,apers=apers,xdomain=[0,1],ydomain=[0,1])
                else :

                    self.model=fitter(mod,self.pix-self.pix0,self.y,
                                      self.waves*self.waves_order,weights=self.weights)

                diff=self.waves-self.wave(pixels=[self.pix,self.y])
                gd = np.where(self.weights > 0)[0]
                print('  rms: {:8.3f}'.format(diff[gd].std()))
                bd = np.where(np.isnan(diff) | (abs(diff) > reject*diff.std()))[0]
                nbd = len(bd)
                print('rejecting {:d} points from {:d} total: '.format(
                      nbd,len(self.waves)))
                self.weights[bd] = 0.

            # plot the results
            if self.ax is not None : 
                self.ax[1].cla()
                scat=self.ax[1].scatter(self.waves,diff,marker='o',
                                        c=self.y,s=5,cmap='viridis')
                plots.plotp(self.ax[1],self.waves[bd],diff[bd],
                            marker='o',color='r',size=5)

                xlim=self.ax[1].get_xlim()
                self.ax[1].set_ylim(np.nanmin(diff)-0.5,np.nanmax(diff)+0.5)
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
                if inter :
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

            gd=np.where(self.weights[irow]>0.)[0]
            diff=self.waves-self.wave(pixels=[self.pix,self.y])
            print('  rms: {:8.3f} Angstroms ({:d} lines)'.format(
                  diff[irow[gd]].std(),len(irow)))
            if self.ax is not None :
                # iterate allowing for interactive removal of points
                done = False
                ymax = self.ax[0].get_ylim()[1]
                print('  Input in plot window: ')
                print('       l : to remove all lines to left of cursor')
                print('       r : to remove all lines to right of cursor')
                print('       n : to remove line nearest cursor x position')
                print('       i : return with True value (to allow iteration)')
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
                    if inter :
                        i = getinput('  input from plot window...', 
                                 self.fig,index=True)
                    else :
                        i = ''
                    if i == '' :
                        done = True
                    elif i[2] == 'l' :
                        bd=np.where(self.waves[irow]<i[0])[0]
                        self.weights[irow[bd]] = 0.
                    elif i[2] == 'r' :
                        bd=np.where(self.waves[irow]>i[0])[0]
                        self.weights[irow[bd]] = 0.
                    elif i[2] == 'u' :
                        bd=np.where(diff>i[1])[0]
                        self.weights[irow[bd]] = 0.
                    elif i[2] == 'd' :
                        bd=np.where(diff<i[1])[0]
                        self.weights[irow[bd]] = 0.
                    elif i[2] == 'n' :
                        #bd=np.argmin(np.abs(self.waves[irow]-i[0]))
                        bd=i[3]
                        self.weights[irow[bd]] = 0.
                    elif i[2] == 'i' :
                        self.weights[:] = 1.
                        redo = True
                        done = True
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
     
          return redo

    def set_spectrum(self,spectrum) :
        """ Set spectrum used to derive fit
        """
        self.spectrum = np.atleast_2d(spectrum)

    def get_spectrum(self) :
        """ Set spectrum used to derive fit
        """
        return self.spectrum 

    def identify(self,spectrum,sky=False,file=None,wav=None,wref=None,inter=False,
                 orders=None,verbose=False,rad=5,thresh=100, fit=True, maxshift=1.e10,
                 disp=None,display=None,plot=None,pixplot=False,domain=False,
                 plotinter=True,
                 xmin=None,xmax=None,lags=range(-300,300), nskip=None, rows=None) :
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

        if sky :
            data = spectrum.sky
            err = spectrum.skyerr
        else :
            data = spectrum.data
            err = spectrum.uncertainty.array

        sz=data.shape
        if len(sz) == 1 : 
            data = np.atleast_2d(data)
            err = np.atleast_2d(err)
            sz=data.shape
        if xmin is None : xmin=0
        if xmax is None : xmax=sz[-1]
        nrow=sz[0]
        if orders is not None : self.orders = orders

        # specify rows to find lines in (e.g., for 2D)
        if rows is None :
            if  nskip is None :
                if len(set(self.orders)) == 1 : nskip=25
                else : nskip=1
            rows = range(0,nrow,nskip)

        # get initial reference wavelengths if not given
        if wav is None :
            pix=np.arange(sz[-1])
            if inter :
                f,a=plots.multi(1,1)
                a.plot(data[0,:])
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
            elif self.spectrum is not None :
                # cross correlate with reference image to get pixel shift
                print('  cross correlating with reference spectrum using lags between: ', lags[0],lags[-1])
                fitpeak,shift = image.xcorr(self.spectrum,data,lags)
                if shift.ndim == 1 :
                    pixshift=(fitpeak+lags[0])[0]
                    print('  Derived pixel shift from input wcal: ',fitpeak+lags[0])
                    if display is not None :
                        display.plotax1.cla()
                        display.plotax1.text(0.05,0.95,'spectrum and reference',
                                             transform=display.plotax1.transAxes)
                        for row in range(data.shape[0]) :
                            display.plotax1.plot(data[row,:],color='m')
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
                    wav=np.atleast_2d(self.wave(image=np.array(sz),domain=domain))
                    #wav=np.atleast_2d(self.wave(image=sz))
                else :
                    # different shift for each row
                    wav=np.zeros(sz)
                    cols = np.arange(sz[-1])
                    orders=[]
                    for row in range(wav.shape[0]) : 
                        print('  Derived pixel shift from input wcal for row: {:d} {:d}'.format
                               (row,shift[row,:].argmax()+lags[0]),end='\r')
                        #rows=np.zeros(len(cols))+row
                        try : order = self.orders[row]
                        except : order=self.orders[0]
                        orders.append(order)
                        pix0 = self.pix0+fitpeak[row]+lags[0]
                        wav[row,:] = self.model(cols-pix0)/order
                    # ensure we have 2D fit
                    self.type = 'chebyshev2D'
                    self.model = self.getmod()
                    self.orders = orders
                    print("")
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
        pix0=[]
        row0=[]
        y=[]
        fwhm=[]
        waves=[]
        waves_order=[]
        weight=[]
        diff=[]
        if display is not None and  isinstance(display,pyvista.tv.TV) :
            display.ax.cla()
            display.ax.axis('off')
            display.tv(data)
        if plot is None or plot == False:
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

        if self.ax is not None : ax[0].cla()
        peaks=[]
        allrows=[]
        gdpeaks=[]
        gdrows=[]
        for row in rows :
            if verbose :print('  identifying lines in row: ', row,end='\r')
            if self.ax is not None :
                # next line for pixel plot
                if pixplot : ax[0].plot(data[row,:])
                else : ax[0].plot(wav[row,:],data[row,:])
                #ax[0].set_yscale('log')
                ax[0].set_ylim(1.,ax[0].get_ylim()[1])
                ax[0].text(0.1,0.9,'row: {:d}'.format(row),transform=ax[0].transAxes)
                ax[0].set_xlabel('Rough wavelength')
                ax[0].set_ylabel('Intensity')
            for line in lines :
                # initial guess from input wavelengths
                peak0=np.nanargmin(abs(line-wav[row,:]))
                peak=np.nanargmin(abs(line-wav[row,:]))
                if ( (data[row,peak] > 1) and (peak > xmin+rad) and (peak < xmax-rad)) :
                  # set peak to highest nearby pixel
                  peak=(data[row,peak-rad:peak+rad+1]).argmax()+peak-rad
                  if ( (peak < xmin+rad) or (peak > xmax-rad)) : continue
                  if isinstance(display,pyvista.tv.TV) :
                      peaks.append(peak)
                      allrows.append(row)
                  # S/N threshold
                  if (data[row,peak-rad:peak+rad+1]/
                      err[row,peak-rad:peak+rad+1]).max() > thresh:

                    oldpeak = 0
                    niter=0
                    xx = np.arange(peak-rad,peak+rad+1)
                    yy = data[row,peak-rad:peak+rad+1]
                    try :  
                      while peak != oldpeak and  niter<10:
                        p0 = [data[row,peak],peak,rad/sig2fwhm,0.]
                        coeff, var_matrix = curve_fit(gauss, xx, yy, p0=p0)
                        cent = coeff[1]
                        oldpeak = peak
                        peak = int(cent)
                        niter+=1
                      #if niter == 10 : continue
                    except : 
                        continue
                    if verbose : print(line,peak,*coeff)
                    if display is not None and  isinstance(display,pyvista.tv.TV) :
                        gdpeaks.append(cent)
                        gdrows.append(row)
                    if plot is not None and plot != False :
                        if pixplot :
                            ax[0].plot([cent,cent],ax[0].get_ylim(),color='r')
                        else : 
                            ax[0].text(line,1.,'{:7.1f}'.format(line),
                                       rotation='vertical',va='top',ha='center')
                    x.append(cent)
                    y.append(row)
                    fwhm.append(np.abs(coeff[2]*sig2fwhm))
                    # we will fit for wavelength*order
                    waves.append(line)
                    try: order = self.orders[row]
                    except: order=self.orders[0]
                    waves_order.append(order)
                    if np.abs(cent-peak0) < maxshift : wt=1.
                    else : 
                        print('bad: ',row,cent,peak0)
                        wt=0.
                    weight.append(wt)
        if isinstance(display,pyvista.tv.TV) :
            display.ax.scatter(peaks,allrows,marker='o',color='r',s=2)
            display.ax.scatter(gdpeaks,gdrows,marker='o',color='g',s=2)
        if self.ax is not None : 
            self.fig.tight_layout()
            print('  See identified lines.')
        self.pix=np.array(x)
        self.y=np.array(y)
        self.fwhm=np.array(fwhm)
        self.waves=np.array(waves)
        self.waves_order=np.array(waves_order)
        self.weights=np.array(weight)
        self.spectrum = data

        redo = False
        if fit: 
            redo = self.fit(inter=plotinter)
            spectrum.add_wave(self.wave(image=spectrum.data.shape))
        print('')

        return redo

    def skyline(self,hd,plot=True,thresh=50,inter=True,linear=False,file='skyline.dat',rows=None) :
        """ Adjust wavelength solution based on sky lines

            Parameters
            ----------
            hd : Data object
                 input pyvista Data object, must contain wave attribute with initial wavelengths
            plot : bool, default=True
                   display plot results
            thresh : float, default=50
                   minimum S/N for line detection
            rows : array-like, default=None
                   if specified, only use specified rows for sky spectrum, relevant for 2D correction
                   to ignore object rows
            inter : bool, default=True
                   allow for interactive removal of lines
            linear : bool, default=False
                   if True, allow for dispersion to be ajusted as well as wavelength zeropoint 
                   requires at least two sky lines!
            file : str, default='skyline.dat'
                   file with sky lines to look for, if you want to override default
        """

        if hd.wave is None :
            raise ValueError('input object must contain wave attribute')

        # set higher order terms to fixed
        if self.type == 'chebyshev2D' :
            for i in range(self.degree) :
                self.model.fixed['c{:d}_1'.format(i+1)] = True
                self.model.fixed['c{:d}_2'.format(i+1)] = True
                if not linear or i>0 :
                    self.model.fixed['c{:d}_0'.format(i+1)] = True
        else :
            for i in range(self.degree) :
                if not linear or i>0 :
                    self.model.fixed['c{:d}'.format(i+1)] = True

        if hd.sky is not None and hd.skyerr is not None : sky=True
        else : sky=False
        self.identify(hd,wav=hd.wave,file=file,plot=plot,thresh=thresh,
                      plotinter=inter,sky=sky,rows=rows)

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
        mask=np.zeros(len(wav),dtype=np.uintc)
        pixmask=bitmask.PixelBitMask()
        # raw wavelengths
        w=self.wave(image=np.array(np.atleast_2d(hd.data).shape))
        for i in range(np.atleast_2d(hd).shape[0]) :
            sort=np.argsort(w[i,:])
            if usemask : 
                gd = np.where(~(np.atleast_2d(hd.bitmask&pixmask.badval()))[i,sort])
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
        return Data(out,uncertainty=StdDevUncertainty(sig),
                       bitmask=mask,header=hd.header,wave=wav)


    def correct(self,hd,wav) :
        """ Resample input image to desired wavelength scale

        Uses current wavelength solution, linearly interpolates to specified
          wavelengths, on a row-by-row basis.

        Parameters
        ----------
        hd : Data, input image to resample
        wav : array_like, new wavelengths to interpolate to

        """

        out=np.zeros([hd.data.shape[0],len(wav)])
        sig=np.zeros_like(out)
        wave=np.zeros([hd.data.shape[0],len(wav)])
        bitmask=np.zeros_like(out,dtype=hd.bitmask.dtype)
        w=self.wave(image=hd.data.shape)
        for i in range(len(out)) :
            sort=np.argsort(w[i,:])
            wmin=w[i,sort].min()
            wmax=w[i,sort].max()
            w2=np.abs(wav-wmin).argmin()
            w1=np.abs(wav-wmax).argmin()
            out[i,:] += np.interp(wav,w[i,sort],hd.data[i,sort])
            sig[i,:] += np.sqrt(
                            np.interp(wav,w[i,sort],hd.uncertainty.array[i,sort]**2))
            wave[i,:] = wav
            for bit in range(0,32) :
                mask = (hd.bitmask[i,sort] & 2**bit)
                if mask.max() > 0 :
                    maskint = np.interp(wav,w[i,sort],mask)
                    bitset = np.where(maskint>0)[0] 
                    bitmask[i,bitset] |= 2**bit

        return Data(out,header=hd.header,uncertainty=StdDevUncertainty(sig),bitmask=bitmask,wave=wave)

class Trace() :
    """ Class for spectral traces

    Attributes 
    ----------
    type : str
        type of astropy model to use
    degree : int
        polynomial degree to use for trace
    sigdegree : int
        polynomial degree to use for fitting gaussian sigma trace width
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
        filename for FITS file with Trace attributes

    """

    def __init__ (self,file=None,inst=None, type='Polynomial1D',degree=2,sigdegree=0,
                  pix0=0,rad=5, spectrum=None,model=None,sc0=None,rows=None,
                  transpose=False,lags=None,channel=None,hdu=1) :

        if file is not None :
            """ Initialize object from FITS file
            """
            if file == '?' :
                out=glob.glob(
                    str(files(pyvista.data).joinpath('*/*trace*.fits')))
                print('available predefined traces: ')
                for o in out :
                   print(o.split('/')[-2]+'/'+o.split('/')[-1])
                return
            try:
                if str(file)[0] == '.' or str(file)[0] == '/' :
                    tab=Table.read(file,hdu=hdu)
                else :
                    tab=Table.read(files(pyvista.data).joinpath(file),hdu=hdu)
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
            # use saved coefficients to instantiate model
            coeffs = tab['coeffs'][0]
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
            try :
                self.sigdegree = tab['sigdegree'][0]
            except :
                self.sigdegree = sigdegree
            try :
                sigcoeffs = tab['sigcoeffs'][0]
                self.sigmodel = []
                for row in sigcoeffs :
                    if self.type == 'Polynomial1D' :
                        kwargs={}
                        for i,c in enumerate(row) :
                            name='c{:d}'.format(i)
                            kwargs[name] = c
                        self.sigmodel.append(
                            models.Polynomial1D(degree=self.degree,**kwargs))
                    else :
                        raise ValueError('Only Polynomial1D currently implemented')
            except KeyError :
                sigcoeffs = None
                self.sigmodel = None
            return

        self.type = type
        self.degree = degree
        self.sigdegree = sigdegree
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
        else :
            self.lags = range(-50,50) 
        if rows is not None : self.rows=rows
        if lags is not None : self.lags=lags
        if model is not None : self.model=model
        else : self.model=None
        self.sigmodel=None
        if sc0 is not None : self.sc0=sc0
        else : self.sc0 = None

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
        if self.sigmodel is not None :
            coeffs = []
            if self.type == 'Polynomial1D' :
                # load model coefficients
                for m in self.sigmodel :
                    row=[]
                    for i in range(self.degree+1) :
                        name='c{:d}'.format(i)
                        row.append(getattr(m,name).value)
                    coeffs.append(row)
            tab['sigcoeffs'] = [np.array(coeffs)]
        if append :
            tab.write(file,append=True)
        else :
            tab.write(file,overwrite=True)
        return tab


    def trace(self,im,srows,sc0=None,plot=None,display=None,
              rad=None, thresh=20,index=None,skip=10,gaussian=False,verbose=False) :
        """ Trace a spectrum from starting position

            Parameters
            ----------
            im : Data
                 input image
            srows : array-like
                 location(s) at sc0 for initial trace location(s) guess(es)
            rad : float, optional, default=self.rad
                 radius of window to use to find trace locations
            index : integer, optional, default=None
                 index to label trace(s) with
            skip : integer, optional, default=10
                 measure trace center every skip pixels, using median of 
                 data from -skip/2 to skip/2
            gaussian : bool, optional, default=False
                 if True, use gaussian fit for trace location instead of centroid. 
                 with gaussian=True, will also fit trace widths into
                 sigmodel, with polynomial of degree self.sigdegree
            sc0 : integer, optional, default=ncol/2
            plot : bool, optional, default=None
            display : TV object, optional, default=None
        """

        if plot == None and display != None : plot = display

        fitter=fitting.LinearLSQFitter()
        if self.type == 'Polynomial1D' :
            mod=models.Polynomial1D(degree=self.degree)
            sigmod=models.Polynomial1D(degree=self.sigdegree)
        else :
            raise ValueError('unknown fitting type: '+self.type)
            return

        if index is None : 
            index = np.arange(len(srows))
        elif len(index) != len(srows) :
            raise ValueError('length of index and srows must be the same')

        if self.transpose :
            hd = dataclass.transpose(im)
        else :
            hd = im

        nrow = hd.data.shape[0]
        ncol = hd.data.shape[1]
        if sc0 is None : 
            if self.sc0 is None : self.sc0 = ncol//2
        else : self.sc0 = sc0
        self.spectrum = hd.data[:,self.sc0]
        self.spectrum[self.spectrum<0] = 0.
        spectrum=np.zeros([len(hd.data),len(range(skip//2,ncol,skip))])
        for icol,col in enumerate(range(skip//2,ncol,skip)) :
            spectrum[:,icol]= np.median(hd.data[:,col-skip//2:col+skip//2+1],axis=1)
        self.spectrum2d=spectrum
        self.skip=skip
        self.cols2d=np.arange(skip//2,ncol,skip)
        rows = np.arange(nrow)
        ypos = np.zeros(ncol)
        ysum = np.zeros(ncol)
        yvar = np.zeros(ncol)
        ymask = np.ones(ncol,dtype=bool)
        ysig = np.zeros(ncol)
        pixmask = bitmask.PixelBitMask()

        # we want to handle multiple traces, so make sure srows is iterable
        if type(srows ) is int or type(srows) is float : srows=[srows]
        self.model=[]
        if gaussian : 
            self.sigmodel=[]
            #fig,ax=plots.multi(1,1)
        if plot : 
            plot.clear()
            plot.tv(hd)
            plot.plotax2.cla()

        if rad is None : rad = self.rad-1
        self.cols=[]
        self.ypos=[]
        newindex=[]
        for irow,(srow,ind) in enumerate(zip(srows,index)) :
            try:
                print('  Tracing row: {:d}'.format(int(srow)),end='\r')
            except: 
                pdb.set_trace()
                continue
            sr=copy.copy(srow)
            sr=int(round(sr))
            sr=hd.data[sr-rad:sr+rad+1,self.sc0].argmax()+sr-rad
            # march left from center
            for col in range(self.sc0,skip//2,-skip) :
                # centroid, calculate for S/N even if we have gaussian
                data = np.median(hd.data[sr-rad:sr+rad+1,col-skip//2:col+skip//2+1],axis=1)
                var = np.median(hd.uncertainty.array[sr-rad:sr+rad+1,col-skip//2:col+skip//2+1]**2,axis=1)/(2*skip)
                cr=sr-rad+data.argmax()
                data = np.median(hd.data[cr-rad:cr+rad+1,col-skip//2:col+skip//2+1],axis=1)
                var = np.median(hd.uncertainty.array[cr-rad:cr+rad+1,col-skip//2:col+skip//2+1]**2,axis=1)/(2*skip)

                ysum[col] = np.sum(data)
                ypos[col] = np.sum(rows[cr-rad:cr+rad+1]*data) / ysum[col]
                yvar[col] = np.sum(var)
                if gaussian :
                    # gaussian fit
                    data = np.median(hd.data[cr-rad-2:cr+rad+2,col-skip//2:col+skip//2+1],axis=1)
                    try : 
                        gcoeff=gfit(data, rad+2,rad=rad,sig=2,back=True)
                        ypos[col] = gcoeff[1]+cr-rad-2
                        ysig[col] = gcoeff[2]
                        ymask[col] = False
                    except : 
                        pass
                else :
                    ymask[col] = False

                # if centroid is too far from starting guess, mask as bad
                if np.abs(ypos[col]-sr) > rad or  (ysum[col]/np.sqrt(yvar[col]) < thresh)  : 
                    ymask[col] = True

                # use this position as starting center for next 
                #if above threshold S/N
                if ((not ymask[col]) & np.isfinite(ysum[col]) & 
                    (ysum[col]/np.sqrt(yvar[col]) > thresh) )  : 
                        sr=int(round(ypos[col]))

            # march right from center
            sr=copy.copy(srow)
            sr=int(round(sr))
            sr=hd.data[sr-rad:sr+rad+1,self.sc0].argmax()+sr-rad

            for col in range(self.sc0+skip,ncol,skip) :
                # centroid, calculate for S/N even if we have gaussian
                data = np.median(hd.data[sr-rad:sr+rad+1,col-skip//2:col+skip//2+1],axis=1)
                var = np.median(hd.uncertainty.array[sr-rad:sr+rad+1,col-skip//2:col+skip//2+1]**2,axis=1)/(2*skip)
                cr=sr-rad+data.argmax()
                data = np.median(hd.data[cr-rad:cr+rad+1,col-skip//2:col+skip//2+1],axis=1)
                var = np.median(hd.uncertainty.array[cr-rad:cr+rad+1,col-skip//2:col+skip//2+1]**2,axis=1)/(2*skip)

                ysum[col] = np.sum(data)
                ypos[col] = np.sum(rows[cr-rad:cr+rad+1]*data) / ysum[col]
                yvar[col] = np.sum(var)
                if gaussian :
                    # gaussian fit
                    data = np.median(hd.data[cr-rad-2:cr+rad+2,col-skip//2:col+skip//2+1],axis=1)
                    try : 
                        gcoeff=gfit(data, rad+2,rad=rad,sig=2,back=True)
                        ypos[col] = gcoeff[1]+cr-rad-2
                        ysig[col] = gcoeff[2]
                        ymask[col] = False
                    except : 
                        pass
                else :
                    ymask[col] = False

                # use this position as starting center for next if above threshold S/N
                if np.abs(ypos[col]-sr) > rad or (ysum[col]/np.sqrt(yvar[col]) < thresh) : 
                    ymask[col] = True
                if ((not ymask[col]) & np.isfinite(ysum[col]) & 
                    (ysum[col]/np.sqrt(yvar[col]) > thresh) ) : 
                        sr=int(round(ypos[col]))

            # do a fit to the measured locations
            cols=np.arange(ncol)
            gd = np.where((~ymask) & (ysum/np.sqrt(yvar)>thresh) )[0]
            model=(fitter(mod,cols[gd],ypos[gd]))
            res = model(cols)-ypos

            # reject outlier points (>1 pixel) and refit
            gd = np.where((~ymask) & (ysum/np.sqrt(yvar)>thresh) & (np.abs(res)<rad))[0]
            model=(fitter(mod,cols[gd],ypos[gd]))
            if gaussian: 
                sigmodel=(fitter(sigmod,cols[gd],ysig[gd]))
                self.sigmodel.append(sigmodel)
            if len(gd) < self.degree*5 : 
                print('  failed trace for row: {:d} {:d} {:d}'.format(irow,len(gd),len(res)))
                continue

            self.model.append(model)
            newindex.append(ind)

            # plot model here, save all points for single plot command later for matplotlib efficiency and to return
            self.cols.append(cols[gd])
            self.ypos.append(ypos[gd])

            if plot :
                plot.ax.plot(cols,model(cols),color='m')
                plot.plotax2.plot(cols,model(cols),color='m')
                plot.plotax2.text(0.05,0.95,'Derived trace',
                       transform=plot.plotax2.transAxes)

        if plot : 
            # append all points for a single matplotlib call
            allcols=[]
            allpos=[]
            for col,ypos in zip(self.cols,self.ypos) :
                allcols.extend(col)
                allpos.extend(ypos)
            allpos=np.array(allpos)
            allcols=np.array(allcols)
            valid = np.where(allpos>0.)[0]
            plot.ax.scatter(allcols[valid],allpos[valid],marker='o',color='r',s=50) 
            plot.ax.scatter(allcols,allpos,marker='o',color='g',s=50) 
            plt.draw()

        self.pix0=0
        if index is not None : self.index = np.array(newindex)
        else : self.index=np.arange(len(self.model))
        print("")
        if plot : 
            while getinput('  See trace. Hit space bar to continue....',plot.fig)[2] != ' ' :
                pass

    def plot(self) :
        """ Plots points and fits for traces
        """
        plt.figure()
        x=np.arange(self.rows[0],self.rows[1])
        for cols,ypos,mod,fiber in zip(self.cols,self.ypos,self.model,self.index) :
            plt.scatter(cols,ypos)
            plt.plot(x,mod(x))
            plt.text(x[-1],mod(x[-1]),'{:d}'.format(fiber))

    def retrace(self,hd,plot=None,display=None,thresh=20,gaussian=False,skip=10,rad=3) :
        """ Retrace starting with existing model
        """
        if plot == None and display != None : plot = display
        self.find(hd,rad=rad)
        srows = []
        for row in range(len(self.model)) :
            srows.append(self.model[row](self.sc0)+self.pix0)
        self.trace(hd,srows,plot=plot,thresh=thresh,gaussian=gaussian,skip=skip,index=self.index)
    
    def findpeak(self,hd,sc0=None,width=100,thresh=50,plot=False,sort=False,
                 back_percentile=10,method='linear',
                 smooth=5,diff=10000,bundle=10000,verbose=False) :
        """ Find peaks in spatial profile for subsequent tracing

            Parameters
            ----------
            hd : Data object
                 Input image
            sc0 : int, default=None
                 pixel location of wavelength to make spatial profile around
                 if none, use sc0 defined in trace
            width : int, default=100
                 width of window around specfied wavelength to median 
                 to give spatial profile
            thresh : float, default = 50
                 threshold for finding objects, as a factor to be 
                 multiplied by the median uncertainty
            smooth : float, default = 5
                 smoothing FWHM (pixels) for cross-section before peak finding
            sort : bool, default=False
                 return peaks sorted with brightest first

            Returns
            -------
            tuple : list of peak locations, and list of indices
                    peak locations can be passed to trace()

        """
        if self.transpose :
            im = dataclass.transpose(hd)
        else :
            im = copy.deepcopy(hd)

        if sc0 is None : 
            if self.sc0 is None: self.sc0 = im.data.shape[1]//2
            sc0 = self.sc0

        print('looking for peaks using {:d} pixels around {:d}, threshhold of {:f}'.
              format(2*width,sc0,thresh))

        nrows=im.data.shape[0]
        if self.rows is None : self.rows=[0,nrows]

        back =np.percentile(im.data[self.rows[0]:self.rows[1],
                            sc0-width:sc0+width],back_percentile,
                            method=method)
        sig =np.median(im.uncertainty.array[self.rows[0]:self.rows[1],
                                    sc0-width:sc0+width])/np.sqrt(2*width)

        data = np.median(im.data[self.rows[0]:self.rows[1],
                                 sc0-width:sc0+width],axis=1)-back
        if smooth > 0 : data = gaussian_filter1d(data, smooth/sig2fwhm)

        if plot :
            plt.figure()
            plt.plot(np.arange(self.rows[0],self.rows[1]),
                     np.median(im.data[self.rows[0]:self.rows[1],
                               self.sc0-width:self.sc0+width],axis=1)-back)
            plt.plot(data)
            plt.xlabel('Spatial pixel')
            plt.ylabel('Median flux')
        
        peaks,fiber = findpeak(data, thresh=thresh*sig, diff=diff, 
                               bundle=bundle,verbose=verbose,sort=sort)
        print('peaks: ',peaks)
        print('aperture/fiber: ',fiber)
        return np.array(peaks)+self.rows[0], fiber

 
    def find(self,hd,sc0=None,width=100,lags=None,plot=None,display=None,inter=False,rad=3) :
        """ Determine shift from existing trace to input frame

            Parameters
            ----------
            hd : Data object
                 Input image
            width : int, default=100
                 width of window around central wavelength to median 
                 to give spatial profile
            lags : array-like, default=self.lags
                 range of cross-correlation lags to allow
            rad : int, default=3
                 radius around xcorr peak to do polynomial fit to
            display : pyvista.tv object, default=None
                 if not None, tv object to display in
        """
        if lags is None : lags = self.lags
        if plot == None and display != None : plot = display

        if self.transpose :
            im = dataclass.transpose(hd)
        else :
            im = copy.deepcopy(hd)

        if inter :
            try : display.tv(im)
            except : raise ValueError('must use display= with inter=True')
            print('Hit "f" on location of spectrum: ')
            button,x,y=display.tvmark()
            self.pix0=y-self.model[0](x)
            print('setting trace offset to: ', self.pix0)
            return 

        #for icol,col in enumerate(self.cols2d) :
        #    lags=np.arange(-9,10)
        #    spec=np.median(im.data[:,col-self.skip//2:col+self.skip//2+1],axis=1)
        #    fitpeak,shift = image.xcorr(self.spectrum2d[:,icol],spec,lags,rad=rad)
        #    pixshift=(fitpeak+lags[0])[0]
        #    print('  Derived pixel shift from input trace: ',pixshift)
      
        # get median around central column
        if sc0 is None : 
            if self.sc0 is None: self.sc0 = im.data.shape[1]//2
            sc0 = self.sc0
        spec=np.median(im.data[:,sc0-width:sc0+width],axis=1)

        # if we have a window, zero array outside of window
        try:
            spec[:self.rows[0]] = 0.  
            spec[self.rows[1]:] = 0.  
        except: pass

        # cross-correlate with saved spectrum to get shift
        fitpeak,shift = image.xcorr(self.spectrum,spec,lags,rad=rad)
        pixshift=(fitpeak+lags[0])[0]
        print('  Derived pixel shift from input trace: ',pixshift)
        if plot is not None :
            plot.clear()
            plot.tv(im)
            plot.plotax1.cla()
            plot.plotax1.text(0.05,0.95,'obj and ref cross-section',
                              transform=plot.plotax1.transAxes)
            plot.plotax1.plot(self.spectrum/self.spectrum.max())
            plot.plotax1.plot(im.data[:,sc0]/im.data[:,sc0].max())
            plot.plotax1.set_xlabel('row')
            plot.histclick=False
            plot.plotax2.cla()
            plot.plotax2.text(0.05,0.95,
                              'cross correlation {:8.3f}'.format(pixshift),
                              transform=plot.plotax2.transAxes)
            plot.plotax2.plot(lags,shift)
            plot.plotax2.set_xlabel('lag')
            plt.draw()
            getinput('  See spectra and cross-correlation.\n'+
                     '  Hit any key in display window to continue....',plot.fig)
        self.pix0=fitpeak+lags[0]
        self.pix0=pixshift
        return fitpeak+lags[0]

    def immodel(self,im,ext,threads=0) :
        """ Create model 2D image from input fluxes
        """
        if self.transpose :
            new = (im.data*0.).T
        else : 
            new = im.data*0.
        ncols = im.data.shape[1]

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
            pars.append((ext.data[:,col*skip:ec],new.shape[0],
                         np.arange(col*skip,ec),self.model,self.sigmodel,self.index))

        if threads > 0 :
            pool = mp.Pool(threads)
            output = pool.map_async(model_col, pars).get()
            pool.close()
            pool.join()
            col=0
            for out in output :
                nc=out.shape[1]
                new[:,col:col+nc] = out
                col+=skip
        else :
            col=0
            for par in pars :
                out=model_col(par)
                new[:,col:col+skip] = out
                col+=skip

        if self.transpose : return new.T
        else : return new


    def extract(self,im,rad=None,back=[],fit=False,old=False,
                display=None,plot=None,medfilt=None,nout=None,threads=0) :
        """ Extract spectrum given trace(s)

            Parameters
            ----------
            hd : Data object
                 Input image
            rad : float, default=self.rad
                 radius for extraction window
            back : array-like of array-like
                 list of two-element lists giving start and end of
                 background window(s), in units of pixels relative to
                 trace location
            nout : integer, default=None
                 used for multi-object spectra.
                 If not None, specifies number of rows of output image;
                 each extracted spectrum will be loaded into indices
                 loaded into index attribute, with an index for each trace
        """
        if plot == None and display != None : plot = display
        if self.transpose :
            hd = dataclass.transpose(im)
        else :
            hd = copy.deepcopy(im)

        if hd.bitmask is None :
            hd.add_bitmask(np.zeros_like(hd.data,dtype=np.uintc))

        if fit and (self.sigmodel is None or len(self.sigmodel) == 0) :
            raise ValueError('must have a sigmodel to use fit extraction.'+
                             'Use gaussian=True in trace')

        if rad is None : rad=self.rad
        if back is None : back = []
        if len(back) > 0 :
            for bk in back:
                try :
                    if (len(bk) != 2 or not isinstance(bk[0],int) 
                       or not isinstance(bk[1],int) ) :
                        raise ValueError('back must be list of [backlo,backhi] integer pairs')
                except :
                    raise ValueError('back must be list of [backlo,backhi] integer pairs')
        nrows=hd.data.shape[0]
        ncols=hd.data.shape[-1]
        if nout is not None :
            spec = np.zeros([nout,hd.data.shape[1]])
            sig = np.zeros([nout,hd.data.shape[1]])
            bitmask = np.zeros([nout,hd.data.shape[1]],dtype=np.uintc)
            background_spec = np.zeros([nout,hd.data.shape[1]])
            background_spec_err = np.zeros([nout,hd.data.shape[1]])
        else :
            spec = np.zeros([len(self.model),hd.data.shape[1]])
            sig = np.zeros([len(self.model),hd.data.shape[1]])
            bitmask = np.zeros([len(self.model),hd.data.shape[1]],dtype=np.uintc)
            background_spec = np.zeros([len(self.model),hd.data.shape[1]])
            background_spec_err = np.zeros([len(self.model),hd.data.shape[1]])

        pars=[]
        if threads == 0 : 
            skip=1
            npars=ncols
            skip=ncols
            npars=1
        else : 
            skip=ncols//threads
            npars=threads
        for col in range(npars) :
            if col == threads-1 : ec=ncols
            else : ec=col*skip+skip
            pars.append((hd.data[:,col*skip:ec],
                         hd.uncertainty.array[:,col*skip:ec],
                         hd.bitmask[:,col*skip:ec],
                         np.arange(col*skip,ec),
                         self.model,rad,self.pix0,back,self.sigmodel))

        print('  extracting ... ')

        if threads > 0 :
            pool = mp.Pool(threads)
            if fit : output = pool.map_async(extract_col_fit, pars).get()
            elif old : output = pool.map_async(extract_col_old, pars).get()
            else : output = pool.map_async(extract_col, pars).get()
            pool.close()
            pool.join()
            col=0
            for out in output :
                nc=out[0].shape[1]
                spec[self.index,col:col+nc] = out[0]
                sig[self.index,col:col+nc] = out[1]
                bitmask[self.index,col:col+nc] = out[2]
                background_spec[self.index,col:col+nc] = out[3]
                background_spec_err[self.index,col:col+nc] = out[4]
                col+=skip
        else :
            col=0
            for par in pars :
                if fit : out=extract_col_fit(par)
                elif old : 
                  print('  may take some time, consider threads=')
                  out=extract_col_old(par)
                else : out=extract_col(par)
                spec[self.index,col:col+skip] = out[0]
                sig[self.index,col:col+skip] = out[1]
                bitmask[self.index,col:col+skip] = out[2]
                background_spec[self.index,col:col+skip] = out[3]
                background_spec_err[self.index,col:col+skip] = out[4]
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
        return Data(spec,uncertainty=StdDevUncertainty(sig),
                    bitmask=bitmask,header=hd.header,sky=background_spec,skyerr=background_spec_err)

  
    def extract2d(self,im,rows=None,plot=None,display=None,buffer=0) :
        """  Extract 2D spectrum given trace(s)

             Assumes all requested rows uses same trace, just offset, 
             not a 2D model for traces. Linear interpolation is used.
        """
        if plot == None and display != None : plot = display
        if self.transpose :
            hd = dataclass.transpose(im)
        else :
            hd = im
        nrows=hd.data.shape[0]
        ncols=hd.data.shape[-1]
        out=[]
        if plot is not None:
            plot.clear()
            plot.tv(hd)
        if rows != None : self.rows = rows
        if self.rows is None : self.rows=[0,nrows]
        if self.sc0 is None : self.sc0 = ncols//2

        if not isinstance(self.rows[0],list) :
            rows = [self.rows]
        else :
            rows = self.rows

        print('extracting: ')
        for rows,model in zip(rows,self.model) :
            outrows=np.arange(rows[0]+buffer,rows[1]-buffer)
            noutrows=len(range(rows[0]+buffer,rows[1]-buffer))
            spec=np.zeros([noutrows,ncols])
            sig=np.zeros([noutrows,ncols])
            bitmask=np.zeros([noutrows,ncols],dtype=np.uintc)
            cr=model(np.arange(ncols))
            cr-=cr[self.sc0]
            if plot is not None :
                #plot.ax.plot([0,ncols],[rows[0],rows[0]],color='g')
                #plot.ax.plot([0,ncols],[rows[1],rows[1]],color='g')
                plot.ax.plot(np.arange(ncols),cr+rows[0],color='g')
                plot.ax.plot(np.arange(ncols),cr+rows[1],color='g')
                plt.draw()
            for col in range(ncols) :
                spec[:,col] = np.interp(outrows+cr[col],np.arange(nrows),
                                        hd.data[:,col])
                sig[:,col] = np.sqrt(np.interp(outrows+cr[col],np.arange(nrows),
                                        hd.uncertainty.array[:,col]**2))
                for bit in range(0,32) :
                    mask = (hd.bitmask[:,col] & 2**bit)
                    if mask.max() > 0 :
                        maskint = np.interp(outrows+cr[col],np.arange(nrows),mask)
                        bitset = np.where(maskint>0)[0] 
                        bitmask[bitset,col] |= 2**bit
            print(' {:d}-{:d}'.format(rows[0],rows[1]))
            header=copy.copy(hd.header)
            header['CNPIX2'] = rows[0]
            out.append(Data(spec,StdDevUncertainty(sig),
                            bitmask=bitmask,header=header))

        if plot is not None: 
           while getinput('  See extraction window(s). Hit space bar to continue....',plot.fig)[2] != ' ' :
               pass
        if len(out) == 1 : return out[0]
        else : return out


    def findslits(self,im,smooth=3,thresh=1500,display=None,cent=None,degree=2,skip=50,sn=False) :
        """ Find slits in a multi-slit flat field image
   
        findslits() attempts to find slit locations by looking for peaks
        in the derivative of the flux (or, if sn=True, in the derivative
        of S/N) in a window of skip//2 pixels around the center of the
        image (or around the location given by cent=). Taking those values,
        it then looks for corresponding edges moving to the left and
        right of the center. Finally, it fits a polynomial to each set
        of slit edges to populate the model attribute of the Trace object.
        It also populates the rows attribute of the Trace object with bottom 
        and top locations of the slit in the center of the detector.
 
        Parameters
        ----------
        data : array or Data object, wavelength dimension horizontal
           input data to find slit edges, usually a flat field
        smooth : float, optional, default=3
               smoothing radius for calculated derivatives
        sn : boolean, optional, default=False
               if True, look for edges in delta(S/N)
        thresh : float, optional, default=1500
               threshold for detecting edges in delta(signal)
        skip : integer, optional, default=50
               spacing of where along spectra to identify edges
        display : TV object, optional
               TV object to display derivatives and slit locations 
        cent :  int, optional, default=None
               location of center of spectra, if None use chip center
        degree : int, option, degree=2
               polynomial degree to use to fit edge locations
        
    
        """
        if self.transpose :
            data = dataclass.transpose(im)
        else :
            data = im

        if sn:
            data = data.data/data.uncertainty.array
    
        # find initial set of edges from center of image
        if cent == None :
            cent = int(data.shape[1] / 2.)
        med = np.median(data[:,cent-skip//2:cent+skip//2],axis=1)
        deriv = gaussian_filter(med[1:] - med[0:-1],smooth)
        bottom_edges,tmp = findpeak(deriv,thresh)
        top_edges,tmp = findpeak(-deriv,thresh)
        if display != None :
            display.clear()
            display.tv(data)
            for peak in bottom_edges :
                display.ax.plot([cent,cent],[peak,peak],'bo')
            for peak in top_edges :
                display.ax.plot([cent,cent],[peak,peak],'ro')
            display.plotax2.cla()
            display.plotax2.plot(deriv)
            display.plotax2.text(0.1,0.9,'Derivative of spatial flux',
                transform=display.plotax2.transAxes)
    
        if len(bottom_edges) != len(top_edges) :
            print("didn't find matching number of bottom and top edges!")
            pdb.set_trace()

        self.rows = []
        for b,t in zip(bottom_edges,top_edges) :
            self.rows.append([b,t])
    
        all_bottom=[]
        all_top=[]
        allrows=[]
   
        # work from center to end
        bottom = copy.copy(bottom_edges) 
        top = copy.copy(top_edges) 
        for irow in np.arange(cent,data.shape[1]-skip,skip) :
            med = np.median(data[:,irow-skip//2:irow+skip//2],axis=1)
            bottom=fitpeak(med,bottom,smooth=smooth,width=5)
            all_bottom.append(bottom)
            top=fitpeak(med,top,smooth=smooth,width=5,desc=True)
            all_top.append(top)
            allrows.append(irow)
    
        # work from center to beginning
        bottom = copy.copy(bottom_edges) 
        top = copy.copy(top_edges) 
        for irow in np.arange(cent-skip,skip,-skip) :
            med = np.median(data[:,irow-skip//2:irow+skip//2],axis=1)
            bottom=fitpeak(med,bottom,smooth=smooth,width=5)
            all_bottom.append(bottom)
            top=fitpeak(med,top,smooth=smooth,width=5,desc=True)
            all_top.append(top)
            allrows.append(irow)
    
        # fit the edges
        allrows = np.array(allrows)
        all_bottom = np.array(all_bottom)
        all_top = np.array(all_top)
        all_bottom_fit = []
        all_top_fit = []
        self.model = []
        for ipeak in range(len(bottom_edges)) :
            poly = Polynomial.fit(allrows,all_bottom[:,ipeak],deg=degree)
            if display != None :
                xx = np.arange(data.shape[1])
                display.ax.plot(xx,poly(xx),color='b')
            all_bottom_fit.append(poly)
            # just uses bottom fit for saved model
            self.model.append(poly)
    
        for ipeak in range(len(top_edges)) :
            poly = Polynomial.fit(allrows,all_top[:,ipeak],deg=degree)
            if display != None :
                xx = np.arange(data.shape[1])
                display.ax.plot(xx,poly(xx),color='r')
            all_top_fit.append(poly)
    
        return all_bottom_fit,all_top_fit

    
class FluxCal() :
    """ Class for flux calibration

    Parameters
    ----------
    degree : int, default=3
             Order of polynomial for response curve, -1 for median/mean
    median : bool, default=False
             if degree<0, use median response curve rather than mean
    extinct  : Table/str, default='flux/apo_extinct.dat'
             mean extinction curve to use, can be input as astropy Table with
             columns ['wave','mag'], or a filename from which these can be read

    Attributes
    ----------
    """

    def __init__(self,degree=3,median=False,extinct='flux/apo_extinct.dat') :
        self.nstars = 0
        self.waves = []
        self.weights = []
        self.obs = []
        self.obscorr = []
        self.true = []
        self.name = []
        self.degree = degree
        self.median = False
        self.mean = True
        if median : 
            self.mean = False
            self.median = True
        self.response_curve = None
        if type(extinct) is astropy.table.table.Table :
            self.meanextinct = extinct
        elif str(extinct)[0] == '.' or str(extinct)[0] == '/' :
            self.meanextinct=Table.read(extinct,format='ascii')
        elif extinct is not None :
            self.meanextinct=Table.read(files(pyvista.data).joinpath(extinct),
                   names=['wave','mag'],format='ascii')
        else :
            raise ValueError('must specify either file= or stdflux=')

    def write(self,file,append=False) :
        """ Save object to file  UNTESTED, and NO READ yet!

            Parameters 
            ----------
            file : str, name of output file to write to, FITS format
            append : bool, append to existing file (untested)
        """
        tab=Table()
        for tag in ['nstars','waves','weights','obs','obscorr',
                'true','name','degree','median','mean','coeffs',
                'response_curve'] :
            tab[tag] = [getattr(self,tag)]
        if append :
            tab.write(file,append=True)
        else :
            tab.write(file,overwrite=True)
        return tab

    def extinct(self,hd,wave) :
        """ Correct input image for atmospheric extinction

            Parameters 
            ----------
            hd : Data object with data
                 Input image
            wave : float, array-like
                 Wavelengths for input image (separate from attribute to allow slicing in hd)
            file : str, default='flux/apo_extinct.dat'
                 Two column file (['wave','mag']) with extinction curve
        """
        x = skycalc.airmass(hd.header)

        ext = np.interp(wave,self.meanextinct['wave'],self.meanextinct['mag'])
        corr = 10**(-0.4*ext*x)
        out=copy.deepcopy(hd)
        out.data /=corr
        out.uncertainty.array /=corr
        return out

    def addstar(self,hd,wave,pixelmask=None,file=None,stdflux=None,
                exptime='EXPTIME',extinct=True,badval=None) :
        """ Derive flux calibration vector from standard star

            Parameters 
            ----------
            hd : Data object with standard star spectrum
                 Input image
            file : str, optional
                 File with calibrated fluxes, with columns 
                 ['wave','flux','bin'], must be readable by astropy.io.ascii
                 with format='ascii'
            stdflux : astropy Table, optional
                 Table with calibrated fluxes in columns 'wave','flux','bin'
            extinct : bool, default=True
                 Use mean extinction curve to correct for atmospheric extinction
            exptime : str, default='EXPTIME'
                 Card name to use for exptime normalization, set to None for no norm
        """
        # any bad pixels?
        if pixelmask is not None :
            bd = np.bitwise_or.reduce(pixelmask.flatten())
            pixelmask=bitmask.PixelBitMask()
            if badval == None : badval = pixelmask.badval()
            if bd & badval :
                print('Bad pixels found in spectrum, not adding')
                return

        if stdflux is not None :
            tab=stdflux
        elif str(file)[0] == '.' or str(file)[0] == '/' :
            tab=Table.read(file,format='ascii')
        elif file is not None :
            tab=Table.read(files(pyvista.data).joinpath(file),
                   names=['wave','flux','mjy','bin'],format='ascii')
        else :
            raise ValueError('must specify either file= or stdflux=')

        if extinct : extcorr = self.extinct(hd,wave)
        else : extcorr=copy.deepcopy(hd)
        if exptime is not None : 
            extcorr.data /= hd.header[exptime]
            extcorr.uncertainty.array /= hd.header[exptime]
        obs=[]
        obscorr=[]
        w=[]
        true=[]
        weights=[]
        for line in tab :
            w1=line['wave']-line['bin']/2.
            w2=line['wave']+line['bin']/2.
            if w1 > wave.min() and w2 < wave.max() :
                j=np.where((wave >= w1) & (wave <= w2) )[0]
                obs.append(np.mean(hd.data[j]))
                obscorr.append(np.mean(extcorr.data[j]))
                w.append(line['wave'])
                true.append(line['flux'])
                weights.append(1.)
        w=np.array(w)
        obs=np.array(obs)
        obscorr=np.array(obscorr)
        true=np.array(true)
        weights=np.array(weights)

        self.waves.append(w)
        self.obs.append(obs)
        self.obscorr.append(obscorr)
        # mask areas around significant lines
        bdlines = [[7570,7730], [6850,7000], [6520, 6600], [4820,4900], [4300,4380]]
        for line in bdlines :
            bd=np.where((w>=line[0]) & (w<=line[1]) )[0]
            weights[bd] = 0.
        bd=np.where(~np.isfinite(obscorr)|(obscorr<=0.))[0]
        weights[bd] = 0.
        self.weights.append(weights)
        self.true.append(true)
        self.name.append('{:s} {:f}'.format(
            hd.header['FILE'],skycalc.airmass(hd.header)))
        self.nstars += 1

    def response(self,degree=None,inter=False,plot=True,legend=True,hard=None,medfilt=None) :
        """ Create response curve from loaded standard star spectra and fluxes

            Parameters 
            ----------
            degree : integer, default=self.degree
                 polynomial degree
            medfilt : integer, default=None
                 width of median filter to apply to mean/median response curve
            plot : bool, default=False
                 set to True to see plot 
            legend : bool, default=True
                 label stars on plot with a legend
            hard : str, default=None
                 file name for hardcopy plot 
        """

        if self.nstars < 1 :
            raise ValueError('you must add at least one star with addstar')
        if degree is not None :
            self.degree = degree

        des=[]
        rhs=[]
        if plot : 
            fig,ax=plots.multi(1,3,hspace=0.001,sharex=True)
        for istar,(wav,obs,true,weight,name) in enumerate(
               zip(self.waves,self.obscorr,self.true,self.weights,self.name)) :
            gd=np.where(weight > 0.)[0]
            if self.degree >= 0 :
                vander=np.vander(wav,self.degree+1)    
                design=np.zeros([len(wav),self.degree+self.nstars-1])
                design[:,0:self.degree]=(vander[:,0:self.degree]*
                      np.repeat(weight,self.degree).reshape(len(wav),self.degree))
                if istar>0 : design[:,self.degree+istar-1] = 1.
                des.append(design)

            tmp=-2.5*np.log10(obs/true)*weight
            bd=np.where(~np.isfinite(tmp))[0]
            tmp[bd] = 0.
            rhs.append(tmp)

            if plot :
                line,=ax[0].plot(wav,-2.5*np.log10(obs/true),lw=1)
                ax[0].scatter(wav[gd],-2.5*np.log10(obs[gd]/true[gd]),
                        marker='o',color=line.get_color(),
                        s=2, label='{:s}'.format(name))
                ax[1].scatter(wav,-2.5*np.log10(obs),
                        s=2,color=line.get_color(),
                        label='{:s}'.format(name))
                ax[2].scatter(wav,-2.5*np.log10(true),
                        s=2,color='k',
                        label='{:s}'.format(name))
                ax[2].set_xlabel('Wavelength')
                ax[0].set_ylabel('-2.5 log(obs/true )')
                ax[1].set_ylabel('-2.5 log(obs)')
                ax[2].set_ylabel('-2.5 log(true)')
        if plot :
            for i in range(3) : 
                yr=ax[i].get_ylim()
                ax[i].set_ylim(yr[0]+5, yr[0])
        if self.degree >= 0 :
            design=np.vstack(des)
            rhs=np.hstack(rhs)
            out=np.linalg.solve(np.dot(design.T,design),np.dot(design.T,rhs))
            self.coeffs = np.append(out[0:self.degree],np.mean(out[self.degree:]))
            self.response_curve=np.polyval(self.coeffs,wav)
        else :
            for wav in self.waves :
                if not np.array_equal(wav,self.waves[0]) : 
                    pdb.set_trace()
                    raise ValueError('cannot median response curves if not all on same wavelengths')
            allobs = np.array(self.obscorr)
            alltrue = np.array(self.true)
            allwav = np.array(self.waves)
            allweights = np.array(self.weights)
            if self.mean :
                self.response_curve = np.mean(-2.5*np.log10(allobs/alltrue),axis=0)
            else :
                self.response_curve = np.nanmedian(-2.5*np.log10(allobs/alltrue),axis=0)
            if medfilt is not None :
                self.response_curve = median_filter(self.response_curve,size=medfilt)
        if plot :
            for istar,(wav,obs,true,weight,name) in enumerate(
                zip(self.waves,self.obscorr,self.true,self.weights,self.name)) :
                ax[2].plot(wav,-2.5*np.log10(obs/10.**(-0.4*self.response_curve)))
            ax[0].plot(wav,self.response_curve,lw=5,color='k',label='response curve')
            if legend : ax[0].legend(fontsize='xx-small')
            plt.draw()

        if plot and hard is not None : 
            print('saving: ', hard)
            fig.savefig(hard)

    def correct(self,hd,waves,extinct=True,exptime='EXPTIME') :
        """ Apply flux correction to input spectrum

            Parameters 
            ----------
            hd : Data object with spectrum to be corrected
                 Input image
            waves : array-like
                 Wavelength array for hd (separate from hd to allow slicing of hd)
            extinct : bool, default=True
                 Apply atmospheric extinction correction
            exptime : str, default='EXPTIME'
                 Card name to use for exptime normalization, set to None for no norm
        """
        if extinct : 
            extcorr = self.extinct(hd,waves)
            hd.data = extcorr.data
            hd.uncertainty.array = extcorr.uncertainty.array
        if exptime is not None: 
            hd.data /= hd.header[exptime]
            hd.uncertainty.array /= hd.header[exptime]
        if self.degree >= 0 :
            for irow,row in enumerate(hd.data) :
                corr = 10.**(-0.4*np.polyval(self.coeffs,waves))
                hd.data[irow] /= corr
                hd.uncertainty.array[irow] /= corr
        else :
            spline = scipy.interpolate.CubicSpline(self.waves[0],self.response_curve)
            for irow,row in enumerate(hd.data) :
                corr = 10.**(-0.4*spline(waves[irow]))
                hd.data[irow] /= corr
                hd.uncertainty.array[irow] /= corr
        hd.add_response(corr)

    def refraction(self,h=2000,temp=10,rh=0.25) :
        p0=101325
        M = 0.02896968 
        g = 9.80665
        T0 = 288.16
        R0 = 8.314462618
        pressure = p0 *np.exp(-g*h*M/T0/R0)*10
        ref=erfa.refco(pressure,temp,rh,wav)[0]*206265

def gfit(data,x0,rad=10,sig=3,back=False) :
    """ Fit 1D gaussian
    """ 
    xx = np.arange(x0-rad,x0+rad+1)
    yy = data[x0-rad:x0+rad+1]
    peak=yy.argmax()+x0-rad
    xx = np.arange(peak-rad,peak+rad+1)
    yy = data[peak-rad:peak+rad+1]
    if back == False : 
        p0 = [data[peak],peak,sig]
        bounds = ((yy.min(),peak-rad,0.3),(yy.max(),peak+rad,rad))
    else : 
        p0 = [data[peak],peak,sig,yy.min()]
        bounds = ((yy.min(),peak-rad,0.3,0),(yy.max()+1.e-3,peak+rad,rad,yy.max()))
    
    coeff, var_matrix = curve_fit(gauss, xx, yy, p0=p0,bounds=bounds)
    fit = gauss(xx,*coeff)
    return coeff

def qgauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))/np.sqrt(2*np.pi)/sigma

def gauss(x, *p):
    """ Gaussian function
    """
    if len(p) == 3 : 
        A, mu, sigma = p
        back = 0.
    elif len(p) == 4 : 
        A, mu, sigma, back = p
    elif len(p) == 5 : 
        A, mu, sigma, back0, backslope = p
        back = back0 + x*backslope
    elif len(p) == 7 : 
        A, mu, sigma, B, Bmu, Bsigma, back = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))+B*np.exp(-(x-Bmu)**2/2.*Bsigma**2)+back

    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+back

def findpeak(x,thresh,diff=10000,bundle=0,verbose=False,sort=False) :
    """ Find peaks in vector x above input threshold
        attempts to associate an index with each depending on spacing

        Parameters
        ----------
        x : float, array-like
            input vector to find peaks in
        thresh : float
            threshold for peak finding 
        diff : int
            maximum difference in pixels between traces before incrementing fiber index
        bundle : int
            number of fibers after which to allow max distance 
            to be exceeded without incrementing
        sort : bool, optional, default=False
            return peak locations and indices sorted with brightest first
    """
    j=[]
    fiber=[]
    peak=[]
    f=-1
    for i in range(len(x)) :
        if i>0 and i < len(x)-1 and x[i]>x[i-1] and x[i]>x[i+1] and x[i]>thresh :
            j.append(i)
            peak.append(x[i])
            if verbose and len(j) > 1 :
                print(j[-1],j[-2],j[-1]-j[-2],f+1)

            if len(j) > 1 : sep=j[-1]-j[-2]
            if len(j)>1 and bundle> 0 and (f+1)%bundle != 0 and (f+1)%bundle != bundle-1 : 
                #print(j[-1],j[-2],j[-1]-j[-2],f)
                f=f+sep//diff  + 1
            elif len(j)>1 and bundle > 0:
                sep = (sep-13) if sep-13 > 0 else 0
                f=f+sep//diff  + 1
            else :
                f=f+1
            fiber.append(f)

    if sort :
      # return brightest peak first
      isort = np.argsort(peak)[::-1]
      fiber = list(np.array(fiber)[isort])
      j = list(np.array(j)[isort])

          
    return j,fiber

def fitpeak(data,peaks,smooth=3,width=3,desc=False) :
    """ Find slit edges given input set of locations

    Parameters
    ----------
    data : 1D array 
           input data to find slit edges, usually a flat field
    smooth : integer, optional, default=3
           smoothing radius for derivative calculation
    width : integer, optional, default=3
           window around peak of derivative to do polynomial fit for edge
    desc : book, optional, default=False
           if True, find descending peaks
    """
    deriv = gaussian_filter(data[1:] - data[0:-1],smooth)
    if desc : deriv *= -1
    newpeaks=[]
    for peak in peaks :
        ipeak = int(peak)
        p0 = [deriv[ipeak-width:ipeak+width].max(),
              deriv[ipeak-width:ipeak+width].argmax()+ipeak-width,1.,0.]
        xx = np.arange(ipeak-width,ipeak+width+1)
        yy = deriv[ipeak-width:ipeak+width+1]
        try :
            coeffs,var = curve_fit(gauss,xx,yy,p0=p0)
            newpeaks.append(coeffs[1])
        except :
            print('Failed fit: ',peak)
            newpeaks.append(peak)
    peaks = np.array(newpeaks)

    return peaks
 
def getinput(prompt,fig=None,index=False) :
    """  Get input from terminal or matplotlib figure
    """
    if fig == None : return '','',input(prompt)
    print(prompt)
    get = plots.mark(fig,index=index)
    return get

def model_col(pars) :
    """ Extract a single column, using boxcar extraction for multiple traces
    """

    ext,nrow,cols,models,sigmodels,index = pars
    new = np.zeros([nrow,len(cols)])
    for jcol,col in enumerate(cols) :

        row = np.arange(0,nrow)
        coldata= np.zeros([nrow])
        for i,(model,sigmodel) in enumerate(zip(models,sigmodels)) :
            cr=model(col)
            sig=sigmodel(col)
            rows=np.arange(int(cr-5*sig),int(cr+5*sig))
            coldata[rows] += qgauss(rows,ext[index[i],jcol],cr,sig)
        new[:,jcol] = coldata
    return new

def extract_col_fit(pars) :
    """ Extract a single column, using boxcar extraction for multiple traces
    """
    data,err,bitmask,cols,models,rad,pix0,back,sigmodels = pars
    spec = np.zeros([len(models),len(cols)])
    sig = np.zeros([len(models),len(cols)])
    mask = np.zeros([len(models),len(cols)],dtype=np.uintc)
    for jcol,col in enumerate(cols) :
        center=[]
        sigma=[]
        for i,(model,sigmodel) in enumerate(zip(models,sigmodels)) :
          center.append(model(col))
          sigma.append(sigmodel(col))
        center=np.array(center)
        sigma=np.array(sigma)
        n=len(models)

        # get background to subtract
        bpix = np.array([])
        bvar = np.array([])
        for bk in back :
            bpix=np.append(bpix,data[bk[0]:bk[1],jcol])
            bvar=np.append(bvar,err[bk[0]:bk[1],jcol]**2)
        bck = np.median(bpix)
        print(col,bck)

        ab=np.zeros([3,n])
        b=np.zeros([n])
        for row in range(data.shape[0]) :
          nearest=np.argmin(np.abs(row-center))
          if nearest == 0 :
              for j in [0,1] :
                ab[j+1,nearest] += ( qgauss(row, 1.,center[nearest],sigma[nearest]) *
                                     qgauss(row, 1.,center[nearest+j],sigma[nearest+j]) )
          elif nearest == n-1 :
              for j in [-1,0] :
                ab[j+1,nearest] += ( qgauss(row, 1.,center[nearest],sigma[nearest]) *
                                     qgauss(row, 1.,center[nearest+j],sigma[nearest+j]) )
          else :
              for j in [-1,0,1] :
                ab[j+1,nearest] += ( qgauss(row, 1.,center[nearest],sigma[nearest]) *
                                     qgauss(row, 1.,center[nearest+j],sigma[nearest+j]) )
          b[nearest] += (data[row,jcol]-bck) * qgauss(row,1.,center[nearest],sigma[nearest])

        x=solve_banded((1,1),ab,b)
        spec[:,jcol] = x
    return spec,sig, mask

def extract_col_old(pars) :
    """ Extract a single column, using boxcar extraction for multiple traces
    """
    data,err,bitmask,cols,models,rad,pix0,back,sigmodels = pars
    spec = np.zeros([len(models),len(cols)])
    sig = np.zeros([len(models),len(cols)])
    mask = np.zeros([len(models),len(cols)],dtype=np.uintc)
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
                mask[i,j] = np.bitwise_or.reduce(bitmask[r1:r2+1,j])
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
            pixmask=bitmask.PixelBitMask()
            mask[i,j] = pixmask.badval('BAD_EXTRACTION')

    return spec,sig, mask


def extract_col(pars) :
    """ Extract a series of columns, using boxcar extraction for multiple traces
    """
    data,err,bitmask,cols,models,rad,pix0,back,sigmodels = pars
    spec = np.zeros([len(models),len(cols)])
    sig2 = np.zeros([len(models),len(cols)])
    mask = np.zeros([len(models),len(cols)],dtype=np.uintc)
    background_spec = np.zeros([len(models),len(cols)])
    background_spec_err = np.zeros([len(models),len(cols)])
    ny=data.shape[0]
    ncol=data.shape[1]
    y,x = np.mgrid[0:data.shape[0],0:data.shape[1]]
    pix=np.zeros(data.shape)

    for i,model in enumerate(models) :

        # center of trace
        ymid=model(cols)+pix0

        # calculate distance of each pixel from trace center
        ylo = int(np.min(np.floor(ymid-rad)))
        yhi = int(np.max(np.ceil(ymid+rad)))
        dist=y[ylo:yhi+1,:]-ymid

        # determine contribution of each pixel to boxcar
        contrib = np.zeros(dist.shape,float)
        # full pixel contribution
        iy,ix = np.where( (np.abs(dist)<rad-0.5) )
        contrib[iy,ix] = 1.
        # fractional pixel contribution
        iy,ix = np.where( (np.abs(dist)>rad-0.5) & (np.abs(dist)<rad+0.5) )
        contrib[iy,ix] = 1-(np.abs(dist[iy,ix])-(rad-0.5))
 
        # add the contributions
        spec[i,:] = np.sum( data[ylo:yhi+1,:]*contrib, axis=0)
        sig2[i,:] = np.sum(err[ylo:yhi+1,:]**2*contrib**2, axis=0)
        # for bitmask take bitwise_or of pixels that have full contribution
        mask[i,:] = np.bitwise_or.reduce(
                       bitmask[ylo:yhi+1,:]*contrib.astype(int),axis=0) 

        # background
        background = np.empty_like(data)
        background[:] = np.nan
        background_err = copy.copy(background)
        if len(back) > 0 :
            dist = y - ymid

            nback=0
            for bk in back :
                iy,ix = np.where( (dist>bk[0]) & (dist<bk[1]) )
                background[iy,ix] = data[iy,ix]
                background_err[iy,ix] = err[iy,ix]**2
                nback+=np.abs(bk[1]-bk[0])

            spec[i,:] -= np.nanmedian(background,axis=0)*2*rad
            sig2[i,:] += np.nansum(background_err,axis=0)/nback*(2*rad)
            background_spec[i,:] = np.nanmedian(background,axis=0)*2*rad
            # rough estimate of error in background
            background_spec_err[i,:] = np.sqrt(np.nanmedian(background_err,axis=0)/nback)*2*rad

    return spec, np.sqrt(sig2), mask, background_spec, background_spec_err

class Model() :
    def __init__(self,func=None,apers=None,coeff=None,xdegree=1,ydegree=0,xdomain=[-1,1],ydomain=[-1,1]) :
        self.func=func
        self.apers=np.array(apers)
        self.coeff=coeff
        self.xdegree=xdegree
        self.ydegree=ydegree
        self.xdomain=xdomain
        self.ydomain=ydomain

    def __call__(self,x,y) :
        return self.func(x,y,*self.coeff,xdegree=self.xdegree,ydegree=self.ydegree,
                         xdomain=self.xdomain,ydomain=self.ydomain,apers=self.apers)

    def evaluate(self,x,*coeff) :
        return self.func(x[0],x[1],*coeff,xdegree=self.xdegree,ydegree=self.ydegree,
                         xdomain=self.xdomain,ydomain=self.ydomain,apers=self.apers)

def chebyshev2D_shift(x,y,*coeff,xdegree=0,ydegree=0,xdomain=[-1,1],ydomain=[-1,1],apers=None):
    """ Evaluate chebyshev2d function
    """
    xshift = np.zeros(len(x))
    if apers is not None :
        xshift[:] = np.nan
        j=np.where(y==int(apers[0]))[0]
        xshift[j] = 0.
        for i,aper in enumerate(apers[1:]) :
            j=np.where(y==int(aper))
            xshift[j] = coeff[(xdegree+1)*(ydegree+1)+i]

    xcheb = (x+xshift-xdomain[0])/(xdomain[1]-xdomain[0])*2 - 1
    ycheb = (y-ydomain[0])/(ydomain[1]-ydomain[0])*2 - 1
    design = chebvander2d(xcheb,ycheb,[xdegree,ydegree])

    return np.dot(design,coeff[0:(xdegree+1)*(ydegree+1)])

