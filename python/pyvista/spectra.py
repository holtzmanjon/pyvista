import matplotlib.pyplot as plt
import pdb
import pickle
import copy
import scipy.signal
import scipy.interpolate
import numpy as np
from astropy.modeling import models, fitting
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.io import ascii
from astropy.convolution import convolve, Box1DKernel, Box2DKernel
import pyvista
from pyvista import image
from pyvista import tv

class WaveCal() :
    """ Class for wavelength solutions
    """
    def __init__ (self,type='chebyshev',degree=2,ydegree=2,pix0=0,orders=[1],spectrum=None) :
        """ Initialize the wavecal object

            type : type of solution ('poly' or 'chebyshev')
            degree : polynomial degree for wavelength
            ydegree : polynomial degree for  y dimension
            pix0 : reference pixel
            orders : spectral order for each row
            spectrum : spectrum from which fit is derived
        """
        self.type = type
        self.degree = degree
        self.ydegree = ydegree
        self.pix0 = pix0
        self.orders = orders
        self.spectrum = spectrum
        self.spectrum.data = np.atleast_2d(spectrum.data)
        self.spectrum.uncertainty.array = np.atleast_2d(spectrum.uncertainty.array)
        self.waves = None
        self.x = None
        self.y = None
        self.weights = None

    def wave(self,pixels=None,image=None) :
        """ Wavelength from pixel using wavelength solution model

            pix : input pixel positions [x] or [y,x]
            image : for input image size [nrows,ncols], return wavelengths at all pixels
            returns wavelength
        """
        if pixels is not None :
            out=np.zeros(len(pixels[0]))
            for i,pixel in enumerate(pixels[0]) :
                if self.type.find('2D') > 0 :
                    order=self.orders[pixels[1][i]]
                    out[i]=self.model(pixel-self.pix0,pixels[1][i])/order
                else :
                    out[i]=self.model(pixel-self.pix0)
            return out
        else :
            out=np.zeros(image)
            cols=np.arange(out.shape[-1])
            if out.ndim == 2 :
                for row in range(out.shape[0]) : 
                    rows=np.zeros(len(cols))+row
                    try : order = self.orders[row]
                    except : order=1
                    out[row,:] = self.model(cols-self.pix0,rows)/order
            else :
                out= self.model(cols-self.pix0)
            return out

    def fit(self,plot=True) :
        """ do a wavelength fit 
        """

        # set up fitter and model
        twod=False
        fitter=fitting.LinearLSQFitter()
        if self.type == 'poly' :
            mod=models.Polynomial1D(degree=self.degree)
        elif self.type == 'chebyshev' :
            mod=models.Chebyshev1D(degree=self.degree)
        elif self.type == 'chebyshev2D' :
            twod=True
            sz=self.spectrum.shape
            mod=models.Chebyshev2D(x_degree=self.degree,y_degree=self.ydegree,
                                   x_domain=[0,sz[1]],y_domain=[0,sz[0]])
        else :
            raise ValueError('unknown fitting type: '+self.type)
            return

        if twod :
            self.model=fitter(mod,self.pix-self.pix0,self.y,self.waves*self.waves_order,weights=self.weights)
            diff=self.waves-self.wave(pixels=[self.pix,self.y])
            print('rms: {:8.3f}'.format(diff.std()))
            if plot : 
                fig,ax=plt.subplots(1,1)
                scat=ax.scatter(self.waves,diff,marker='o',c=self.y,s=2)
                ax.text(0.1,0.9,'rms: {:8.3f}'.format(diff.std()),transform=ax.transAxes)
                cb=plt.colorbar(scat,ax=ax,orientation='vertical')
                cb.ax.set_ylabel('Row')
                plt.draw()
                plt.show()

        else :
            self.model=fitter(mod,self.pix-self.pix0,self.waves,weights=self.weights)
            diff=self.waves-self.wave(pixels=[self.pix])
            print('rms: {:8.3f} Angstroms'.format(diff.std()))
            if plot :
                self.ax[1].plot(self.pix,diff,'ro')
                # iterate allowing for interactive removal of points
                done = False
                ymax = self.ax[0].get_ylim()[1]
                while not done :
                    gd=np.where(self.weights>0.)[0]
                    bd=np.where(self.weights<=0.)[0]
                    self.model=fitter(mod,self.pix[gd]-self.pix0,self.waves[gd],weights=self.weights[gd])
                    diff=self.waves-self.wave(pixels=[self.pix])
                    print('rms: {:8.3f} Anstroms'.format(diff[gd].std()))

                    # plot
                    self.ax[1].cla()
                    self.ax[1].plot(self.pix[gd],diff[gd],'go')
                    self.ax[1].text(0.1,0.9,'rms: {:8.3f} Angstroms'.format(diff[gd].std()),transform=self.ax[1].transAxes)
                    self.ax[1].set_xlabel('Pixel')
                    self.ax[1].set_ylabel('obs wave - fit wave')
                    if len(bd) > 0 : self.ax[1].plot(self.pix[bd],diff[bd],'ro')
                    self.ax[1].set_ylim(diff[gd].min()-1,diff[gd].max()+1)
                    for i in range(len(self.pix)) :
                        self.ax[1].text(self.pix[i],diff[i],'{:2d}'.format(i),va='top',ha='center')
                        if self.weights[i] > 0 :
                            self.ax[0].plot([self.pix[i],self.pix[i]],[0,ymax],'g')
                        else :
                            self.ax[0].plot([self.pix[i],self.pix[i]],[0,ymax],'r')
                    plt.draw()
                    plt.show()

                    # get input from user on lines to remove
                    for i in range(len(self.pix)) :
                        print('{:3d}{:8.2f}{:8.2f}{:8.2f}{:8.2f}'.format(
                               i, self.pix[i], self.waves[i], diff[i], self.weights[i]))
                    i = input('enter ID of line to remove (-n for all lines<n, +n for all lines>n, return to continue): ')
                    if i is '' :
                        done = True
                    elif '+' in i :
                        self.weights[int(i)+1:] = 0.
                    elif '-' in i :
                        self.weights[0:abs(int(i))] = 0.
                    elif int(i) >= 0 :
                        self.weights[int(i)] = 0.
                    else :
                        print('invalid input')

    def set_spectrum(self,spectrum) :
        """ Set spectrum used to derive fit
        """
        self.spectrum = np.atleast_2d(spectrum)

    def get_spectrum(self) :
        """ Set spectrum used to derive fit
        """
        return self.spectrum 

    def identify(self,file=None,wav=None,wcal0=None,wref=None,disp=None,plot=True,rad=5,thresh=10,
                 xmin=None, xmax=None, lags=range(-300,300)) :
        """ Given some estimate of wavelength solution and file with lines,
            identify peaks and centroid
        """

        sz=self.spectrum.shape
        if xmin is None : xmin=0
        if xmax is None : xmax=sz[-1]
        nrow=sz[0]

        # get initial reference wavelengths if not given
        if wav is None :
            pix=np.arange(sz[-1])
            if wcal0 is not None :
                # cross correlate with reference image to get pixel shift
                print('cross correlating with reference spectrum using lags: ', lags)
                shift = image.xcorr(wcal0.spectrum,self.spectrum.data,lags)
                wnew=copy.copy(wcal0)
                if shift.ndim == 1 :
                    print('Derived pixel shift from input wcal0: ',shift.argmax()+lags[0])
                    # single shift for all pixels
                    wnew.pix0 = wcal0.pix0+shift.argmax()+lags[0]
                    wav=np.atleast_2d(wnew.wave(image=np.array(sz)))
                else :
                    # different shift for each row
                    wav=np.zeros(sz)
                    cols = np.arange(sz[-1])
                    for row in range(wav.shape[0]) : 
                        print('Derived pixel shift from input wcal0 for row: {:d} {:d}'.format
                               (row,shift[row,:].argmax()+lags[0]),end='\r')
                        rows=np.zeros(len(cols))+row
                        try : order = self.orders[row]
                        except : order=1
                        wnew.pix0 = wcal0.pix0+shift[row,:].argmax()+lags[0]
                        wav[row,:] = wnew.model(cols-wnew.pix0)/order
                    print("")
            else :
                # get dispersion guess from header cards if not given in disp
                if disp is None: disp=hd.header['DISPDW']
                if wref is not None :
                    w0=wref[0]
                    pix0=wref[1]
                else:
                    w0=hd.header['DISPWC']
                    pix0=sz[1]/2 
                wav=np.atleast_2d(w0+(pix-pix0)*disp)

        # open file with wavelengths and read
        if file is not None :
            f=open(file,'r')
            lines=[]
            for line in f :
                if line[0] != '#' :
                    w=float(line.split()[0])
                    # if we have microns, convert to Angstroms
                    if w<10 : w*=10000
                    if w > wav.min() and w < wav.max() : lines.append(w)
            lines=np.array(lines)
            f.close()
        elif wcal0 is not None :
            lines = wcal0.waves
            weights = wcal0.weights
            gd = np.where(weights >0)[0]
            lines = lines[gd]
        else :
            raise ValueError('Need to specify previous solution or lamps file')
            return

        # get centroid around expected lines
        x=[]
        y=[]
        waves=[]
        waves_order=[]
        weight=[]
        if plot is not None : 
            if isinstance(plot,pyvista.tv.TV) :
                plot.ax.cla()
                plot.ax.axis('off')
                plot.tv(self.spectrum.data)
            else :
                fig,ax = plt.subplots(2,1,sharex=True,figsize=(14,7))
                self.ax = ax
                fig.subplots_adjust(hspace=1.1)

        for row in range(nrow) :
            print('identifying lines in row: ', row,end='\r')
            if plot is not None :
                if type(plot) is bool :
                    ax[0].cla()
                    ax[0].plot(self.spectrum.data[row,:])
                    ax[0].set_yscale('log')
                    ax[0].set_ylim(1.,ax[0].get_ylim()[1])
                    ax[0].text(0.1,0.9,'row: {:d}'.format(row),transform=ax[0].transAxes)
                    ax[0].set_xlabel('Pixel')
                    ax[0].set_ylabel('Intensity')
            for line in lines :
                peak=abs(line-wav[row,:]).argmin()
                if isinstance(plot,pyvista.tv.TV) :
                    if (peak > xmin+rad) and (peak < xmax-rad) : plot.ax.scatter(peak,row,marker='o',color='r',s=2)
                if ( (peak > xmin+rad) and (peak < xmax-rad) and 
                     ((self.spectrum.data[row,peak-rad:peak+rad]/self.spectrum.uncertainty.array[row,peak-rad:peak+rad]).max() > thresh) ) :
                    cent = (self.spectrum.data[row,peak-rad:peak+rad]*np.arange(peak-rad,peak+rad)).sum()/self.spectrum.data[row,peak-rad:peak+rad].sum()
                    if isinstance(plot,pyvista.tv.TV) :
                        plot.ax.scatter(cent,row,marker='o',color='m',s=2)
                    elif plot :
                        ax[0].text(cent,1.,'{:7.1f}'.format(line),rotation='vertical',va='top',ha='center')
                    x.append(cent)
                    y.append(row)
                    # we will fit for wavelength*order
                    waves.append(line)
                    waves_order.append(self.orders[row])
                    weight.append(1.)
        if plot is not None : 
            plt.draw()
            plt.show()
        self.pix=np.array(x)
        self.y=np.array(y)
        self.waves=np.array(waves)
        self.waves_order=np.array(waves_order)
        self.weights=np.array(weight)
        print('')

    def scomb(self,hd,wav,average=True) :
        """ Resample onto input wavelength grid
        """
        #output grid
        out=np.zeros(len(wav))
        if average: sig=np.zeros(len(wav))
        # raw wavelengths
        w=self.wave(image=np.array(np.atleast_2d(hd.data).shape))
        for i in range(np.atleast_2d(hd).shape[0]) :
            w1=np.abs(wav-w[i,0]).argmin()
            w2=np.abs(wav-w[i,-1]).argmin()
            sort=np.argsort(w[i,:])
            if average :
                out[w2:w1] += ( np.interp(wav[w2:w1],w[i,sort],np.atleast_2d(hd.data)[i,sort]) /
                                np.interp(wav[w2:w1],w[i,sort],np.atleast_2d(hd.uncertainty.array)[i,sort])**2 )
                sig[w2:w1] += 1./np.interp(wav[w2:w1],w[i,sort],np.atleast_2d(hd.uncertainty.array)[i,sort])**2 
            else :
                out[w2:w1] += np.interp(wav[w2:w1],w[i,sort],np.atleast_2d(hd.data)[i,sort])
        if average :
            out = out / sig
        return out

    def save(self,file) :
        """ Save object to file
        """
        try : delattr(self,'ax')
        except: pass
        f=open(file,'wb')
        pickle.dump(self,f)
        f.close()

class Trace() :
    """ Class for spectral traces
    """

    def __init__ (self,inst=None, type='poly',order=2,pix0=0,rad=5,spectrum=None,model=None,sc0=None,channel=None) :
        self.type = type
        self.order = order
        self.pix0 = pix0
        self.spectrum = spectrum
        self.rad = rad
        if model is not None : self.model=model
        if sc0 is not None : self.sc0=sc0
        if inst == 'TSPEC' :
            self.order = 3
            self.rows = [[135,235],[295,395],[435,535],[560,660],[735,830]]
            self.lags = range(-75,75) 
        elif inst == 'DIS' :
            if channel == 0 : self.rows=[[215,915]]
            elif channel == 1 : self.rows=[[100,800]]
            else : raise ValueError('need to specify channel')
            self.lags = range(-300,300) 
        elif inst == 'ARCES' :
            self.lags = range(-10,10) 

    def trace(self,hd,srows,sc0=None,plot=None,thresh=20) :
        """ Trace a spectrum from starting position
        """

        fitter=fitting.LinearLSQFitter()
        if self.type == 'poly' :
            mod=models.Polynomial1D(degree=self.order)
        else :
            raise ValueError('unknown fitting type: '+self.type)
            return

        nrow = hd.data.shape[0]
        ncol = hd.data.shape[1]
        if sc0 is None : self.sc0 = int(ncol/2)
        else : self.sc0 = sc0
        self.spectrum = hd[:,self.sc0]
        self.spectrum.data[self.spectrum.data<0] = 0.
        rows = np.arange(nrow)
        ypos = np.zeros(ncol)
        ysum = np.zeros(ncol)
        yvar = np.zeros(ncol)
        ymask = np.zeros(ncol,dtype=bool)

        # we want to handle multiple traces, so make sure srows is iterable
        if type(srows ) is int or type(srows) is float : srows=[srows]
        self.model=[]
        if plot is not None : 
            plot.clear()
            plot.tv(hd)

        for srow in srows :
            print('  Tracing row: {:d}'.format(int(srow)),end='\r')
            sr=copy.copy(srow)
            sr=int(round(sr))
            # march left from center
            for col in range(self.sc0,0,-1) :
                # centroid
                cr=sr-self.rad+hd.data[sr-self.rad:sr+self.rad+1,col].argmax()
                ysum[col] = np.sum(hd.data[cr-self.rad:cr+self.rad+1,col]) 
                ypos[col] = np.sum(rows[cr-self.rad:cr+self.rad+1]*hd.data[cr-self.rad:cr+self.rad+1,col]) / ysum[col]
                yvar[col] = np.sum(hd.uncertainty.array[cr-self.rad:cr+self.rad+1,col]**2) 
                ymask[col] = np.any(hd.mask[cr-self.rad:cr+self.rad+1,col]) 
                # use this position as starting center for next if above threshold S/N
                if (not ymask[col]) & np.isfinite(ysum[col]) & (ysum[col]/np.sqrt(yvar[col]) > thresh) : sr=int(round(ypos[col]))
            sr=copy.copy(srow)
            sr=int(round(sr))
            # march right from center
            for col in range(self.sc0+1,ncol,1) :
                # centroid
                cr=sr-self.rad+hd.data[sr-self.rad:sr+self.rad+1,col].argmax()
                ysum[col] = np.sum(hd.data[cr-self.rad:cr+self.rad+1,col]) 
                ypos[col] = np.sum(rows[cr-self.rad:cr+self.rad+1]*hd.data[cr-self.rad:cr+self.rad+1,col]) / ysum[col]
                yvar[col] = np.sum(hd.uncertainty.array[cr-self.rad:cr+self.rad+1,col]**2) 
                ymask[col] = np.any(hd.mask[cr-self.rad:cr+self.rad+1,col]) 
                # use this position as starting center for next if above threshold S/N
                if (not ymask[col]) & np.isfinite(ysum[col]) & (ysum[col]/np.sqrt(yvar[col]) > thresh) : sr=int(round(ypos[col]))

            cols=np.arange(ncol)
            gd = np.where((~ymask) & (ysum/np.sqrt(yvar)>thresh) )[0]
            model=(fitter(mod,cols[gd],ypos[gd]))
            self.model.append(model)

            if plot : 
                plot.ax.scatter(cols,ypos,marker='o',color='r',s=2) 
                plot.ax.scatter(cols[gd],ypos[gd],marker='o',color='g',s=2) 
                plot.ax.plot(cols,model(cols),color='m')

        print("")
        if plot : 
            input('  enter something to continue....')
     
    def find(self,hd,lags=None,plot=None) :
        """ Determine shift from existing trace to input frame
        """
        if type(hd) is not list : hd = [hd]
        if lags is None : lags = self.lags
         
        for im in hd : 
            shift = image.xcorr(self.spectrum,im.data[:,self.sc0],lags)
            peak=shift.argmax()
            fit=np.polyfit(range(-3,4),shift[peak-3:peak+4],2)
            fitpeak=peak+-fit[1]/(2*fit[0])
            print(peak,fitpeak)
            print('shift: ', shift.argmax()+lags[0])
            if plot is not None :
                plot.tv(im)
                plot.plotax1.cla()
                plot.plotax1.plot(self.spectrum.data/self.spectrum.data.max())
                plot.plotax1.plot(im.data[:,self.sc0]/im.data[:,self.sc0].max())
                plot.plotax2.cla()
                plot.plotax2.plot(lags,shift)
                input('  enter something to continue....')
        return fitpeak+lags[0]
 
    def extract(self,hd,rad=None,scat=False,plot=None) :
        """ Extract spectrum given trace(s)
        """
        if rad is None : rad=self.rad
        nrows=hd.data.shape[0]
        ncols=hd.data.shape[-1]
        spec = np.zeros([len(self.model),hd.data.shape[1]])
        sig = np.zeros([len(self.model),hd.data.shape[1]])

        if plot is not None:
            plot.clear()
            plot.tv(hd)
        for i,model in enumerate(self.model) :
            print('  extracting aperture {:d}'.format(i),end='\r')
            cr=model(np.arange(ncols))
            icr=np.round(cr).astype(int)
            rfrac=cr-icr+0.5   # add 0.5 because we rounded
            rlo=[]
            rhi=[]
            for col in range(ncols) :
                r1=icr[col]-self.rad
                r2=icr[col]+self.rad+1
                spec[i,col]=np.sum(hd.data[r1+1:r2-1,col])
                spec[i,col]+=hd.data[r1,col]*(1-rfrac[col])
                spec[i,col]+=hd.data[r2,col]*rfrac[col]
                sig[i,col]=np.sqrt(np.sum(hd.uncertainty.array[r1:r2,col]**2))
                if plot :
                    rlo.append(r1)
                    rhi.append(r2-1)
            if plot :
                if i%2 == 0 : color='b'
                else : color='m'
                plot.ax.plot(range(ncols),icr,color='g')
                plot.ax.plot(range(ncols),rlo,color=color)
                plot.ax.plot(range(ncols),rhi,color=color)
                plt.draw()
        print("")
        return CCDData(spec,uncertainty=StdDevUncertainty(sig),unit='adu')
  
    def extract2d(self,hd,rows=None) :
        """  Extract 2D spectrum given trace(s)
             Assumes all requests row uses same trace, just offset, not a 2D model for traces
        """
        nrows=hd.data.shape[0]
        ncols=hd.data.shape[-1]
        out=[]
        for model,row in zip(self.model,self.rows) :
            outrows=np.arange(row[0],row[1])
            noutrows=len(range(row[0],row[1]))
            spec=np.zeros([noutrows,ncols])
            sig=np.zeros([noutrows,ncols])
            cr=model(np.arange(ncols))
            cr-=cr[self.sc0]
            for col in range(ncols) :
                spec[:,col] = np.interp(outrows+cr[col],np.arange(nrows),hd.data[:,col])
                sig[:,col] = np.sqrt(np.interp(outrows+cr[col],np.arange(nrows),hd.uncertainty.array[:,col]**2))
            out.append(CCDData(spec,StdDevUncertainty(sig),unit='adu'))
        if len(out) == 1 : return out[0]
        else : return out

    def save(self,file) :
        """ Save object to file
        """
        try : delattr(self,'ax')
        except: pass
        f=open(file,'wb')
        pickle.dump(self,f)
        f.close()

def mash(hd,sp=None,bks=None) :
    """
    Mash image into spectra using requested window
    """
    if sp is None :
        sp=[0,hd.data.shape[0]]
    obj = hd.data[sp[0]:sp[1]].sum(axis=0)
    obj = hd.data[sp[0]:sp[1]].sum(axis=0)

    if bks is not None :
        back=[]
        for bk in bks :
           tmp=np.median(data[bk[0]:bk[1]],axis=0)
           back.append(tmp)
        obj-= np.mean(back,axis=0)

    return obj

def wavecal(hd,file=None,wref=None,disp=None,wid=[3],rad=5,snr=3,degree=2,wcal0=None,thresh=100,type='poly'):
    """
    Get wavelength solution for single 1D spectrum
    """

    # choose middle row +/ 5 rows
    sz=hd.data.shape
    spec=hd.data[int(sz[0]/2)-5:int(sz[0]/2)+5,:].sum(axis=0)
    spec=spec-scipy.signal.medfilt(spec,kernel_size=101)
    pix = np.arange(len(spec))

    fig,ax = plt.subplots(2,1,sharex=True,figsize=(14,6))
    ax[0].plot(spec)

    # get wavelength guess from input WaveCal if given, else use wref and dispersion, else header
    if wcal0 is not None :
        lags=range(-300,300)
        shift = image.xcorr(wcal0.spectrum,spec,lags)
        wnew=copy.deepcopy(wcal0)
        wnew.pix0 = wcal0.pix0+shift.argmax()+lags[0]
        print('  Derived pixel shift from input wcal0: ',shift.argmax()+lags[0])
        wav=wnew.wave(pix)
    else :
        # get dispersion guess from header cards if not given in disp
        if disp is None: disp=hd.header['DISPDW']
        if wref is not None :
            w0=wref[0]
            pix0=wref[1]
            wav=w0+(pix-pix0)*disp
        else:
            w0=hd.header['DISPWC']
            pix0=sz[1]/2 
            wav=w0+(pix-pix0)*disp
    ax[1].plot(wav,spec)

    # open file with wavelengths and read
    f=open(file,'r')
    lines=[]
    for line in f :
        if line[0] != '#' :
            w=float(line.split()[0])
            name=line[10:].strip()
            lpix=abs(w-wav).argmin()
            if lpix > 1 and lpix < sz[1]-1 :
                ax[0].text(lpix,0.,'{:7.1f}'.format(w),rotation='vertical',va='top',ha='center')
                lines.append(w)
    lines=np.array(lines)
    f.close()

    # get centroid around expected lines
    cents=[]
    for line in lines :
        peak=abs(line-wav).argmin()
        if (peak > rad) and (peak < sz[1]-rad) and (spec[peak-rad:peak+rad].max() > thresh) :
            print(peak,spec[peak-rad:peak+rad].max())
            cents.append((spec[peak-rad:peak+rad]*np.arange(peak-rad,peak+rad)).sum()/spec[peak-rad:peak+rad].sum())
    cents=np.array(cents)
    print('  cents:', cents)

    waves=[]
    weight=[]
    print('  Centroid  W0  Wave')
    for cent in cents :
        w=wav[int(cent)]
        ax[0].plot([cent,cent],[0,10000],'k')
        print('  {:8.2f}{:8.2f}{:8.2f}'.format(cent, w, lines[np.abs(w-lines).argmin()]))
        waves.append(lines[np.abs(w-lines).argmin()])
        weight.append(1.)
    waves=np.array(waves)
    weight=np.array(weight)

    # set up new WaveCal object
    pix0 = int(sz[1]/2)
    wcal = WaveCal(order=degree,type=type,spectrum=spec,pix0=pix0)

    # iterate allowing for interactive removal of points
    done = False
    ymax = ax[0].get_ylim()[1]
    while not done :
        gd=np.where(weight>0.)[0]
        bd=np.where(weight<=0.)[0]
        wcal.fit(cents[gd],waves[gd],weights=weight[gd])

        # plot
        ax[1].cla()
        ax[1].plot(cents[gd],wcal.wave(cents[gd])-waves[gd],'go')
        if len(bd) > 0 : ax[1].plot(cents[bd],wcal.wave(cents[bd])-waves[bd],'ro')
        diff=wcal.wave(cents[gd])-waves[gd]
        ax[1].set_ylim(diff.min()-1,diff.max()+1)
        for i in range(len(cents)) :
            ax[1].text(cents[i],wcal.wave(cents[i])-waves[i],'{:2d}'.format(i),va='top',ha='center')
            if weight[i] > 0 :
              ax[0].plot([cents[i],cents[i]],[0,ymax],'g')
            else :
              ax[0].plot([cents[i],cents[i]],[0,ymax],'r')
        plt.draw()

        # get input from user on lines to remove
        for i in range(len(cents)) :
            print('  {:3d}{:8.2f}{:8.2f}{:8.2f}{:8.2f}{:8.2f}'.format(
                   i, cents[i], wcal.wave(cents[i]), waves[i], waves[i]-wcal.wave(cents[i]),weight[i]))
        print('  rms: {:8.2f} Anstroms'.format(diff.std()))
        i = input('enter ID of line to remove (-n for all lines<n, +n for all lines>n, return to continue): ')
        if i is '' :
            done = True
        elif '+' in i :
            weight[int(i)+1:] = 0.
        elif '-' in i :
            weight[0:abs(int(i))] = 0.
        elif int(i) >= 0 :
            weight[int(i)] = 0.
        else :
            print('invalid input')

    plt.close()

    return wcal.wave(pix),wcal

def fluxcal(obs,wobs,file=None) :
    """
    flux calibration
    """

    fluxdata=ascii.read(file)
    stan=np.interp(wobs,fluxdata['col1'],fluxdata['col2'])
    return stan/obs
  

def trace(hd,apertures=None,pix0=1024) : 
    """ Get all traces
        apertures is a list of row numbers at pixel 1024
    """
    alltr=[]
    for i in range(len(apertures)) :
        tr=Trace()
        print('tracing aperture {:d}'.format(i),end='\r')
        sr=apertures[i]
        tr.trace(hd,pix0,sr)
        alltr.append(tr)

    return alltr

def extract(hd,apertures) :
    """ Do all extractions
    """
    spec = np.zeros([len(apertures),hd.data.shape[1]])
    for i,order in enumerate(apertures) :
        print('extracting aperture {:d}'.format(i),end='\r')
        spec[i] = order.extract(hd)

    return spec


from pyvista import imred
import readmultispec
import os
import pickle
def dis() :
    dred=imred.Reducer(inst='DIS',dir='UT191019/DIS/')
    dcomb=imred.Combiner(reducer=dred)
    arc=dcomb.sum([1,2,3])

    # define a flat trace model
    def model(x) : return(np.zeros(len(x))+500)
    traces=Trace(rad=10,model=[model],sc0=1024)

    spec=traces.extract(arc[0])
    spec-=scipy.signal.medfilt(spec[0,:],kernel_size=101)

    wcal=WaveCal(type='chebyshev',spectrum=spec)
    f=open(os.environ['PYVISTA_DIR']+'/data/dis/dis_blue_lowres.pkl','rb')
    wcal0=pickle.load(f)
    wcal.identify(wcal0=wcal0,file=os.environ['PYVISTA_DIR']+'/data/henear.dat',rad=3)
    wcal.fit()
    w=wcal.wave(image=np.array(spec.data.shape))

    spec2d=traces.extract2d(arc[0],rows=range(200,900))
    pdb.set_trace()

    star=dred.reduce(26)
    return spec,w,wcal

def arces() :
    ered=imred.Reducer(inst='ARCES',dir='UT191020/ARCES/')
    ecomb=imred.Combiner(reducer=ered)
    flat=ecomb.sum([11,15])
    thar=ecomb.sum([19,20])

    t=tv.TV()

    apertures=np.loadtxt('newap')[:,1]
    #traces=trace(flat,apertures/4)
    traces=Trace()
    traces.trace(flat,apertures/4,sc0=1024,thresh=1000,plot=t)
    ec=traces.extract(thar,scat=True)

    wcal=WaveCal(type='chebyshev2D',order0=54,spectrum=ec)
    wav=readmultispec.readmultispec('w131102.0004.ec.fits')['wavelen']
    new=ec*0.
    new[:,204:1855]=wav
    wcal.identify(wav=new,file=os.environ['PYVISTA_DIR']+'/data/thar_arces',xmin=201,xmax=1849,thresh=200,rad=3,plot=t)
    wcal.fit()

    wcal0=copy.deepcopy(wcal)
    wcal=WaveCal(type='chebyshev2D',order0=54,spectrum=ec)
    wcal.identify(wcal0=wcal0,file=os.environ['PYVISTA_DIR']+'/data/thar_arces',xmin=150,xmax=1900,thresh=200,rad=3,plot=t)
    wcal.fit()

    w=wcal.wave(image=np.array(ec.data.shape))

    hr7950=ered.reduce(1,superflat=flat)
    hr7950ec=traces.extract(hr7950,scat=True)

    return flat,thar,traces,ec,wcal,w

def tspec() :

    tspec=imred.Reducer(inst='TSPEC',dir='UT191026/TSPEC')   
    a=tspec.rd(21) 
    b=tspec.rd(22) 

    t=tv.TV()

    tapers=[155,316,454,591,761]    
    traces=Trace(order=3)
    traces.trace(a.subtract(b),tapers,sc0=350,thresh=1500,plot=t)

    rows=[[135,235],[295,395],[435,535],[735,830]]
    #spec2d=traces.extract2d(a.subtract(b),rows=range(200,900))
    return traces

def test():
    a=np.loadtxt('ecnewarc.ec')
    x=a[:,2]
    y=a[:,1]
    w=a[:,4]
    fitter=fitting.LinearLSQFitter()
    mod=models.Chebyshev2D(x_degree=3,y_degree=3,x_domain=[1,2000],y_domain=[55,160])
    fit=fitter(mod,x,y,w*y)
    plt.figure()
    plt.plot(w,w-fit(x,y)/y,'ro')

def testfit() :
    a=np.loadtxt('ecnewarc.ec')
    a[:,2]
    x=a[:,2]
    y=a[:,1]
    w=a[:,4]
    fitter=fitting.LinearLSQFitter()
    mod=models.Chebyshev2D(x_degree=3,y_degree=3,x_domain=[1,2000],y_domain=[55,160])
    fit=fitter(mod,x,y,w*y)
    plt.figure()
    plt.plot(w,w-fit(x,y)/y)
    plt.clf()
    plt.plot(w,w-fit(x,y)/y,'ro')


def echfit() :
    mod=models.custom_model(echelle_model)

def ripple(w,ord,amp=1,alpha=1,wc=5500) :
#def echelle_model(ind,amp,alpha,wc) :
#    w=ind[0,:]
#    ord=ind[1,:]
    x = ord * (1 - wc/w)
    out=amp*(np.sin(np.pi*alpha*x)/(np.pi*alpha*x))**2
    bd=np.where(np.pi*alpha*x == 0)[0]
    out[bd] = amp[bd]
    return out

def ripple2d(w,ord,amp0=1,amp1=0,amp2=0,alpha0=1,alpha1=0,alpha2=0,wc0=5500,wc1=0,wc2=0) :
#def echelle_model(ind,amp,alpha,wc) :
#    w=ind[0,:]
#    ord=ind[1,:]
    wc = wc0 + wc1*(ord-109) + wc2*(ord-109)**2
    amp = amp0 + amp1*(ord-109) + amp2*(ord-109)**2
    alpha = alpha0 + alpha1*(ord-109) + alpha2*(ord-109)**2
    #print(ord,amp,alpha,wc)
    x = ord * (1 - wc/w)
    out=amp*(np.sin(np.pi*alpha*x)/(np.pi*alpha*x))**2
    bd=np.where(np.pi*alpha*x == 0)[0]
    out[bd] = amp[bd]
    return out

def echelle_model_deriv(w,ord,amp=1,alpha=1,wc=5500) :
    x = ord * (1 - wc/w)
    f=amp*(np.sin(np.pi*alpha*x)/(np.pi*alpha*x))**2
    deriv=[]
    deriv.append(f/amp)
    deriv.append(2*amp*np.sin(np.pi*alpha*x)/(np.pi*alpha*x)**2*np.cos(np.pi*alpha*x)*np.pi*x
                 - 2*f/alpha )
    deriv.append(2*amp*np.sin(np.pi*alpha*x)/(np.pi*alpha*x)**2*np.cos(np.pi*alpha*x)*np.pi*alpha*-1*ord/w
                 + 2*f/x*ord/w)

    return np.array(deriv)

