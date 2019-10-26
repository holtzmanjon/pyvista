import matplotlib.pyplot as plt
import pdb
import copy
import scipy.signal
import numpy as np
from astropy.modeling import models, fitting
from astropy.io import ascii
from pyvista import image

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

class WaveCal() :
    """ Class for a wavelength solution
    """
    def __init__ (self,type='chebyshev',degree=2,ydegree=2,pix0=0,order0=1) :
        """ Initialize the wavecal object

            type : type of solution ('poly' or 'chebyshev')
            degree : polynomial degree for wavelength
            ydegree : polynomial degree for  y dimension
            pix0 : reference pixel
            order0 : spectral order for first y pixel (needed for cross-dispersed solutions
            spectrum : spectrum from which fit is derived
        """
        self.type = type
        self.degree = degree
        self.ydegree = ydegree
        self.pix0 = pix0
        self.order0 = order0
        self.spectrum = None
        self.waves = None
        self.x = None
        self.y = None
        self.weights = None

    def wave(self,pix) :
        """ Wavelength from pixel using wavelength solution model

            pix : input pixel positions [x] or [y,x]
            returns wavelength
        """
        if type(pix) is int : pix=np.array([pix])
        elif type(pix) is list : pix=np.array(pix)
        elif type(pix) is not np.ndarray : raise ValueError('Unknown input type for pix')
        sz=pix.shape()
        if len(sz) == 1:
            return self.model(pix-self.pix0)
        else :
            x=pix[1]
            y=pix[0]
            return self.model(x-self.pix0,y)/(y+self.order0)

    def fit(self,plot=True) :
        """ do a wavelength fit 
        """
        twod=False
        fitter=fitting.LinearLSQFitter()
        if self.type == 'poly' :
            mod=models.Polynomial1D(degree=self.degree)
        elif self.type == 'chebyshev' :
            mod=models.Chebyshev1D(degree=self.degree)
        elif self.type == 'chebyshev2D' :
            twod=True
            mod=models.Chebyshev2D(degree=self.degree,y_degree=self.ydegree,x_domain=[1,2000],y_domain=[54,160])
        else :
            raise ValueError('unknown fitting type: '+self.type)
            return

        if plot : plt.figure()
        if twod :
            self.model=fitter(mod,self.pix-self.pix0,self.y,self.waves,weights=self.weights)
            plt.plot(self.waves,self.waves-fit(self.x,self.y)/(wcal.y+self.order0),'ro')
        else :
            self.model=fitter(mod,self.pix-self.pix0,self.waves,weights=self.weights)
            plt.plot(self.waves,self.waves-fit(self.x),'ro')
        if plot : plt.draw()

    def set_spectrum(self,spectrum) :
        """ Set spectrum used to derive fit
        """
        self.spectrum = spectrum

    def get_spectrum(self) :
        """ Set spectrum used to derive fit
        """
        return self.spectrum 

    def identify(self,file=None,wav=None,wcal0=None,wref=None,disp=None,plot=True,rad=5,thresh=100,xmin=None, xmax=None) :
        """ Given some estimate of wavelength solution and file with lines,
            identify peaks and centroid
        """

        sz=self.spectrum.shape
        if xmin is None : xmin=0
        if xmax is None : xmax=sz[1]
        nrow=sz[0]
        # get initial reference wavelengths if not given
        if wav is None :
            if wcal0 is not None :
                lags=range(-300,300)
                shift = image.xcorr(wcal0.spectrum,spec,lags)
                print('Derived pixel shift from input wcal0: ',shift.argmax()+lags[0])
                wnew=copy.deepcopy(wcal0)
                wnew.pix0 = wcal0.pix0+shift.argmax()+lags[0]
                wav=wnew.wave(pix)
            else :
                # get dispersion guess from header cards if not given in disp
                if disp is None: disp=hd.header['DISPDW']
                if wref is not None :
                    w0=wref[0]
                    pix0=wref[1]
                else:
                    w0=hd.header['DISPWC']
                    pix0=sz[1]/2 
                wav=w0+(pix-pix0)*disp

        # open file with wavelengths and read
        f=open(file,'r')
        lines=[]
        for line in f :
            if line[0] != '#' :
                w=float(line.split()[0])
                if w > wav.min() and w < wav.max() :
                    lines.append(w)
        lines=np.array(lines)
        f.close()

        # get centroid around expected lines
        x=[]
        y=[]
        waves=[]
        weight=[]
        if plot : 
            fig,ax = plt.subplots(2,1,sharex=True,figsize=(14,7))
            fig.subplots_adjust(hspace=1.1)

        for row in range(nrow) :
            print('identifying row: ', row,end='\r')
            if plot :
                ax[0].cla()
                ax[0].plot(self.spectrum[row,:])
                ax[0].set_yscale('log')
                ax[0].set_ylim(1.,ax[0].get_ylim()[1])
                ax[0].text(0.1,0.9,'row: {:d}'.format(row),transform=ax[0].transAxes)
            for line in lines :
                peak=abs(line-wav[row,:]).argmin()
                if (peak > xmin+rad) and (peak < xmax-rad) and (self.spectrum[row,peak-rad:peak+rad].max() > thresh) :
                    if plot: ax[0].text(peak,1.,'{:7.1f}'.format(line),rotation='vertical',va='top',ha='center')
                    x.append((self.spectrum[row,peak-rad:peak+rad]*np.arange(peak-rad,peak+rad)).sum()/self.spectrum[row,peak-rad:peak+rad].sum())
                    y.append(row)
                    waves.append(line)
                    weight.append(1.)
            plt.draw()
        self.pix=np.array(x)
        self.y=np.array(y)
        self.waves=np.array(waves)
        self.weight=np.array(weight)


        

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
        print('Derived pixel shift from input wcal0: ',shift.argmax()+lags[0])
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
    print('cents:', cents)

    waves=[]
    weight=[]
    print('Centroid  W0  Wave')
    for cent in cents :
        w=wav[int(cent)]
        ax[0].plot([cent,cent],[0,10000],'k')
        print('{:8.2f}{:8.2f}{:8.2f}'.format(cent, w, lines[np.abs(w-lines).argmin()]))
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
        ax[1].plot(cents[bd],wcal.wave(cents[bd])-waves[bd],'ro')
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
            print('{:3d}{:8.2f}{:8.2f}{:8.2f}{:8.2f}{:8.2f}'.format(
                   i, cents[i], wcal.wave(cents[i]), waves[i], waves[i]-wcal.wave(cents[i]),weight[i]))
        print('rms: {:8.2f} Anstroms'.format(diff.std()))
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
  

class Trace() :
    def __init__ (self,type='poly',order=2,pix0=0,rad=5,spectrum=None,coeffs=None) :
        self.type = type
        self.order = order
        self.pix0 = pix0
        self.spectrum = spectrum
        self.rad = rad
        if coeffs is None :self.coeffs = np.zeros(order)
        else : self.coeffs = coeffs

    def trace(self,hd,sc0,sr0) :
        """ Trace a spectrum from starting position
        """

        nrow = hd.data.shape[0]
        ncol = hd.data.shape[1]
        rows = np.arange(nrow)
        ypos = np.zeros(ncol)
        sr=copy.copy(sr0)
        sr=int(round(sr))
        for col in range(sc0,0,-1) :
            # centroid
            cr=sr-self.rad+hd.data[sr-self.rad:sr+self.rad+1,col].argmax()
            ypos[col] = (np.sum(rows[cr-self.rad:cr+self.rad+1]*hd.data[cr-self.rad:cr+self.rad+1,col]) /
                         np.sum(hd.data[cr-self.rad:cr+self.rad+1,col]) )
            sr=int(round(ypos[col]))
        sr=copy.copy(sr0)
        sr=int(round(sr))
        for col in range(sc0+1,ncol,1) :
            # centroid
            cr=sr-self.rad+hd.data[sr-self.rad:sr+self.rad+1,col].argmax()
            ypos[col] = (np.sum(rows[cr-self.rad:cr+self.rad+1]*hd.data[cr-self.rad:cr+self.rad+1,col]) /
                         np.sum(hd.data[cr-self.rad:cr+self.rad+1,col]) )
            sr=int(round(ypos[col]))

        fitter=fitting.LinearLSQFitter()
        if self.type == 'poly' :
            mod=models.Polynomial1D(degree=self.order)

        cols=np.arange(ncol)
        gd = np.where((cols>100) & (cols<ncol-100) )[0]
        self.model=fitter(mod,np.arange(ncol)[gd],ypos[gd])

        return ypos,self.model(np.arange(ncol))
        

    def extract(self,hd) :
        """ Extract spectrum given trace
        """
        ncol = hd.data.shape[1]
        data=hd.data.astype(float)
        spec = []
        cr=np.round(self.model(np.arange(ncol))).astype(int)
        for col in range(ncol) :
            spec.append(np.sum(data[cr[col]-self.rad:cr[col]+self.rad+1,col]))
        return np.array(spec)
   
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
def arces() :
    eread=imred.Reader(inst='ARCES',dir='UT191020/ARCES/')
    ered=imred.Reducer(inst='ARCES')
    ecomb=imred.Combiner(reader=eread,reducer=ered)
    flat=ecomb.sum([11,15])
    thar=ecomb.sum([19,20])

    apertures=np.loadtxt('newap')[:,1]
    traces=trace(flat,apertures/4)
    ec=extract(thar,traces)

    wcal=WaveCal()
    wcal.set_spectrum(ec)
    wav=readmultispec.readmultispec('w131102.0004.ec.fits')['wavelen']
    new=ec*0.
    new[:,204:1855]=wav
    pdb.set_trace()
    wcal.identify(wav=new,file=os.environ['PYVISTA_DIR']+'/data/thar_arces',xmin=201,xmax=1849,thresh=200,rad=3)
    wcal.fit()
    pdb.set_trace()
    fitter=fitting.LinearLSQFitter()
    mod=models.Chebyshev2D(x_degree=3,y_degree=3,x_domain=[1,2000],y_domain=[54,160])
    fit=fitter(mod,wcal.x,wcal.y+54,wcal.waves*(wcal.y+54))
    plt.figure()
    plt.plot(wcal.waves,wcal.waves-fit(wcal.x,wcal.y+54)/(wcal.y+54),'ro')
    plt.draw()
    return flat,thar,traces,ec,wcal,fit

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

