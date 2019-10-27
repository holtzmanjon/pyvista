import matplotlib.pyplot as plt
import pdb
import copy
import scipy.signal
import numpy as np
from astropy.modeling import models, fitting
from astropy.io import ascii
from pyvista import image

class WaveCal() :
    """ Class for wavelength solutions
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

    def wave(self,pixels=None,image=None) :
        """ Wavelength from pixel using wavelength solution model

            pix : input pixel positions [x] or [y,x]
            returns wavelength
        """
        if pixels is not None :
            out=np.zeros(len(pixels[0]))
            for i,pixel in enumerate(pixels[0]) :
                if self.type.find('2D') > 0 :
                    out[i]=self.model(pixel-self.pix0,pixels[1][i])/(pixels[1][i]+self.order0)
                    print(pixel,pixels[1][i],out[i])
                else :
                    out[i]=self.model(pixel-self.pix0)
            return out
        else :
            out=np.zeros(image)
            cols=np.arange(out.shape[-1])
            if self.type.find('2D') > 0 :
                for row in range(out.shape[0]) : 
                    rows=np.zeros(len(cols))+row
                    out[row,:] = self.model(cols-self.pix0,rows)/(row+self.order0)
            else :
                out= self.model(cols-self.pix0)
            return out

        if type(pix) is int : pix=np.array([pix])
        elif type(pix) is list : pix=np.array(pix)
        elif type(pix) is not np.ndarray : raise ValueError('Unknown input type for pix')
        if y is not None :
            if len(pix) != len(y) : raise ValueError('Number of x positions must equal number of y positions')
            out=[]
            for x,row in zip(pix,y) :
                out.append(self.model(x-self.pix0)/(y+self.order0))
            return np.array(out)
        else :
            out=np.array(len(self.model),len(pix))
            for y,model in enumerate(self.model) :
                out[i] = model(pix-self.pix0)/(y+self.order0)

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
            sz=self.spectrum.shape
            mod=models.Chebyshev2D(x_degree=self.degree,y_degree=self.ydegree,
                                   x_domain=[0,sz[1]],y_domain=[0+self.order0,sz[0]+self.order0])
        else :
            raise ValueError('unknown fitting type: '+self.type)
            return

        if twod :
            self.model=fitter(mod,self.pix-self.pix0,self.y,self.waves*(self.y+self.order0),weights=self.weights)
            diff=self.waves-self.wave(pixels=[self.pix,self.y])
            print('rms: {:8.2f}'.format(diff.std()))
            if plot : 
                plt.figure()
                plt.plot(self.waves,deff,'ro')
                plt.draw()
        else :
            self.model=fitter(mod,self.pix-self.pix0,self.waves,weights=self.weights)
            diff=self.waves-self.wave(pixels=[self.pix])
            print('rms: {:8.2f} Angstroms'.format(diff.std()))
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
                    print('rms: {:8.2f} Anstroms'.format(diff[gd].std()))

                    # plot
                    self.ax[1].cla()
                    self.ax[1].plot(self.pix[gd],diff[gd],'go')
                    self.ax[1].text(0.1,0.9,'rms: {:8.2f} Angstroms'.format(diff[gd].std()),transform=self.ax[1].transAxes)
                    if len(bd) > 0 : self.ax[1].plot(self.pix[bd],diff[bd],'ro')
                    self.ax[1].set_ylim(diff[gd].min()-1,diff[gd].max()+1)
                    for i in range(len(self.pix)) :
                        self.ax[1].text(self.pix[i],diff[i],'{:2d}'.format(i),va='top',ha='center')
                        if self.weights[i] > 0 :
                            self.ax[0].plot([self.pix[i],self.pix[i]],[0,ymax],'g')
                        else :
                            self.ax[0].plot([self.pix[i],self.pix[i]],[0,ymax],'r')
                    plt.draw()

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

        pdb.set_trace()

    def set_spectrum(self,spectrum) :
        """ Set spectrum used to derive fit
        """
        self.spectrum = np.atleast_2d(spectrum)

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
        if xmax is None : xmax=sz[-1]
        nrow=sz[0]

        # get initial reference wavelengths if not given
        if wav is None :
            pix=np.arange(sz[-1])
            if wcal0 is not None :
                lags=range(-300,300)
                shift = image.xcorr(wcal0.spectrum,self.spectrum[0,:],lags)
                print('Derived pixel shift from input wcal0: ',shift.argmax()+lags[0])
                wnew=copy.deepcopy(wcal0)
                wnew.pix0 = wcal0.pix0+shift.argmax()+lags[0]
                wav=np.atleast_2d(wnew.wave(image=sz))
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
            self.ax = ax
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
        self.weights=np.array(weight)

class Trace() :
    """ Class for spectral traces
    """

    def __init__ (self,type='poly',order=2,pix0=0,rad=5,spectrum=None,coeffs=None) :
        self.type = type
        self.order = order
        self.pix0 = pix0
        self.spectrum = spectrum
        self.rad = rad
        if coeffs is None :self.coeffs = np.zeros(order)
        else : self.coeffs = coeffs

    def trace(self,hd,srows,sc0=None,plot=None,thresh=500) :
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
        if sc0 is None : sc0 = int(ncol/2)
        rows = np.arange(nrow)
        ypos = np.zeros(ncol)
        ysum = np.zeros(ncol)

        # we want to handle multiple traces, so make sure srows is iterable
        if type(srows ) is int or type(srows) is float : srows=[srows]
        self.model=[]
        if plot : plot.tv(hd)

        for srow in srows :
            print('Tracing row: {:d}'.format(int(srow)),end='\r')
            sr=copy.copy(srow)
            sr=int(round(sr))
            # march left from center
            for col in range(sc0,0,-1) :
                # centroid
                cr=sr-self.rad+hd.data[sr-self.rad:sr+self.rad+1,col].argmax()
                ysum[col] = np.sum(hd.data[cr-self.rad:cr+self.rad+1,col]) 
                ypos[col] = np.sum(rows[cr-self.rad:cr+self.rad+1]*hd.data[cr-self.rad:cr+self.rad+1,col]) / ysum[col]
                sr=int(round(ypos[col]))
            sr=copy.copy(srow)
            sr=int(round(sr))
            # march right from center
            for col in range(sc0+1,ncol,1) :
                # centroid
                cr=sr-self.rad+hd.data[sr-self.rad:sr+self.rad+1,col].argmax()
                ysum[col] = np.sum(hd.data[cr-self.rad:cr+self.rad+1,col]) 
                ypos[col] = np.sum(rows[cr-self.rad:cr+self.rad+1]*hd.data[cr-self.rad:cr+self.rad+1,col]) / ysum[col]
                sr=int(round(ypos[col]))

            cols=np.arange(ncol)
            gd = np.where(ysum>thresh )[0]
            model=(fitter(mod,cols[gd],ypos[gd]))
            self.model.append(model)

            if plot : 
                plot.ax.scatter(cols,ypos,marker='o',color='r',s=2) 
                plot.ax.scatter(cols[gd],ypos[gd],marker='o',color='g',s=2) 
                plot.ax.plot(cols,model(cols),color='m')

        return 
        

    def extract(self,hd) :
        """ Extract spectrum given trace
        """
        ncols=hd.data.shape[-1]
        spec = np.zeros([len(self.model),hd.data.shape[1]])
        for i,model in enumerate(self.model) :
            print('extracting aperture {:d}'.format(i),end='\r')
            cr=np.round(model(np.arange(ncols))).astype(int)
            for col in range(ncols) :
                spec[i,col]=np.sum(hd.data[cr[col]-self.rad:cr[col]+self.rad+1,col])

        return spec

#        ncol = hd.data.shape[1]
#        data=hd.data.astype(float)
#        cr=np.round(self.model(np.arange(ncol))).astype(int)
#        spec = []
#        for col in range(ncol) :
#            spec.append(np.sum(data[cr[col]-self.rad:cr[col]+self.rad+1,col]))
#        return np.array(spec)
   

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
    dread=imred.Reader(inst='DIS',dir='UT191019/DIS/')
    dred=imred.Reducer(inst='DIS')
    dcomb=imred.Combiner(reader=dread,reducer=dred)
    arc=dcomb.sum([1,2,3])
    wcal=WaveCal(type='chebyshev')
    f=open(os.environ['PYVISTA_DIR']+'/data/dis/dis_blue_lowres.pkl','rb')
    wcal0=pickle.load(f)
    spec=np.sum(arc[0].data[490:510,:],axis=0)
    spec-=scipy.signal.medfilt(spec,kernel_size=101)
    wcal.set_spectrum(spec)
    wcal.identify(wcal0=wcal0,file=os.environ['PYVISTA_DIR']+'/data/henear.dat',rad=3)
    wcal.fit()
    w=wcal.wave(image=np.array(spec.data.shape))

    star=dred.reduce(26)
    return spec,w

def arces() :
    eread=imred.Reader(inst='ARCES',dir='UT191020/ARCES/')
    ered=imred.Reducer(inst='ARCES')
    ecomb=imred.Combiner(reader=eread,reducer=ered)
    flat=ecomb.sum([11,15])
    thar=ecomb.sum([19,20])

    apertures=np.loadtxt('newap')[:,1]
    #traces=trace(flat,apertures/4)
    traces=Trace()
    traces.trace(flat,apertures/4,sc=1024,thresh=1000)
    ec=traces.extract(thar)

    wcal=WaveCal(type='chebyshev2D',order0=54)
    wcal.set_spectrum(ec)
    wav=readmultispec.readmultispec('w131102.0004.ec.fits')['wavelen']
    new=ec*0.
    new[:,204:1855]=wav
    wcal.identify(wav=new,file=os.environ['PYVISTA_DIR']+'/data/thar_arces',xmin=201,xmax=1849,thresh=200,rad=3)
    wcal.fit()
    w=wcal.wave(image=np.array(ec.data.shape))

    return flat,thar,traces,ec,wcal,fit,w

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

