import matplotlib.pyplot as plt
import pdb
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
    def __init__ (self,type='poly',order=1,coeffs=None,spectrum=None,pix0=0)  :
        self.type = type
        self.order = order
        self.pix0 = pix0
        self.spectrum = spectrum
        if coeffs is None :self.coeffs = np.zeros(order)
        else : self.coeffs = coeffs

    def wave(self,pix) :
        """ Wavelength from pixel
        """
        return self.model(pix-self.pix0)

    def fit(self,pix,wav,weights=None) :
        """ do a wavelength fit
        """
        fitter=fitting.LinearLSQFitter()
        if self.type == 'poly' :
            mod=models.Polynomial1D(degree=self.order)
        elif self.type == 'chebyshev' :
            mod=models.Chebyshev1D(degree=self.order)
        else :
            print('unknown fitting type!')
            pdb.set_trace()
        self.model=fitter(mod,pix-self.pix0,wav,weights=weights)

    def set_spectrum(self,spectrum) :
        """ Set spectrum used to derive fit
        """
        self.spectrum = spectrum

    def get_spectrum(self) :
        """ Set spectrum used to derive fit
        """
        return self.spectrum 



def wavecal(hd,file=None,wref=None,disp=None,wid=[3],rad=5,snr=3,degree=2,wcal=None,thresh=100,type='poly'):
    """
    Get wavelength solution for single 1D spectrum
    """

    # choose middle row +/ 5 rows
    sz=hd.data.shape
    spec=hd.data[int(sz[0]/2)-5:int(sz[0]/2)+5,:].sum(axis=0)
    spec=spec-scipy.signal.medfilt(spec,kernel_size=101)

    fig=plt.figure()
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(spec)


    # get wavelength guess from input WaveCal if given, else use wref and dispersion, else header
    pix = np.arange(len(spec))
    if wcal is not None :
        lags=range(-300,300)
        shift = image.xcorr(wcal.spectrum,spec,lags)
        wcal.pix0 = wcal.pix0+shift.argmax()+lags[0]
        wav=wcal.wave(pix)
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
            pix=abs(w-wav).argmin()
            print(pix, w, name)
            if pix > 0 and pix < sz[1] :
                ax[0].text(pix,0.,'{:7.1f}'.format(w),rotation='vertical',va='top',ha='center')
                lines.append(w)
    lines=np.array(lines)
    f.close()

    # find peaks
    #peaks=scipy.signal.find_peaks_cwt(spec,np.array(wid),min_snr=snr)
    # get centroid around peaks using window of width rad
    #for peak in peaks :

    tmp=spec-scipy.signal.medfilt(spec,kernel_size=101)
    # get centroid around expected lines
    cents=[]
    for line in lines :
        peak=abs(line-wav).argmin()
        if (peak > rad) and (peak < sz[1]-rad) and (tmp[peak-rad:peak+rad].max() > thresh) :
            print(peak,tmp[peak-rad:peak+rad].max())
            cents.append((tmp[peak-rad:peak+rad]*np.arange(peak-rad,peak+rad)).sum()/tmp[peak-rad:peak+rad].sum())
    cents=np.array(cents)
    print('cents:', cents)

    waves=[]
    weight=[]
    print('Centroid  W0  Wave')
    for cent in cents :
        #w=(cent-pix0)*disp+w0
        w=wav[int(cent)]
        ax[0].plot([cent,cent],[0,10000],'k')
        print('{:8.2f}{:8.2f}{:8.2f}'.format(cent, w, lines[np.abs(w-lines).argmin()]))
        waves.append(lines[np.abs(w-lines).argmin()])
        weight.append(1.)
    waves=np.array(waves)
    weight=np.array(weight)

    # do polynomial fit    
    done = False
    fit=fitting.LinearLSQFitter()
    mod=models.Polynomial1D(degree=degree)
    ymax = ax[0].get_ylim()[1]

    pix0 = int(sz[1]/2)
    wcal = WaveCal(order=degree,type=type,spectrum=spec,pix0=pix0)

    while not done :
        gd=np.where(weight>0.)[0]
        bd=np.where(weight<=0.)[0]
        wcal.fit(cents[gd],waves[gd],weights=weight[gd])

        #p=fit(mod,cents[gd]-pix0,waves[gd],weights=weight[gd])
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
        for i in range(len(cents)) :
            print('{:3d}{:8.2f}{:8.2f}{:8.2f}{:8.2f}{:8.2f}'.format(
                   i, cents[i], wcal.wave(cents[i]), waves[i], waves[i]-wcal.wave(cents[i]),weight[i]))
        i = input('enter ID of line to remove (-n for all lines<n, +n for all lines>n, return to continue): ')
        if i is '' :
            done = True
        elif '+' in i :
            weight[int(i):] = 0.
        elif '-' in i :
            weight[0:abs(int(i))] = 0.
        elif int(i) >= 0 :
            weight[int(i)] = 0.
        else :
            print('invalid input')

    print('rms: ', diff.std())

    return wcal

def fluxcal(obs,wobs,file=None) :
    """
    flux calibration
    """

    fluxdata=ascii.read(file)
    stan=np.interp(wobs,fluxdata['col1'],fluxdata['col2'])
    return stan/obs
    
