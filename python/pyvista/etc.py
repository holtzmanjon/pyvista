import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pdb
import astropy.constants as c
from astropy.modeling.models import BlackBody


class Object:
    """ Class representing an object
        Given a type (blackbody or input spectrum) and a magnitude,
        provides method for Fnu, Flam, photon flux
    """
    def __init__(self,type='bb',teff=10000,mag=0,refmag='V'):
        self.type = type
        self.teff = teff
        self.mag = mag
        self.refmag = refmag
        
    @u.quantity_input(wave=u.m)
    def sed(self,wave):
        """ Return sed in Fnu or Flambda, with units (depending on source of data)
        """
        if self.type == 'bb' :
            bb = BlackBody(temperature=self.teff*u.K)
            norm= 3.63e-9*u.erg/u.cm**2/u.s/u.AA / \
                  (bb(5500*u.AA)*u.sr*c.c/(5500*u.AA)**2).to(u.erg/u.cm**2/u.s/u.AA)
            return bb(wave)*norm*u.sr*10.**(-0.4*self.mag)
        else :
            raise ValueError('Input type {:s} not yet implemented'.format(type))
          
    @u.quantity_input(wave=u.m)
    def photflux(self,wave):
        """ Return SED in photon flux"""
        return (self.flam(wave)*wave/c.h/c.c).to(1/u.cm**2/u.s/u.AA)
    
    @u.quantity_input(wave=u.m)
    def flam(self,wave):
        """ Return SED in Flambda"""
        sed=self.sed(wave)
        if sed.unit.is_equivalent(u.erg/u.cm**2/u.s/u.AA) :
            # sed is already Flambda
            return sed
        elif sed.unit.is_equivalent(u.erg/u.cm**2/u.s/u.Hz) :
            # convert from Fnu to Flambda
            return (sed*c.c/wave**2).to(u.erg/u.cm**2/u.s/u.AA)          
        else: 
            print('sed has incorrect dimensions')
            
    @u.quantity_input(wave=u.m)    
    def fnu(self,wave):
        """ Return SED in Fnu"""
        sed=self.sed(wave)
        if sed.unit.is_equivalent(u.erg/u.cm**2/u.s/u.Hz) :
            # sed is already Fnu
            return sed
        if sed.unit.is_equivalent(u.erg/u.cm**2/u.s/u.AA) :
            # convert from Flambda to Fnu
            return (sed*wave**2/c.c).to(1/u.cm**2/u.s/u.Hz)   
        else: 
            print('sed has incorrect dimensions')


class Telescope :
    """ Class representing a telescope
        Given a name (or diameter and mirror array), provide methods
        area
        throughput
    """
    def __init__(self,name='',diameter=1*u.m,mirrors=['Al']) :
        self.name = name
        if name == 'ARC3.5m' :
            self.diameter=3.5*u.m
            self.mirrors=['Al','Al','Al']
            self.eps=0.4
        elif name == 'TMO' :
            self.diameter=0.6*u.m
            self.mirrors=['Al','Al']
            self.eps=0.4
        elif name == '' :
            self.diameter=diameter
            self.eps=0.
            self.mirrors=mirrors
        else :
            raise ValueError('unknown telescope')
    
    def area(self) :
        """ Return telescope collecting area
        """
        return (np.pi*(self.diameter/2)**2*(1-self.eps**2)).to(u.cm**2)
    
    def throughput(self,wave) :
        """ Return telescope throughput
        """
        t=np.ones(len(wave))
        for mir in self.mirrors :
            if type(mir) is float :
                # if mirrors is a float, use it as a constant throughput (bad practice!)
                t *= mir
            else :
                tmp = Mirror(mir)
                t *= tmp.reflectivity(wave)
        return t
            
class Mirror :
    """ class representing a mirror
        given coating name, provide method for reflectivity
    """
    def __init__(self,type,const=0.9) :
        self.type = type
        self.const = const
        
    def reflectivity(self,wave) :
        if self.type == 'const' :
            return np.ones(len(wave)) + self.const
            
        else :
            # read data file depending on mirror coating type
            try: dat=ascii.read(os.environ['ETC_DIR']+'/data/mirror/'+self.type+'.txt')
            except FileNotFoundError :
                raise ValueError('unknown coating type: {:s}',self.type)
            wav=dat['col1']
            ref=dat['col2']/100.
            interp=scipy.interpolate.interp1d(wav,ref)
            return interp(wave.to(u.nm).value)
    
class Instrument :
    """ class representing an Instrument
    """
    def __init__(self,name='',efficiency=0.8,pixscale=1,dispersion=1*u.AA) :
        self.name=name
        self.efficiency=efficiency
        self.pixscale=pixscale
        self.dispersion=dispersion
        self.detector = Detector(efficiency=1.,rn=5)
            
    def throughput(self,wave) :
        if self.name == '' :
            return np.ones(len(wave))*self.efficiency
        else :
            raise ValueError('need to implement instrument {:s}',self.name)
        
    def filter(self,wave,filter='',cent=5500*u.AA,wid=850*u.AA,trans=0.8) :
        if filter=='' :
            out = np.zeros(len(wave))
            out[np.where((wave>cent-wid/2)&(wave<cent+wid/2))] = trans
            return out
        else :
            try: dat=ascii.read(os.environ['ETC_DIR']+'/data/inst/'+self.name+'/'+filter+'.txt')
            except FileNotFoundError :
                raise ValueError('cant find filter file: {:s} for instrument {:s}',
                                 filter, self.name)

            raise ValueError('Name {:s} not yet implemented'.format(name))
        
class Detector :
    """ class representing a detector
    """
    def __init__(self,name='',efficiency=0.8,rn=5) :
        self.name=name
        self.efficiency=efficiency
        self.rn=rn
    
    def throughput(self,wave) :
        if self.name == '' :
            return np.ones(len(wave))*self.efficiency
        else :
            raise ValueError('need to implement detector {:s}',self.name)

    
class Atmosphere :
    """ class representing the Earth's atmosphere
    """
    def __init__(self,name='',transmission=0.8) :
        self.name=name
        self.transmission=transmission
    
    def throughput(self,wave) :
        if self.name == '' :
            return np.ones(len(wave))*self.transmission
        else :
            raise ValueError('need to implement atmosphere {:s}',self.name)
            
class Observation :
    """ Object representing an observation
    """
    def __init__(self,obj=None,atmos=None,telescope=None,instrument=None,wave=np.arange(3000,10000,1)*u.AA,seeing=1) :
        self.obj = obj
        self.atmos = atmos
        self.telescope = telescope
        self.instrument = instrument
        self.wave=wave
        self.seeing=seeing
        
    def photonflux(self) :
        """ Return photon flux
        """
        photflux = (self.obj.photflux(self.wave)* 
                self.atmos.throughput(self.wave) *
                self.telescope.area()*self.telescope.throughput(self.wave)*
                self.instrument.throughput(self.wave)*self.instrument.filter(self.wave) )
        return photflux
    
    def counts(self) :
        """ Return integral of photon flux
        """
        return np.trapz(self.photonflux(),self.wave)

    def sn(self,t) :
        """ Calculate S/N given exposure time
        """
        npix = np.pi*self.seeing**2*self.instrument.pixscale**2
        sn = self.counts()*t/np.sqrt(self.counts()*t+npix*self.instrument.detector.rn**2)
        sn_wave = ( self.photonflux()*self.instrument.dispersion*t/
                    np.sqrt(self.photonflux()*self.instrument.dispersion*t+npix*self.instrument.detector.rn**2) )
        return sn, sn_wave

    def exptime(self,sn) :
        """ Calculate exptime given S/N
            sn**2 (St + BAt + Npix*rn**2)  = S**2*t**2
            S**2 t**2 - sn**2*(S+BA)*t - sn**2*npix*rn**2
        """
        S = self.counts().value
        npix = np.pi*self.seeing**2*self.instrument.pixscale**2
        a = S**2
        b = -sn**2*S
        c = -sn**2*npix*self.instrument.detector.rn**2
        t = (-b + np.sqrt(b**2 - 4*a*c)) / 2 / a
        #t = sn**2 / self.counts()

        S = self.photonflux().value
        a = S**2
        b = -sn**2*S
        t_wave = (-b + np.sqrt(b**2 - 4*a*c)) / 2 / a
        #t_wave = sn**2 / self.photonflux()
        return t, t_wave
                

def signal(wave=np.arange(4000,8000,5)*u.AA) :  

    # set up object, telescope, instrument, atmosphere
    obj=Object(type='bb',teff=6000,mag=20)
    tel=Telescope(diameter=0.6*u.m,mirrors=[1.0])
    inst=Instrument(efficiency=0.5)
    atmos=Atmosphere(transmission=0.8)

    obs=Observation(obj=obj,atmos=atmos,telescope=tel,instrument=inst,wave=wave)
    plt.plot(wave,obs.photonflux())
    counts=obs.counts()
    print(obs.counts())

    #individual components for plotting, demonstrating use of object methods
    photflux=obj.photflux(wave)
    plt.plot(wave,photflux)
    photflux*=tel.throughput(wave)*tel.area()
    plt.plot(wave,photflux)
    photflux*=atmos.throughput(wave)
    plt.plot(wave,photflux)
    photflux*=inst.throughput(wave)
    plt.plot(wave,photflux)
    photflux*=inst.filter(wave,trans=0.8)
    plt.plot(wave,photflux)

    print(counts)
    pdb.set_trace()
    sn,sn_wave = obs.sn(t=1*u.s)
    t,t_wave = obs.exptime(sn=100)
    plt.figure()
    plt.plot(wave,sn_wave)
    plt.plot(wave,t_wave)
    return counts


