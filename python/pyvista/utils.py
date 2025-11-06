# encoding: utf-8
#
# @Author: Jon Holtzman
# @Date: March 2018
# @Filename: spectra.py
# @License: BSD 3-Clause
# @Copyright: Jon Holtzman

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import copy
from astropy.coordinates import SkyCoord, EarthLocation, Angle
import astropy.units as u
from astropy.time import Time, TimeDelta

# utility routines for working with spectra

def fits2vector(header,axis) :
    """ Routine to return vector of axis values from a FITS header CRVAL, CDELT, NAXIS for specified axis
    """
    caxis='{:1d}'.format(axis)
    return header['CRVAL'+caxis]+header['CDELT'+caxis]*np.arange(header['NAXIS'+caxis])

def add_dim(header,crval,cdelt,crpix,ctype,idim) :
    """ Add a set of CRVAL/CDELT,CRPIX,CTYPE cards to header
               wavelength, in Angstroms
    """
    header.append(('CRVAL{:d}'.format(idim),crval))
    header.append(('CDELT{:d}'.format(idim),cdelt))
    header.append(('CRPIX{:d}'.format(idim),crpix))
    header.append(('CTYPE{:d}'.format(idim),ctype))

def vactoair(wave,type='ciddor',t=0, x_CO2=450, p=101325, rh=50) :
    """ Convert vacuum to air wavelengths using specified conversion

        Parameters
        ==========
        wave : float or array-like
        type : str, default='ciddor'
               'ciddor" for NIST+Ciddor formulae (https://emtoolbox.nist.gov/Wavelength/Documentation.asp), 'greisen' for Greisen 2006, 
                or 'idl' for IDL Users library (incomplete?) implementation of Ciddor
        t : float, default=0
            temperature, only used for Ciddor
        x_CO2 : float, default=450
                CO2 fraction, only used for Ciddor
        p : float, default=101325
            pressure, only used for Ciddor
        rh : float, default=50
             relative humidity, only used for Ciddor
    """

    if type == 'ciddor' :
        return ciddor_vac_to_air(wave,t=t,x_CO2=x_CO2,p=p,rh=rh)
    elif type == 'greisen' :
        return greisen_vac_to_air(wave)
    elif type == 'idl' :
        return idl_vac_to_air(wave)
    else :
        raise ValueError('No such conversion type!')

def airtovac(wave,type='ciddor',t=0, x_CO2=450, p=101325, rh=50) :
    """ Convert air to vacuum wavelengths using specified conversion

        Parameters
        ==========
        wave : float or array-like
        type : str, default='ciddor'
               'ciddor" for NIST+Ciddor formulae (https://emtoolbox.nist.gov/Wavelength/Documentation.asp), 'greisen' for Greisen 2006, 
                or 'idl' for IDL Users library (incomplete?) implementation of Ciddor
        t : float, default=0
            temperature, only used for Ciddor
        x_CO2 : float, default=450
                CO2 fraction, only used for Ciddor
        p : float, default=101325
            pressure, only used for Ciddor
        rh : float, default=50
             relative humidity, only used for Ciddor
    """
    if type == 'ciddor' :
        return ciddor_air_to_vac(wave,t=t,x_CO2=x_CO2,p=p,rh=rh)
    elif type == 'greisen' :
        return greisen_air_to_vac(wave)
    elif type == 'idl' :
        return idl_air_to_vac(wave)
    else :
        raise ValueError('No such conversion type!')


def idl_vac_to_air(wave_vac) :
    """ Convert vacuum wavelengths to air wavelengths

        Corrects for the index of refraction of air under standard conditions.  
        Wavelength values below 2000 A will not be altered.  Accurate to about 10 m/s.

        From IDL Astronomy Users Library, which references Ciddor 1996 Applied Optics 35, 1566
    """
    if not isinstance(wave_vac, np.ndarray) : 
        vac = np.array([wave_vac])
    else :
        vac = wave_vac

    air = copy.copy(vac).flatten()
    g = np.where(vac >= 2000)     #Only modify above 2000 A
    sigma2 = (1.e4/vac[g] )**2.       #Convert to wavenumber squared

    # Compute conversion factor
    fact = 1. +  5.792105E-2/(238.0185E0 - sigma2) + 1.67917E-3/( 57.362E0 - sigma2)
    
    air[g] = vac.flatten()[g]/fact
    return np.reshape(air,vac.shape)

def idl_air_to_vac(wave_air) :
    """ Convert air wavelengths to vacuum wavelengths

        Corrects for the index of refraction of air under standard conditions.  
        Wavelength values below 2000 A will not be altered.  Accurate to about 10 m/s.

        From IDL Astronomy Users Library, which references Ciddor 1996 Applied Optics 35, 1566
    """
    if not isinstance(wave_air, np.ndarray) : 
        air = np.array([wave_air])
    else :
        air = wave_air

    vac = copy.copy(air).flatten()
    g = np.where(vac >= 2000)     #Only modify above 2000 A

    for iter in range(2) :
        sigma2 = (1e4/vac[g])**2.     # Convert to wavenumber squared
        # Compute conversion factor
        fact = 1. +  5.792105E-2/(238.0185E0 - sigma2) + 1.67917E-3/( 57.362E0 - sigma2)

        vac[g] = air.flatten()[g]*fact              #Convert Wavelength
    return np.reshape(vac,air.shape)

def greisen_air_to_vac(wavelength):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006
    """
    wlum = (wavelength*u.Angstrom).to(u.um).value
    return (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * wavelength

def greisen_vac_to_air(wavelength):
    """
    Greisen 2006 reports that the error in naively inverting Eqn 65 is less
    than 10^-9 and therefore acceptable.  This is therefore eqn 67
    """
    wlum = (wavelength*u.Angstrom).to(u.um).value
    nl = (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4))
    return wavelength/nl

def ciddor_index(wave,t=0,x_CO2=450, p=101325,rh=50) :
    """ Formulae from NIST, https://emtoolbox.nist.gov/Wavelength/Documentation.asp#CommentsRegardingtheCalculations

    Parameters:
    wave : wavelength in Angstroms
 
    """
    w0=295.235
    w1=2.6422
    w2=-0.03238
    w3=0.004028
    k0=238.0185
    k1=5792105
    k2=57.362
    k3=167917
    a0=1.58123e-6
    a1=-2.9331e-8
    a2=1.1043e-10
    b0=5.707e-6
    b1=-2.051e-8
    c0=1.9898e-4
    c1=-2.376e-6
    d=1.83e-11
    e=-0.765e-8
    p_R1=101325
    T_R1=288.15
    Z_a=0.9995922115
    rho_nu5=0.00985938
    R=8.314472
    M_nu=0.018015
 
    S = 1. / (wave/1e4)**2
 
    r_a5=1.e-8*(k1/(k0-S)+k3/(k2-S))
    r_nu5=1.022e-8*(w0+w1*S+w2*S**2+w3*S**3)

    M_a=0.0289635+1.2011e-8*(x_CO2-400)
    r_ax5=r_a5*(1+5.34e-7*(x_CO2-450))
 
    T=t+273.15

    K_1= 1.16705214528E+03
    K_2= -7.24213167032E+05
    K_3= -1.70738469401E+01
    K_4= 1.20208247025E+04
    K_5= -3.23255503223E+06
    K_6= 1.49151086135E+01
    K_7= -4.82326573616E+03
    K_8= 4.05113405421E+05
    K_9= -2.38555575678E-01
    K_10= 6.50175348448E+02

    Omega = T + K_9/(T-K_10)
    A = Omega**2 + K_1*Omega + K_2
    B = K_3*Omega**2 + K_4*Omega + K_5
    C = K_6*Omega**2 + K_7*Omega + K_8
    X = -B + np.sqrt(B**2-4*A*C)
    p_SV = 10**6*(2*C/X)**4

    def f(p,t) :
        alpha = 1.00062
        beta = 3.14e-8
        gamma = 5.60e-7 
        return alpha + beta*p + gamma*t**2

    x_nu = (rh/100)*f(p,t)*p_SV/p 
 
    Z_m = 1-(p/T)*(a0+a1*t+a2*t**2+(b0+b1*t)*x_nu+(c0+c1*t)*x_nu**2) + (p/T)**2*(d+e*x_nu**2)

    rho_ax5=p_R1*M_a/(Z_a*R*T_R1)
    rho_nu = x_nu*p*M_nu/(Z_m*R*T)
    rho_a = (1-x_nu)*p*M_a/(Z_m*R*T)

    n = 1 + (rho_a/rho_ax5)*r_ax5 + (rho_nu/rho_nu5)*r_nu5
    return n

def ciddor_vac_to_air(wave,t=0,x_CO2=450, p=101325,rh=50) :
    return wave/ciddor_index(wave,t=t,x_CO2=x_CO2,p=p,rh=rh)

def ciddor_air_to_vac(wave,t=0,x_CO2=450, p=101325,rh=50) :
    out=copy.copy(wave)
    for iter in range(3) :
        out = wave*ciddor_index(out,t=t,x_CO2=x_CO2,p=p,rh=rh)
    return out

def getbc(header=None,dateobs=None,ra=None,dec=None,exptime=None,obs='APO') :
    """ Get barycentric correction given DATE-OBS, RA, and DEC as char strings (sexagesimal), observatory location
    """

    if header is not None :
        ra = header['RA']
        dec = header['DEC']
        dateobs = header['DATE-OBS']
        exptime = header['EXPTIME']

    print('using observatory: ', obs)
    if isinstance(ra,str) :
        ra = Angle(ra+' hours')
    else :
        ra = ra*u.degree
    if isinstance(dec,str) :
        dec = Angle(dec+' degrees')
    else :
        dec = dec*u.degree
    t = Time(dateobs)
    sc = SkyCoord(ra,dec)
    barycorr = sc.radial_velocity_correction(kind='barycentric',obstime=t, location=EarthLocation.of_site(obs))
    ltt = t.light_travel_time(sc,location=EarthLocation.of_site(obs))
    exphalf = TimeDelta(exptime/2.*u.second)
    barytime = t.tdb + ltt + exphalf
    return barycorr.to(u.km/u.s), barytime

