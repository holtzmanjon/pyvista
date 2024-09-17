import numpy as np
import matplotlib.pyplot as plt
import erfa
from holtztools import plots

import pdb
import astroplan
from astroplan import Observer, time_grid_from_range
from astroplan import FixedTarget
from astroplan.plots import plot_airmass
from astroplan.plots import plot_parallactic
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy.coordinates import get_moon, get_sun
from astropy import coordinates as coord
from astropy import units
from astropy.table import Table
from datetime import datetime
from astropy.visualization import time_support

def calendar(obs='apo',tz='US/Mountain',year=2023,plot=False,sun_dt=10.) :
    """ Get LST at midnight and moon phase and position for every night of a calendar year"

        Parameters
        ----------
        obs : str, default='apo'
                   observatory to compute for
        tz : str, default='US/Mountain'
                   timezone
        year : float, default=2023
                   year to calculate for
        plot : bool, default=False
                   if True, plot LST midnight for year
    """

    # set the site
    site=Observer.at_site(obs,timezone=tz)

    # times at local midnight
    time1 = Time(site.datetime_to_astropy_time(datetime(year,1,1,0,00)),
                location=(site.location.lon,site.location.lat),precision=0)
    time2 = Time(site.datetime_to_astropy_time(datetime(year+1,1,1,0,00)),
                location=(site.location.lon,site.location.lat),precision=0)
    time_range = Time([time1,time2])

    print('calculating approximate sunrise/sunset, make take a bit ...')
    # Measure the altitude of the Sun at each time
    time_resolution = sun_dt*units.minute
    suntimes = time_grid_from_range(time_range, time_resolution=time_resolution)
    sun_alt = site.altaz(suntimes, coord.get_sun(suntimes)).alt

    # Sunrise = altitude was below horizon, now is above horizon:
    horizon = 0*units.deg
    sun1 = np.where((sun_alt[:-1] < horizon) & (sun_alt[1:] >= horizon))[0] 
    sun2 = np.where((sun_alt[:-1] < horizon) & (sun_alt[1:] >= horizon))[0] + 1
    sunrises = suntimes[sun1] + (0-sun_alt[sun1])/(sun_alt[sun2]-sun_alt[sun1])*(suntimes[sun2]-suntimes[sun1])

    # Sunset = altitude was above horizon, now is below horizon: 
    sun1 = np.where((sun_alt[:-1] > horizon) & (sun_alt[1:] <= horizon))[0] 
    sun2 = np.where((sun_alt[:-1] > horizon) & (sun_alt[1:] <= horizon))[0] + 1
    sunsets = suntimes[sun1] + (0-sun_alt[sun1])/(sun_alt[sun2]-sun_alt[sun1])*(suntimes[sun2]-suntimes[sun1])

    # now sample at 1 day resolution
    time_resolution = 1.0*units.day
    times = time_grid_from_range(time_range, time_resolution=time_resolution)

    # get LSTs and Moon information
    lsts=times.sidereal_time('mean',longitude=site.location.lon).hms
    illums=astroplan.moon_illumination(times)
    moons = get_moon(times)

    # store output in table
    pdb.set_trace()
    local=site.astropy_time_to_datetime(times)
    out=Table()
    out['date'] = times.isot # local
    out['local'] = local.strftime('%H:%M')
    out['time_sunrise'] = site.astropy_time_to_datetime(sunrises)
    out['time_sunset'] = site.astropy_time_to_datetime(sunsets)
    out['sunrise'] = '00:00'
    out['sunset'] = '00:00'
    out['lst_h'] = np.array(lsts[0]).astype(int)
    out['lst_m'] = np.array(lsts[1]).astype(int)
    out['LSTmidnight'] = '00:00'
    out['moon_illum'] = illums
    out['moon_rah'] =  moons.ra.hms.h.astype(int)
    out['moon_ram'] =  moons.ra.hms.m.astype(int)
    out['moon_decd'] =  moons.dec.dms.d.astype(int)
    out['moon_decm'] =  moons.dec.dms.m.astype(int)
    out['moon_ra'] = '00:00'
    out['moon_dec'] = ' 00:00'
    for line in out :
        sunrise =line['time_sunrise'].time()
        line['sunrise']  = '{:02d}:{:02d}'.format(sunrise.hour,sunrise.minute)
        sunset =line['time_sunset'].time()
        line['sunset']  = '{:02d}:{:02d}'.format(sunset.hour,sunset.minute)
        line['LSTmidnight'] = '{:02d}:{:02d}'.format(line['lst_h'],line['lst_m'])
        line['moon_ra'] = '{:02d}:{:02d}'.format(line['moon_rah'],line['moon_ram'])
        if line['moon_decm'] < 0 : sign='-'
        else : sign=' '
        line['moon_dec'] = '{:s}{:02d}:{:02d}'.format(sign,abs(line['moon_decd']),abs(line['moon_decm']))
    out['moon_illum'].info.format = '7.2f'
    out.remove_column('time_sunrise')
    out.remove_column('time_sunset')
    out.remove_column('lst_h')
    out.remove_column('lst_m')
    out.remove_column('moon_rah')
    out.remove_column('moon_ram')
    out.remove_column('moon_decd')
    out.remove_column('moon_decm')

    if plot :
        time_support()

        plt.plot(times.datetime,lsts[0]+lsts[1]/60.+lsts[2]/3600.,label='LST midnight')
        val = []
        for time in sunsets :
            dt=site.astropy_time_to_datetime(time)
            val.append(dt.hour+dt.minute/60.+dt.second/3600.)
        plt.plot(times.datetime,val,label='Sunset')
        val = []
        for time in sunrises :
            dt=site.astropy_time_to_datetime(time)
            val.append(dt.hour+dt.minute/60.+dt.second/3600.)
        plt.plot(times.datetime,val,label='Sunrise')
        plt.xlabel('Date')
        plt.ylabel('Time')
        plt.legend(fontsize='x-small')

        # create the secondary axis showing mjd at the top
        def plot2mjd(t):
            '''Convert from matplotlib plot date to mjd'''
            return Time(t, format="plot_date").mjd


        def mjd2plot(mjd):
            '''Convert from mjd to matplotlib plot'''
            return Time(mjd, format="mjd").plot_date

        mjd_ax = plt.gca().secondary_xaxis('top', functions=(plot2mjd, mjd2plot))
        mjd_ax.set_xlabel('MJD')

    return out

    
def object(ras=[], decs=[], names=[], file=None, obs='apo', date=None,plot=False,tz='US/Mountain') :
    """  Get airmass table for specified object position, observatory, date

    Parameters
    ----------
    ras : float, str, or array-like
          input RA(s) of objects (sexagesimal hh:mm:ss or degrees)
    decs : float, str, or array-like
          input DEC(s) of objects (sexagesimal dd:mm:ss or degrees)
    names : str, or array-like
          name(s) of  objects
    file : str
          filename to read name, ra, dec from
    obs : str
          observatory for site
    plot : bool
          show plots of airmass and parallactic angle?
    tz : str
         timezone
    """

    # set the site
    site=Observer.at_site(obs,timezone=tz)

    # basic information for the night
    if date == None : date = Time.now().iso.split()[0]
    time = Time('{:s} 00:00:00'.format(date),scale='utc',
                location=(site.location.lon,site.location.lat),precision=0)
    sunset =site.sun_set_time(time)
    civil = site.twilight_evening_civil(time)
    nautical = site.twilight_evening_nautical(time)
    astronomical = site.twilight_evening_astronomical(time)
    for t in [sunset,civil,nautical,astronomical] :
        t.format='isot'
        t.precision=0
    print('Observatory: ',obs)
    print('Sunset: ',sunset)
    print('Civil twilight: ',civil)
    print('Nautical twilight: ',nautical)
    print('Astronomical twilight: ',astronomical)

    if plot: 
        fig,ax = plots.multi(1,2,sharex=True,hspace=0.001)

    # set the objects
    if file is not None :
        fp=open(file,'r')
        for line in fp:
            names.append(line.split()[0])
            ras.append(line.split()[1])
            decs.append(line.split()[2])
        fp.close()
    else :
        if isinstance(ras,float) or isinstance(ras,int) or isinstance(ras,str) : ras=[ras]
        if isinstance(decs,float) or isinstance(decs,int) or isinstance(decs,str) : decs=[decs]
        if isinstance(names,float) or isinstance(names,str) : names=[names]
        if len(names) != len(ras) : 
            names=[]
            for ra in ras : names.append('object')

    for ra,dec,name in zip(ras,decs,names) :
        # object information
        print('\nObject at RA={:}, DEC={:}'.format(ra,dec))
        if isinstance(ra,float) or isinstance(ra,int) :
            obj=FixedTarget(name=name,coord=SkyCoord(str(ra)+'d',str(dec)+'d'))
        else :
            obj=FixedTarget(name=name,coord=SkyCoord(ra+'h',dec+'d'))

        if plot :
            plot_airmass(obj,site,time,ax=ax[0])
            ax[0].legend()
            plot_parallactic(obj,site,time,ax=ax[1])
            ax[1].legend()

        time1 = Time('{:s} {:d}:00:00'.format(date,0),scale='utc',
                   location=(site.location.lon,site.location.lat),precision=0)
        time2 = Time('{:s} {:d}:00:00'.format(date,23),scale='utc',
                   location=(site.location.lon,site.location.lat),precision=0)
        time_range = Time([time1,time2])
        time_resolution = 1.0*units.hour
        times = time_grid_from_range(time_range, time_resolution=time_resolution)
        sun=get_sun(times)
        gd = np.where(site.sun_altaz(times).alt.value < 10)[0]
        times = times[gd]

        # get LSTs and Moon information
        lsts=times.sidereal_time('mean',longitude=site.location.lon).hms
        illums=astroplan.moon_illumination(times)
        moons = get_moon(times)
        airmass=site.altaz(times,obj).secz
        moonalt=site.moon_altaz(times).alt.value 
        local=site.astropy_time_to_datetime(times)
        parallactic_angle =  site.parallactic_angle(times,obj).deg
        ha=site.target_hour_angle(times,obj)
        ha.wrap_angle=180 *units.deg

        out=Table()
        out['date'] = local
        out['time'] = times
        out['LOCAL'] = '00:00'
        out['UT'] = '00:00'
        out['ha_h'] = ha.hms[0].astype(int)
        out['ha_m'] = ha.hms[1].astype(int)
        out['HA'] = ' 00:00'
        out['lst_h'] = np.array(lsts[0]).astype(int)
        out['lst_m'] = np.array(lsts[1]).astype(int)
        out['LST'] = '00:00'
        out['AIRMASS'] = airmass
        out['PARALLACTIC_ANGLE'] = parallactic_angle
        out['MOON_PHASE'] = astroplan.moon_illumination(times)
        out['MOON_ALT'] = moonalt
        out['moon_rah'] =  moons.ra.hms.h.astype(int)
        out['moon_ram'] =  moons.ra.hms.m.astype(int)
        out['moon_decd'] =  moons.dec.dms.d.astype(int)
        out['moon_decm'] =  moons.dec.dms.m.astype(int)
        out['MOON_RA'] = '00:00'
        out['MOON_DEC'] = ' 00:00'
        out['PARALLACTIC_ANGLE'].info.format = '7.2f'
        out['AIRMASS'].info.format = '7.2f'
        out['MOON_ALT'].info.format = '7.2f'
        out['MOON_PHASE'].info.format = '7.2f'
    
        for line in out :
            line['LOCAL'] = line['date'].time().strftime('%H:%M')
            line['UT'] = line['time'].datetime.strftime('%H:%M')
            if line['ha_m'] < 0 : sign='-'
            else : sign=' '
            line['HA'] = '{:s}{:02d}:{:02d}'.format(sign,abs(line['ha_h']),abs(line['ha_m']))
            line['LST'] = '{:02d}:{:02d}'.format(line['lst_h'],line['lst_m'])
            line['MOON_RA'] = '{:02d}:{:02d}'.format(line['moon_rah'],line['moon_ram'])
            if line['moon_decm'] < 0 : sign='-'
            else : sign=' '
            line['MOON_DEC'] = '{:s}{:02d}:{:02d}'.format(sign,abs(line['moon_decd']),abs(line['moon_decm']))

        out.remove_column('date')
        out.remove_column('time')
        out.remove_column('ha_h')
        out.remove_column('ha_m')
        out.remove_column('lst_h')
        out.remove_column('lst_m')
        out.remove_column('moon_rah')
        out.remove_column('moon_ram')
        out.remove_column('moon_decd')
        out.remove_column('moon_decm')
        print(out)

    return out

def airmass(header,obs=None) :
    """ Get airmass from header cards

        Tries AIRMASS, AIRMAS, and SECZ (in that order) first
        otherwise computes from DATE-OBS, RA, DEC if obs= is given
    """

    if 'AIRMASS' in header :
        return header['AIRMASS']
    elif 'AIRMAS' in header :
        return header['AIRMAS']
    elif 'SECZ' in header :
        return header['SECZ']
    elif 'ALT' in header :
        return 1./np.cos(header['ALT']*np.pi/180.)
    elif 'DATE-OBS' in header and obs is not None:
        site=Observer.at_site(obs)
        time=Time(header['DATE-OBS'])
        ra=header['RA']
        dec=header['DEC']
        if type(ra) is float :
            obj=FixedTarget(name=name,coord=SkyCoord(str(ra)+'d',str(dec)+'d'))
        else :
            obj=FixedTarget(name=name,coord=SkyCoord(ra+'h',dec+'d'))
        val=site.altaz(time,obj).secz
        print('airmass computed from DATE-OBS')
        return val
    else :
        raise ValueError('no AIRMASS/AIRMAS/SECZ card in header and no obs specified')


def parang_header(hd,obs='apo',tz='US/Mountain') :
    """ Calculates parallactic angle given header information DATE-OBS, RA, DEC, plus observatory
    """
    site=Observer.at_site(obs,timezone=tz)
    time=Time(hd.header['DATE-OBS'])
    obj=FixedTarget(coord=SkyCoord(hd.header['RA']+'h',hd.header['DEC']+'d'))
    print('Time: ',time)
    print('Object: ',obj)
    return site.parallactic_angle(time,obj).deg

def parang(h,dec,site='APO') :
    """ Calculate parallactic angle from HA, DEC, site

    Parameters
    ----------
    ha : hour angle, hrs
    dec : declination, degrees
    site : str, site name, default='APO'

    Returns
    -------
    parallactic angle in degrees
    """
    phi=EarthLocation.of_site(site).lat.value*np.pi/180.
    pa = np.arctan2(np.sin(h*15*np.pi/180),np.cos(dec*np.pi/180.)*np.tan(phi)-np.sin(dec*np.pi/180)*np.cos(h*15*np.pi/180.))
    return pa*180/np.pi

def zd(h,dec,site='APO') :
    """ Calculate zenith distance from HA, DEC, site

    Parameters
    ----------
    ha : hour angle, hrs
    dec : declination, degrees
    site : str, site name, default='APO'

    Returns
    -------
    zenith distance in degrees
    """
    phi = EarthLocation.of_site(site).lat.value*np.pi/180.
    z = (np.sin(phi)*np.sin(dec*np.pi/180.)+np.cos(phi)*np.cos(dec*np.pi/180)*np.cos(h*15*np.pi/180))
    return np.arccos(z)*180/np.pi

def refraction(obs=None,wav=0.5,h=2000,temp=20,rh=0.25) :
    """ Calculate coefficient of refraction
    """
    # set the site
    if obs != None :
        site=Observer.at_site(obs,timezone='US/Mountain')
        h=site.location.height.value

    p0=101325
    M = 0.02896968
    g = 9.80665
    T0 = 288.16
    R0 = 8.314462618
    pressure = p0 *np.exp(-g*h*M/T0/R0)*10
    tsl=temp+273
    pressure = 1013.25 * np.exp ( -h / ( 29.3 * tsl ) )
    ref=erfa.refco(pressure,temp,rh,wav)[0]*206265
    return ref


def readtui(file) :
    """ Opens and parses TUI-type file and returns columns 1, 2, and 3 as lists (e.g. names, ras, decs)
    """
    fp=open(file,'r')
    names=[]
    ras=[]
    decs=[]
    for line in fp:
        names.append(line.split()[0])
        ras.append(line.split()[1])
        decs.append(line.split()[2])
    fp.close()
    return names, ras, decs
