import numpy as np
import matplotlib.pyplot as plt

import pdb
import astroplan
from astroplan import Observer, time_grid_from_range
from astroplan import FixedTarget
from astroplan.plots import plot_airmass
from astroplan.plots import plot_parallactic
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates import get_moon, get_sun
from astropy import coordinates as coord
from astropy import units

def calendar(obs='apo',tz='US/Mountain',year=2023,plot=False) :


    # set the site
    site=Observer.at_site(obs,timezone=tz)

    print('{:<20s} {:s}  {:s} {:s}'.format('DATE','LST(midnight)','MOONILLUM', 'MOON position'))


    time1 = Time('{:4d}-{:02d}-01 12:00:00'.format(year,1), scale='utc',
                location=(site.location.lon,site.location.lat),precision=0)
    time2 = Time('{:4d}-{:02d}-01 12:00:00'.format(year+1,1), scale='utc',
                location=(site.location.lon,site.location.lat),precision=0)
    time_range = Time([time1,time2])
    time_resolution = 1.0*units.day
    times = time_grid_from_range(time_range, time_resolution=time_resolution)
    lsts=times.sidereal_time('mean',longitude=site.location.lon).hms
    illums=astroplan.moon_illumination(times)
    moons = get_moon(times)

    for time,h,m,illum,moon in zip(times,lsts[0],lsts[1],illums,moons) :
        if moon.dec < 0 : sign='-'
        else : sign = ' '
        print('{:<12s} {:02d}:{:02d}      {:.2f}      {:02d}:{:02d} {:s}{:02d}:{:02d}'.format(
               time.iso.split()[0],int(h),int(m),illum,
               int(moon.ra.hms.h),  int(moon.ra.hms.m), sign,int(abs(moon.dec.dms.d)),int(abs(moon.dec.dms.m))))

    if plot :
        plt.plot(times.value,lsts[0]+lsts[1]/60.+lsts[2]/3600.)
        plt.xlabel('JD')
        plt.ylabel('LST (midnight)')

    return

    # Measure the altitude of the Sun at each time
    sun_alt = site.altaz(times, coord.get_sun(times)).alt

    # Sunrise = altitude was below horizon, now is above horizon:
    horizon = 5*units.deg
    approx_sunrises = np.argwhere((sun_alt[:-1] < horizon) & (sun_alt[1:] >= horizon)) + 1

    # Sunset = altitude was above horizon, now is below horizon: 
    approx_sunsets = np.argwhere((sun_alt[:-1] > horizon) & (sun_alt[1:] <= horizon)) + 1 
    
    times=[]
    for month in range(1,13) :
        time = Time('{:4d}-{:02d}-01 12:00:00'.format(year,month), scale='utc',
                location=(site.location.lon,site.location.lat),precision=0)
        times.append(time)

    for time in times :
        #sunrise =site.sun_rise_time(time)
        #sunset =site.sun_set_time(time)
        ##civil = site.twilight_evening_civil(time)
        ##nautical = site.twilight_evening_nautical(time)
        #astronomical_eve = site.twilight_evening_astronomical(time)
        #astronomical_morn = site.twilight_morning_astronomical(time)
        ##for t in [sunset,sunrise,astronomical_eve,astronomical_morn] :
        ##    t.format='isot'
        ##    t.precision=0
        lst=time.sidereal_time('mean').hms
        date = time.iso.split()[0]
        #moon=get_moon(time)
        #print('{:<20s} {:02d}:{:02d}  {:02d}:{:02d}    {:02d}:{:02d}  {:02d}:{:02d}   {:02d}:{:02d}   {:.2f}  {:s}  {:s}'.format(
        #       date,
        #       site.astropy_time_to_datetime(sunset).hour,site.astropy_time_to_datetime(sunset).minute,
        #       site.astropy_time_to_datetime(sunrise).hour,site.astropy_time_to_datetime(sunrise).minute,
        #       site.astropy_time_to_datetime(astronomical_eve).hour,site.astropy_time_to_datetime(astronomical_eve).minute,
        #       site.astropy_time_to_datetime(astronomical_morn).hour,site.astropy_time_to_datetime(astronomical_morn).minute,
        #       int(round(lst[0])),int(round(lst[1])),
        #       astroplan.moon_illumination(time),
        #       str(moon.ra.to_string(units.hour)), str(moon.dec)))


def table(ra=0., dec=0., obs='apo', date=None,name='object',plot=False,tz='US/Mountain') :

    """  Get airmass table for specified object position, observatory, date
    """

    # set the site
    site=Observer.at_site(obs,timezone=tz)

    # set the objects
    if type(ra) is float :
        obj=FixedTarget(name=name,coord=SkyCoord(str(ra)+'d',str(dec)+'d'))
    else :
        obj=FixedTarget(name=name,coord=SkyCoord(ra+'h',dec+'d'))

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


    # loop over all UTC hours for this date (would prefer local!)
    print('{:8s}{:8s}{:8s}{:8s}{:8s}{:9s}{:8s}{:8s} {:16s}{:20s}'.format(
          'Local','UT','LST','HA','Airmass','ParAng','Phase','Moon Alt','Moon RA', 'Moon DEC'))
    for hr in np.arange(24) :
        time = Time('{:s} {:d}:00:00'.format(date,hr),scale='utc',
               location=(site.location.lon,site.location.lat),precision=0)
        sun=get_sun(time)
        if site.sun_altaz(time).alt.value > 10 : continue

        moon=get_moon(time)

        val=site.altaz(time,obj).secz
        if val < 0 :
            airmass = '     ...'
        else :
            airmass = '{:8.2f}'.format(val)
        val=site.moon_altaz(time).alt.value 
        if val < 0 :
            moonalt = '     ...'
        else :
            moonalt = '{:8.2f}'.format(val)
        lst=time.sidereal_time('mean').hms
        ha=site.target_hour_angle(time,obj)
        ha.wrap_angle=180 *units.deg
        local=site.astropy_time_to_datetime(time)
        print('{:02d}:{:02d}  {:02d}:{:02d}  {:02d}:{:02d} {:3d}:{:02d} {:8s} {:8.2f} {:8.2f} {:8s} {:s} {:s}'.format(
               local.hour,local.minute,
               time.datetime.hour,time.datetime.minute,
               int(round(lst[0])),int(round(lst[1])),
               int(round(ha.hms[0])),int(abs(round(ha.hms[1]))),
               airmass,
               #site.altaz(time,obj).secz,
               site.parallactic_angle(time,obj).deg,
               astroplan.moon_illumination(time),moonalt,
               str(moon.ra.to_string(units.hour)), str(moon.dec)))

    if plot: 
        fig,ax = plt.subplots(2,1)
        fig.subplots_adjust(hspace=0.001)

        plot_airmass(obj,site,time,ax=ax[0])
        plot_parallactic(obj,site,time,ax=ax[1])

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


def parang(hd,obs='apo',tz='US/Mountain') :
    """ Calculates parallactic angle given header information DATE-OBS, RA, DEC, plus observatory
    """
    site=Observer.at_site(obs,timezone=tz)
    time=Time(hd.header['DATE-OBS'])
    obj=FixedTarget(coord=SkyCoord(hd.header['RA']+'h',hd.header['DEC']+'d'))
    print(time)
    print(obj)
    return site.parallactic_angle(time,obj).deg

