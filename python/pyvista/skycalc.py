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
from astropy.table import Table
from datetime import datetime

def calendar(obs='apo',tz='US/Mountain',year=2023,plot=False) :
    """ Get LST at midnight and moon phase and position for every night of a calendar year"
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
    time_resolution = 10.0*units.minute
    suntimes = time_grid_from_range(time_range, time_resolution=time_resolution)
    sun_alt = site.altaz(suntimes, coord.get_sun(suntimes)).alt
    # Sunrise = altitude was below horizon, now is above horizon:
    horizon = 0*units.deg
    #approx_sunrises = np.argwhere((sun_alt[:-1] < horizon) & (sun_alt[1:] >= horizon)) + 1
    sun1 = np.where((sun_alt[:-1] < horizon) & (sun_alt[1:] >= horizon))[0] 
    sun2 = np.where((sun_alt[:-1] < horizon) & (sun_alt[1:] >= horizon))[0] + 1
    sunrises = suntimes[sun1] + (0-sun_alt[sun1])/(sun_alt[sun2]-sun_alt[sun1])*(suntimes[sun2]-suntimes[sun1])

    # Sunset = altitude was above horizon, now is below horizon: 
    #approx_sunsets = np.argwhere((sun_alt[:-1] > horizon) & (sun_alt[1:] <= horizon)) + 1 
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
    local=site.astropy_time_to_datetime(times)
    out=Table()
    out['date'] = local
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
        plt.plot(times.value,lsts[0]+lsts[1]/60.+lsts[2]/3600.)
        plt.xlabel('JD')
        plt.ylabel('LST (midnight)')

    return out

    
def object(ra=0., dec=0., obs='apo', date=None,name='object',plot=False,tz='US/Mountain') :

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

    print('\nObject at RA={:}, DEC={:}'.format(ra,dec))

    # loop over all UTC hours for this date (would prefer local!)
    #print('{:8s}{:8s}{:8s}{:8s}{:8s}{:9s}{:8s}{:8s} {:16s}{:20s}'.format(
    #      'Local','UT','LST','HA','Airmass','ParAng','Phase','Moon Alt','Moon RA', 'Moon DEC'))
    #for hr in np.arange(24) :
    #    time = Time('{:s} {:d}:00:00'.format(date,hr),scale='utc',
    #           location=(site.location.lon,site.location.lat),precision=0)
    #    sun=get_sun(time)
    #    if site.sun_altaz(time).alt.value > 10 : continue
#
#        moon=get_moon(time)
#
#        val=site.altaz(time,obj).secz
#        if val < 0 :
#            airmass = '     ...'
#        else :
#            airmass = '{:8.2f}'.format(val)
#        val=site.moon_altaz(time).alt.value 
#        if val < 0 :
#            moonalt = '     ...'
#        else :
#            moonalt = '{:8.2f}'.format(val)
#        lst=time.sidereal_time('mean').hms
#        ha=site.target_hour_angle(time,obj)
#        ha.wrap_angle=180 *units.deg
#        local=site.astropy_time_to_datetime(time)
#        print('{:02d}:{:02d}  {:02d}:{:02d}  {:02d}:{:02d} {:3d}:{:02d} {:8s} {:8.2f} {:8.2f} {:8s} {:s} {:s}'.format(
#               local.hour,local.minute,
#               time.datetime.hour,time.datetime.minute,
#               int(round(lst[0])),int(round(lst[1])),
#               int(round(ha.hms[0])),int(abs(round(ha.hms[1]))),
#               airmass,
#               #site.altaz(time,obj).secz,
#               site.parallactic_angle(time,obj).deg,
#               astroplan.moon_illumination(time),moonalt,
#               str(moon.ra.to_string(units.hour)), str(moon.dec)))

    if plot: 
        fig,ax = plt.subplots(2,1)
        fig.subplots_adjust(hspace=0.001)

        plot_airmass(obj,site,time,ax=ax[0])
        plot_parallactic(obj,site,time,ax=ax[1])

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

    pdb.set_trace()

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


def parang(hd,obs='apo',tz='US/Mountain') :
    """ Calculates parallactic angle given header information DATE-OBS, RA, DEC, plus observatory
    """
    site=Observer.at_site(obs,timezone=tz)
    time=Time(hd.header['DATE-OBS'])
    obj=FixedTarget(coord=SkyCoord(hd.header['RA']+'h',hd.header['DEC']+'d'))
    print(time)
    print(obj)
    return site.parallactic_angle(time,obj).deg

