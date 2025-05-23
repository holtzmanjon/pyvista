import numpy as np
import tempfile
try: from esutil import htm
except: print('esutil not available!')
from astropy.table import Table, Column
from astropy.io import fits
from holtztools import match
import os
import pdb
from astroquery.gaia import Gaia
import pyvo
import requests

def get(ra,dec,posn_match=30,vers='dr3',verbose=True,cols=None) :
    tab=Table()
    tab.add_column(Column(ra,name='my_ra'))
    tab.add_column(Column(dec,name='my_dec'))
    if cols is not None :
        for col in cols :
            tab.add_column(Column(col[0],name=col[1]))
    xmlfilename= tempfile.mktemp('.xml',dir=os.getcwd())
    tab.write(xmlfilename,format='votable',overwrite=True)
    if vers == 'dr3' :
        job= Gaia.launch_job_async(
            """SELECT p.source_id, g.ra, g.dec, g.pmra, g.pmdec, 
                      g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.phot_g_mean_mag,
                      g.azero_gspphot,g.ebpminrp_gspphot,
                      p.teff_gspspec, p.logg_gspspec, p.mh_gspspec,
                      p.alphafe_gspspec,p.fem_gspspec,p.sife_gspspec,p.cafe_gspspec,p.mgfe_gspspec,p.ndfe_gspspec,
                      p.feiim_gspspec,p.sfe_gspspec,p.zrfe_gspspec,p.nfe_gspspec,p.crfe_gspspec,p.cefe_gspspec,
                      p.nife_gspspec, p.azero_esphs, p.teff_esphs, p.logg_esphs,
                      distance(
                        point('', ids.my_ra, ids.my_dec),
                        point('', g.ra, g.dec)
                      ) * 3600 as dist_arcsec
               FROM gaiadr3.astrophysical_parameters AS p 
               JOIN gaiadr3.gaia_source AS g on p.source_id = g.source_id
               JOIN tap_upload.my_table as ids on 1 = contains(
                 point('', ids.my_ra, ids.my_dec),
                 circle('', g.ra, g.dec, {:f})
               )""".format(posn_match/3600.),
               upload_resource=xmlfilename,upload_table_name='my_table',verbose=verbose)
        posn_gaia = job.get_results()
        print('returned ', len(posn_gaia))
        os.remove(xmlfilename)
        return  posn_gaia
    elif vers == 'dr3_tap' :

        # Heidelberg GAIA TAP
        #service = pyvo.dal.TAPService("https://dc.zah.uni-heidelberg.de/tap")
        service = pyvo.dal.TAPService("https://gaia.ari.uni-heidelberg.de/tap")

        # VizieR
        #service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
        #source_id =  '"I/355/gaiadr3"'
        #xp = '"I/355/xpsample"'

        # AIP GAIA TAP
        #token='feb854aa631c089300cff1c1edfba95e18ff50ca'
        #tap_session = requests.Session()
        #tap_session.headers['Authorization'] = token
        #lang = 'ADQL'
        #query_name = "gaia_posn"
        #service = pyvo.dal.TAPService("https://gaia.aip.de/tap",
        #           session=tap_session)

        source_id =  'gaiadr3.gaia_source_lite'
        xp = 'gaiadr3.xp_sampled_mean_spectrum'
        #posn_gaia = service.submit_job(
        posn_gaia = service.search(
            """SELECT * FROM {:s} as g
               JOIN TAP_UPLOAD.coords as coords 
               ON contains(POINT('ICRS', g.ra, g.dec),
                           CIRCLE('ICRS',coords.my_ra, coords.my_dec,{:f})) = 1""".format(
                      source_id,posn_match/3600.),
               uploads={'coords' : tab})
        #       language=lang,runid=query_name,queue="30s")
        #posn_gaia.run()

        tab=Table()
        tab['SOURCE_ID'] = posn_gaia['source_id'].data
        xp_gaia = service.search(
            """SELECT * FROM {:s} as xp
               JOIN TAP_UPLOAD.coords as coords 
               ON coords.source_id = xp.source_id""".format(xp),
               uploads={'coords' : tab})

        retrieval_type = 'XP_CONTINUOUS'          # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS', 'ALL'
        data_structure = 'COMBINED'
        data_release   = 'Gaia DR3'     # Options are: 'Gaia DR3' (default), 'Gaia DR2'

        datalink = Gaia.load_data(ids=posn_gaia['source_id'], data_release = data_release, retrieval_type=retrieval_type, 
                                  data_structure = data_structure, verbose = False, output_file = None)
        dl_keys  = [inp for inp in datalink.keys()]
        dl_keys.sort()

        return posn_gaia, xp_gaia,datalink['XP_CONTINUOUS_COMBINED.xml'][0]   #.to_table()

def getdata(data,vers='dr2',posn_match=30,verbose=False) :
    """ Given input structure, get GAIA information from 2MASS matches
        and positional match
        Returns two tables
    """

    tab=Table()
    tab.add_column(Column(data['APOGEE_ID'],name='twomass'))
    tab.add_column(Column(data['RA'],name='apogee_ra'))
    tab.add_column(Column(data['DEC'],name='apogee_dec'))
    tab.add_column(Column(data['GAIAEDR3_SOURCE_ID'],name='edr3_source_id'))
    #if type(data['APOGEE_ID'][0]) is str or type(data['APOGEE_ID'][0]) is np.str_ : 
    try:
        j=np.where(np.core.defchararray.find(data['APOGEE_ID'],'2M') == 0)[0]
        out,ind=np.unique(np.core.defchararray.replace(data['APOGEE_ID'][j],'2M',''),return_index=True)
    except :
        j=np.where(np.core.defchararray.find(data['APOGEE_ID'],b'2M') == 0)[0]
        out,ind=np.unique(np.core.defchararray.replace(data['APOGEE_ID'][j],b'2M',b''),return_index=True)
    tab['twomass'][ind] = out
    #tab.add_column(Column(out,name='twomass'))
    #tab.add_column(Column(data['RA'][ind],name='apogee_ra'))
    #tab.add_column(Column(data['DEC'][ind],name='apogee_dec'))
    xmlfilename= tempfile.mktemp('.xml',dir=os.getcwd())
    tab.write(xmlfilename,format='votable',overwrite=True)
    if vers == 'dr2' :
        try :
            job= Gaia.launch_job_async(
                """SELECT tmass_match.original_ext_source_id, g.source_id, g.ra, g.dec, g.parallax, g.parallax_error, 
                           g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, g.ref_epoch,
                           g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, 
                           g.radial_velocity, g.radial_velocity_error, g.a_g_val, g.e_bp_min_rp_val, 
                           dist.r_est, dist.r_lo, dist.r_hi
                   FROM gaiadr2.gaia_source AS g
                   INNER JOIN gaiadr2.tmass_best_neighbour AS tmass_match ON tmass_match.source_id = g.source_id
                   INNER JOIN tap_upload.my_table as ids on ids.twomass = tmass_match.original_ext_source_id
                   LEFT OUTER JOIN external.gaiadr2_geometric_distance as dist ON  g.source_id = dist.source_id""",
                   upload_resource=xmlfilename,upload_table_name='my_table',verbose=verbose)
            twomass_gaia = job.get_results()
        except:
            print("error with gaia 2mass search")
            twomass_gaia = None
    elif vers == 'dr3' :
        try :
            job= Gaia.launch_job_async(
                """SELECT tmass_match.original_ext_source_id, p.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.phot_rp_mean_mag,
                          g.azero_gspphot,g.ebpminrp_gspphot, 
                          p.source_id, p.teff_gspspec, p.logg_gspspec, p.mh_gspspec,
                          p.alphafe_gspspec,p.fem_gspspec,p.sife_gspspec,p.cafe_gspspec,p.mgfe_gspspec,p.ndfe_gspspec,
                          p.feiim_gspspec,p.sfe_gspspec,p.zrfe_gspspec,p.nfe_gspspec,p.crfe_gspspec,p.cefe_gspspec,
                          p.nife_gspspec, p.azero_esphs, p.teff_esphs, p.logg_esphs 
                   FROM gaiadr3.astrophysical_parameters AS p
                   INNER JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS tmass_match ON tmass_match.source_id = p.source_id
                   INNER JOIN tap_upload.my_table as ids on ids.twomass = tmass_match.original_ext_source_id""",
                   upload_resource=xmlfilename,upload_table_name='my_table',verbose=verbose)
            twomass_gaia = job.get_results()
        except:
            print("error with gaia 2mass search")
            twomass_gaia = None
    else : twomass_gaia = None

    try: 
        if vers == 'dr2' :
            job= Gaia.launch_job_async(
                """SELECT  g.source_id, g.ra, g.dec, g.parallax, g.parallax_error, 
                           g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, g.ref_epoch,
                           g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, 
                           g.radial_velocity, g.radial_velocity_error, g.a_g_val, g.e_bp_min_rp_val, 
                           dist.r_est, dist.r_lo, dist.r_hi,
                           distance(
                             point('', ids.apogee_ra, ids.apogee_dec),
                             point('', g.ra, g.dec)
                           ) * 3600 as dist_arcsec
                   FROM gaiadr2.gaia_source as g
                   JOIN tap_upload.my_table as ids on 1 = contains(
                     point('', ids.apogee_ra, ids.apogee_dec),
                     circle('', g.ra, g.dec, {:f})
                   )
                   LEFT OUTER JOIN external.gaiadr2_geometric_distance as dist ON  g.source_id = dist.source_id""".format(posn_match/3600.),
                   upload_resource=xmlfilename,upload_table_name='my_table',verbose=verbose)
            posn_gaia = job.get_results()
            print('returned ', len(posn_gaia))
        elif vers == 'dr3' :
            job= Gaia.launch_job_async(
                """SELECT p.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.phot_rp_mean_mag,
                          g.azero_gspphot,g.ebpminrp_gspphot,
                          p.teff_gspspec, p.logg_gspspec, p.mh_gspspec,
                          p.alphafe_gspspec,p.fem_gspspec,p.sife_gspspec,p.cafe_gspspec,p.mgfe_gspspec,p.ndfe_gspspec,
                          p.feiim_gspspec,p.sfe_gspspec,p.zrfe_gspspec,p.nfe_gspspec,p.crfe_gspspec,p.cefe_gspspec,
                          p.nife_gspspec, p.azero_esphs, p.teff_esphs, p.logg_esphs 
                   FROM gaiadr3.astrophysical_parameters AS p 
                   JOIN gaiadr3.gaia_source AS g on p.source_id = g.source_id
                   JOIN tap_upload.my_table as ids on ids.edr3_source_id = p.source_id""",
                   upload_resource=xmlfilename,upload_table_name='my_table',verbose=verbose)
            posn_gaia = job.get_results()
            print('returned ', len(posn_gaia))
        elif vers == 'edr3' :
            service = pyvo.dal.TAPService("https://dc.zah.uni-heidelberg.de/tap")
            # this used to work but stopped
            #posn_gaia = service.search(
            #    """SELECT * FROM gedr3dist.litewithdist as g
            #       JOIN TAP_UPLOAD.coords as coords 
            #       ON contains(POINT('ICRS', g.ra, g.dec),CIRCLE('ICRS',coords.apogee_ra, coords.apogee_dec,{:f})) = 1""".format(posn_match/3600.),
            #       uploads={'coords' : tab})
            # Markus at GAVO recommended:
            posn_gaia = service.search(
                """WITH withpar AS (
                     SELECT *
                         FROM gaia.edr3lite AS db
                         JOIN TAP_UPLOAD.coords AS coords
                         ON distance(db.ra, db.dec, coords.apogee_ra, coords.apogee_dec)< {:f}) 
                     SELECT * from withpar
                     LEFT OUTER JOIN gedr3dist.main as dist using (source_id)
                """.format(posn_match/3600.), uploads={'coords' : tab},maxrec=1000000)
            print('pyvo returned: ',len(posn_gaia))

            #m1,m2=match.match(posn_gaia_archive['source_id'],posn_gaia['source_id'])
            #print(len(m1),len(m2))

    except: 
        print("error with gaia position search")
        posn_gaia = None
    finally: os.remove(xmlfilename)

    return twomass_gaia, posn_gaia


def add_gaia(data,vers='edr3') :
    """ Add GAIA data to input structure, with 2MASS match and coordinate match to (cross-matched) GAIA reference file
    """

    # get the GAIA data from both matches
    gaia_twomass, gaia_posn = getdata(data,vers=vers)

    # add new columns
    tab=Table(data)
    if vers == 'dr2'  :
        in_names=('source_id','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error',
                  'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','a_g_val', 'e_bp_min_rp_val',
                  'radial_velocity','radial_velocity_error', 'r_est','r_lo','r_hi')
        dtypes=('i8','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4')
    elif vers == 'dr3'  :
        in_names=('source_id',
                  'azero_gspphot','ebpminrp_gspphot',
                  'teff_gspphot', 'logg_gspphot', 'mh_gspphot',
                  'teff_gspspec', 'logg_gspspec', 'mh_gspspec',
                  'alphafe_gspspec','fem_gspspec','sife_gspspec','cafe_gspspec','mgfe_gspspec','ndfe_gspspec',
                  'feiim_gspspec','sfe_gspspec','zrfe_gspspec','nfe_gspspec','crfe_gspspec','cefe_gspspec',
                  'nife_gspspec', 'azero_esphs', 'teff_esphs', 'logg_esphs')
        dtypes=('i8','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4',
                'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4')
    elif vers == 'edr3' :
        in_names=('source_id','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error',
                  'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag',
                  'dr2_radial_velocity','dr2_radial_velocity_error', 'r_med_geo','r_lo_geo','r_hi_geo',
                  'r_med_photogeo','r_lo_photogeo','r_hi_photogeo')
        dtypes=('i8','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4')
    out_names=[]
    root = 'GAIA'+vers.upper()+'_'
    for name in in_names: out_names.append((root+name).upper())
    # initialize
    newcols=Table(np.zeros([len(tab),len(out_names)]),names=out_names,dtype=dtypes)
    for name in out_names :
        if name != root+'SOURCE_ID' : newcols[name] = np.nan

    # rename targetting proper motions to avoid confusion!
    try: tab.rename_column('PMRA','TARG_PMRA')
    except: pass
    try: tab.rename_column('PMDEC','TARG_PMDEC')
    except: pass
    try: tab.rename_column('PM_SRC','TARG_PM_SRC')
    except: pass
    # add unpopulated columns
    for col in newcols.columns.values() :
        try: tab.add_column(col)
        except ValueError: pass

    #if gaia_twomass is None or gaia_posn is None : return tab

    if gaia_twomass is not None :
        # remove dups in GAIA twomass in favor of brightest
        print('number in GAIA-2MASS xmatch catalog: ',len(gaia_twomass),len(set(gaia_twomass['original_ext_source_id'])))
        ind=[]
        for tid in set(gaia_twomass['original_ext_source_id']) :
            j=np.where(gaia_twomass['original_ext_source_id'] == tid)[0]
            if len(j)> 1:
                ii=np.argsort(gaia_twomass['phot_rp_mean_mag'][j])
                ind.append(j[ii[0]])
                print('duplicate 2MASS: ',gaia_twomass['phot_rp_mean_mag'][j[ii]])
            else : ind.append(j)

        # read gaia 2MASS matched file, match by 2MASS ID, and populate
        while len(gaia_twomass)>0 :
            # loop for matches since we may have repeats and want them all matched
            j=np.where(tab[root+'SOURCE_ID'] == 0)[0]
            print('Number missing gaia_source_id: ', len(j))
            if len(j) == 0 : break
            if type(tab['APOGEE_ID'][0]) is np.str_ : 
                m1,m2=match.match(np.core.defchararray.replace(tab['APOGEE_ID'][j],'2M',''),gaia_twomass['original_ext_source_id'])
            else :
                m1,m2=match.match(np.core.defchararray.replace(tab['APOGEE_ID'][j],b'2M',b''),gaia_twomass['original_ext_source_id'])
            print('Number matched by 2MASS: ', len(m1))
            if len(m1) == 0 : break
            for inname,outname in zip(in_names,out_names) :
                tab[outname][j[m1]] = gaia_twomass[inname][m2]

        j=np.where(tab[root+'SOURCE_ID'] > 0)[0]
        print('number of unique APOGEE_ID matches: ',len(set(tab['APOGEE_ID'][j])))

    j=np.where(tab[root+'SOURCE_ID'] == 0)[0]
    print('missing sources after 2MASS matches: ',len(j))

    # now do a positional match, take the brightest object within 3 arcsec
    h=htm.HTM()
    maxrad=3./3600.
    #m1,oldm2,rad=h.match(tab['RA'][j],tab['DEC'][j],gaia_posn['ra'],gaia_posn['dec'],maxrad,maxmatch=10)
    #for m in set(m1) :
    #    jj=np.where(m1 == m)[0]
    #    ii=np.argsort(gaia_posn['phot_rp_mean_mag'][m2[jj]])
    #    for inname,outname in zip(in_names,out_names) :
    #        tab[outname][j[m]] = gaia_posn[inname][m2[jj[ii[0]]]]
    
    # now do a positional match, take the brightest object within 3 arcsec, accounting from proper motion (query was within 10")
    j=np.where(tab[root+'SOURCE_ID'] == 0)[0]
    tmass_epoch=1999.
    try: ref_epoch = gaia_posn['ref_epoch']
    except KeyError : ref_epoch = 2016.
    m1,m2,rad=h.match(tab['RA'][j],tab['DEC'][j],
                      gaia_posn['ra']+gaia_posn['pmra']/3600e3*(tmass_epoch-ref_epoch)/np.cos(gaia_posn['dec']*np.pi/180.),
                      gaia_posn['dec']+gaia_posn['pmdec']/3600e3*(tmass_epoch-ref_epoch),
                      maxrad,maxmatch=10)
    for m in set(m1) :
        jj=np.where(m1 == m)[0]
        ii=np.argsort(gaia_posn['phot_rp_mean_mag'][m2[jj]])
        #if gaia_posn['source_id'][m2[jj[ii[0]]]] != tab['GAIA_SOURCE_ID'][j[m]] : 
        #    print('different gaia match')
        #    print(tab['GAIA_PMRA'][j[m]],tab['H'][j[m]])
        #    pdb.set_trace()
        #    print(gaia_posn['phot_rp_mean_mag'][m2[jj[ii[0]]]])
        #    k=np.where(gaia_posn['source_id'] == tab['GAIA_SOURCE_ID'][j[m]])[0]
        #    print(gaia_posn['phot_rp_mean_mag'][k[0]])
        #else :
        #    print(gaia_posn['source_id'][m2[jj[ii[0]]]],tab['GAIA_SOURCE_ID'][j[m]])
        #    print(gaia_posn['phot_rp_mean_mag'][m2[jj[ii[0]]]],tab['J'][j[m]])
        for inname,outname in zip(in_names,out_names) :
            try : tab[outname][j[m]] = gaia_posn[inname][m2[jj[ii[0]]]]
            except KeyError : pass
    
    j=np.where(tab[root+'SOURCE_ID'] == 0)[0]
    print('missing sources after second match: ',len(j))

    # replace NaNs
    #for name in out_names :
    #    bd = np.where(np.isnan(tab[name]))[0]
    #    tab[name][bd] = -9999.

    return tab

