import copy
import numpy as np
import os
import multiprocessing as mp
import pdb
import yaml
import matplotlib.pyplot as plt
from collections.abc import Iterable
from pyvista import imred, image, spectra, tv
from tools import plots, html
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from pyvista.dataclass import Data
import scipy.signal

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/../../'

def all(ymlfile,display=None,plot=None,verbose=True,clobber=True,wclobber=None,
        groups='all',solve=False,htmlfile='index.html',threads=0) :
    """ Reduce full night(s) of data given input configuration file
    """

    # read input configuration file for reductions
    f=open(ymlfile,'r')
    d=yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    if display is None : plt.ioff()
    if display and threads > 0 :
        raise ValueError('no multiprocessing with display!')

    if type(groups) is not list : groups = [groups]

    # loop over multiple groups in input file
    for group in d['groups'] :

        if 'skip' in group:
            if group['skip']  : continue
        if groups[0] != 'all' and group['name'] not in groups  : continue

        # clear displays if given
        if display is not None : display.clear()
        if plot is not None : plot.clf()

        # set up Reducer, Combiner, and output directory
        inst = group['inst']
        try : conf = group['conf']
        except : conf=''
        print('Instrument: {:s}'.format(inst))
        try : red = imred.Reducer(inst=inst,conf=conf,dir=group['rawdir'],
                                  verbose=verbose,nfowler=group['nfowler'])
        except KeyError : red = imred.Reducer(inst=group['inst'],conf=conf,
                                  dir=group['rawdir'],verbose=verbose)
        reddir = group['reddir']+'/'
        try: os.makedirs(reddir)
        except FileExistsError : pass
        if htmlfile is not None : fhtml = html.head(reddir+'/'+htmlfile)
        else : fhtml = None

        #create superbiases if biases given
        if 'biases' in group : 
            sbias = mkcal(group['biases'],'bias',red,reddir,
                          clobber=clobber,display=display,html=fhtml)
        else: 
            print('no bias frames given')
            sbias = None

        #create superdarks if darks given
        if 'darks' in group : 
            sdark = mkcal(group['darks'],'dark',red,reddir,
                          clobber=clobber,display=display,sbias=sbias,
                          html=fhtml)
        else: 
            print('no dark frames given')
            sdark = None

        #create superflats if darks given
        if 'flats' in group : 
            sflat = mkcal(group['flats'],'flat',red,reddir,
                          clobber=clobber,display=display,
                          sbias=sbias,sdark=sdark,html=fhtml)
        else: 
            print('no flat frames given')
            sflat = None

        # create wavecals if arcs given
        nwind = 1
        if 'arcs' in group :

            if wclobber is None : wclobber = clobber
            wavedict={}
            wavecals=group['arcs']
            for wavecal in wavecals :
                print('create wavecal : {:s}'.format(wavecal['id']))
                # existing wavecal template
                waves=spectra.WaveCal(inst+'/'+wavecal['wref']+'.fits')
                if wclobber :
                    make = True
                else :
                    make = False
                    try: 
                        traces_all=[]
                        waves_all=[]
                        for ichan,arc in enumerate(red.channels) :
                            traces_channel=[]
                            waves_channel=[]
                            for iwind in range(nwind) :
                                traces_channel.append(spectra.Trace(inst+'/'+
                                  group['traces']['traceref']+'.fits',
                                  hdu=iwind+1))
                                waves_channel.append(
                                  spectra.WaveCal(reddir+wavecal['id']+'.fits',
                                  hdu=iwind+1))
                            traces_all.append(traces_channel)
                            waves_all.append(waves_channel)
                    except FileNotFoundError : make=True

                if make :
                    # combine frames
                    try : superbias = sbias[wavecal['bias']]
                    except KeyError: superbias = None
                    arcs=red.sum(wavecal['frames'],return_list=True, 
                                 bias=superbias, crbox=[5,1], 
                                 display=display)

                    print('  extract wavecal')
                    # loop over channels
                    traces_all=[]
                    waves_all=[]
                    for ichan,arc in enumerate(arcs) :
                        # existing trace template
                        trace = spectra.Trace(inst+'/'+
                                  group['traces']['traceref']+'.fits')
                        wave=spectra.WaveCal(inst+'/'+wavecal['wref']+'.fits')
                 
                        # loop over windows -- not yet implemented!
                        traces_channel=[]
                        waves_channel=[]
                        for iwind in range(nwind) :
                            wtrace = trace
                            wcal = wave
                            try:
                                delattr(wcal,'ax')
                                delattr(wcal,'fig')
                            except: pass
                            try : file = wavecal['file']
                            except KeyError : file = None
                            # extract and ID lines
                            if wavecal['wavecal_type'] == 'echelle' :
                                shift=wtrace.find(arc,plot=display) 
                                arcec=wtrace.extract(arc,plot=display)
                                wcal.identify(spectrum=arcec, rad=3, 
                                          display=display, plot=plot,file=file)
                            elif wavecal['wavecal_type'] == 'longslit' :
                                r0=wtrace.rows[0]
                                r1=wtrace.rows[1]
                                # 1d for inspection
                                wtrace.pix0 +=30
                                arcec=wtrace.extract(arc,plot=display,rad=20)
                                arcec.data = arcec.data - \
                                   scipy.signal.medfilt(arcec.data,kernel_size=[1,101])
                                wcal.identify(spectrum=arcec, rad=3, plot=plot,
                                              display=display,
                                              lags=range(-500,500),file=file)

                                print("doing 2D wavecal...")
                                arcec=wtrace.extract2d(arc)
                                print(" remove continuum and smooth in rows")
                                arcec.data=arcec.data - \
                                        scipy.signal.medfilt(arcec.data,
                                                         kernel_size=[1,101])
                                # smooth vertically for better S/N, then 
                                #    sample accordingly
                                image.smooth(arcec,[5,1])
                                wcal.identify(spectrum=arcec, rad=3, 
                                              display=display, plot=plot, 
                                              nskip=5,lags=range(-50,50))
   
                            if plot is not None :
                                delattr(wcal,'ax')
                                delattr(wcal,'fig')
                            if iwind == 0 : append = False
                            else : append = True
                            wcal.write(reddir+wavecal['id']+'.fits',append=append)
                            waves_channel.append(wcal)
                            traces_channel.append(trace)
                        traces_all.append(traces_channel)
                        waves_all.append(waves_channel)
                        if display is not None : display.clear()
                    #waves_all[0].write(reddir+wavecal['id']+'.fits')
                    if plot is not None : plot.clf()
                else :
                    print('  already made!')
                wavedict[wavecal['id']] = waves_all
        else :
            print('no wavecal frames given')
            w=None

        # reduce objects
        if 'objects' in group :
            objects = group['objects']
            if 'image' in objects :
                # images
                for obj in objects['image']:
                    if html is not None : 
                        try : fhtml.write('<BR><h3>{:s}</h3>\n'.format(obj['id']))
                        except KeyError : pass
                        fhtml.write('<br><TABLE BORDER=2>\n')
                    try : superbias = sbias[obj['bias']]
                    except KeyError: superbias = None
                    try : superdark = sdark[obj['dark']]
                    except KeyError: superdark = None
                    try : superflat = sflat[obj['flat']]
                    except KeyError: superflat = None

                    if threads > 0  :
                        # if multiprocessing, do all frames in this object 
                        #    in parallel
                        pars=[]
                        for id in obj['frames'] :
                            pars.append((red,id,superbias,superdark,
                                         superflat,red.scat,red.crbox,
                                         solve,reddir))
                        pool = mp.Pool(threads)
                        output = pool.map_async(process_thread, pars).get()
                        pool.close()
                        pool.join()
                    else :
                        output = obj['frames']

                    for iframe,(id,frames) in enumerate(zip(obj['frames'],output)) : 
                        if display is not None : display.tvclear()
                        if threads == 0 :
                            # if not multiprocessing, then do the reduction
                            frames=red.reduce(id,bias=superbias,
                                             dark=superdark,
                                             flat=superflat,
                                             scat=red.scat,
                                             crbox=red.crbox,
                                             solve=solve,
                                             return_list=True,
                                             display=display)
                            name=frames[0].header['FILE']
                            red.write(frames,reddir+name,overwrite=True,png=True)
                        else : name=frames[0].header['FILE']
                        if html is not None : 
                            name=name.replace('.fits','') 
                            fhtml.write('<TR><TD>{:s}'.format(name))
                            for frame in frames :
                                fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                '</a>\n').format(name,name))
                    if html is not None : fhtml.write('</TABLE>')
                if html is not None : html.tail(fhtml) 
                plt.ion()
            elif 'extract1d' in objects :
                # 1D spectra
                for obj in objects['extract1d'] :
                    if html is not None : 
                        try : fhtml.write('<BR><h3>{:s}</h3>\n'.format(obj['id']))
                        except KeyError : pass
                        fhtml.write('<br><TABLE BORDER=2>\n')
                    try : superbias = sbias[obj['bias']]
                    except KeyError: superbias = None
                    try : superdark = sdark[obj['dark']]
                    except KeyError: superdark = None
                    try : superflat = sflat[obj['flat']]
                    except KeyError: superflat = None
                    waves_all = wavedict[obj['wavecal']]

                    if obj['flat_type'] == '1d' :
                        print('extracting 1d flat')
                        ecflat=[]
                        for trace in traces_all :
                            tmp=[]
                            for wtrace in trace :
                                shift=wtrace.find(
                                          red.trim(superflat,trimimage=True),
                                          plot=display) 
                                tmp.append(wtrace.extract(
                                          red.trim(superflat,trimimage=True),
                                          plot=display,threads=threads))
                            ecflat.append(tmp)
                        superflat = None

                    if threads > 0  :
                        # if multiprocessing, do all frames in this object 
                        #    in parallel
                        pars=[]
                        for id in obj['frames'] :
                            pars.append((red,id,superbias,superdark,
                                        superflat,red.scat,red.crbox,
                                        solve,reddir))
                        pool = mp.Pool(threads)
                        output = pool.map_async(process_thread, pars).get()
                        pool.close()
                        pool.join()
                    else :
                        output = obj['frames']

                    # now frames
                    for iframe,(id,frames) in enumerate(zip(obj['frames'],output)) : 
                        if display is not None : display.clear() 
                        print("extracting object {}".format(id))
                        if threads == 0 :
                            frames=red.reduce(id,bias=superbias,dark=superdark,
                                          flat=superflat,scat=red.scat,
                                          return_list=True,crbox=red.crbox,
                                          display=display) 
                        if 'skyframes' in obj :
                            id = obj['skyframes'][iframe]
                            skyframes=red.reduce(id,bias=superbias,
                                                 dark=superdark,
                                                 flat=superflat,scat=red.scat,
                                                 return_list=True,
                                                 crbox=red.crbox,
                                                 display=display) 
                            for iframe,(frame,skyframe) in enumerate(zip(frames,skyframes)) : 
                                header = frame.header
                                frames[iframe]= frame.subtract(skyframe)
                                frames[iframe].header = header

                        # extraction radius
                        try : rad = obj['rad'] 
                        except KeyError : rad = None

                        # retrace?
                        try : retrace = obj['retrace'] 
                        except KeyError : retrace = True

                        # initialize plots
                        if plot is not None : 
                            plot.clf()
                            ax=[]
                            ax.append(plot.add_subplot(2,1,1))
                            ax.append(plot.add_subplot(2,1,2,sharex=ax[0]))
                            plot.subplots_adjust(hspace=0.001)

                        # loop over channels
                        max=0
                        for ichannel,(frame,wave,trace) in enumerate(zip(frames,waves_all,traces_all)) :
                            # loop over windows
                            for iwind,(wcal,wtrace) in enumerate(zip(wave,trace)) :
                                tmptrace=copy.deepcopy(wtrace)
                                if retrace : 
                                    print('  retracing ....')
                                    shift=tmptrace.retrace(frame,
                                                       plot=display,thresh=10) 
                                else : 
                                    shift=tmptrace.find(frame,plot=display) 
                                ec=tmptrace.extract(frame,rad=rad,
                                      threads=threads, plot=display)
                                w=wcal.wave(image=np.array(ec.data.shape))
                                if obj['flat_type'] == '1d' : 
                                    ec.data/=ecflat[ichannel][iwind].data
                                    ec.uncertainty.array/= \
                                       ecflat[ichannel][iwind].uncertainty.array
                                if plot is not None :
                                    gd=np.where(ec.mask == False) 
                                    med=np.median(ec.data[gd[0],gd[1]])
                                    max=np.max([max,
                                         scipy.signal.medfilt(ec.data,[1,101]).max()])
                                    for row in range(ec.data.shape[0]) :
                                        gd=np.where(ec.mask[row,:] == False)[0]
                                        plots.plotl(ax[0],w[row,gd],
                                                    ec.data[row,gd],
                                                    yr=[0,1.2*max],
                                                    xt='Wavelength',yt='Flux')
                                        plots.plotl(ax[1],w[row,gd],
                                                    ec.data[row,gd]/ec.uncertainty.array[row,gd],
                                                    xt='Wavelength',yt='S/N')
                                    plot.suptitle(ec.header['OBJNAME'])
                                    plt.draw()
                            if plot is not None :
                                plots.plotl(ax[0],wnew,comb.data,color='k')
                                plots.plotl(ax[1],wnew,
                                       comb.data/comb.uncertainty.array,
                                       color='k')
                                plt.draw()
                                plot.canvas.draw_idle()
                                plt.pause(0.1)
                                input("  hit a key to continue")
                            ec.add_wave(w)
                            ec.write(reddir+ec.header['FILE'].replace(
                                     '.fits','.ec.fits'),png=True)
                        if html is not None : 
                            for frame in frames :
                                name=frames[0].header['FILE'].replace('.fits','')
                                fhtml.write('<TR><TD>{:s}'.format(name))
                                fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                             '</a>\n').format(name,name))
                                fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                             '</a>\n').format(name+'.ec',name+'.ec'))
                    if html is not None : fhtml.write('</TABLE>')
                if html is not None : html.tail(fhtml) 
            elif 'extract2d' in objects :
                # 2D spectra
                print('extract2d')
#    traces=pickle.load(open(group['inst']+'_trace.pkl','rb'))
#    if type(traces) is not list : traces = [traces]
#    pdb.set_trace()
#    for id in group['objects']['extract2d']['frames'] : 
#        frames=red.reduce(id,bias=sbias,dark=sdark,flat=sflat,scat=red.scat,return_list=True) 
#        for frame,trace in zip(frames,traces) :
#            if red.transpose : frame=red.imtranspose(frame)
#            out=trace.extract2d(frame,plot=t)


def mkcal(cals,caltype,reducer,reddir,sbias=None,sdark=None,clobber=False,
          html=None,**kwargs) :
    """ Make calibration frames given input lists
 
        Args :
            cals : list of different sets of given calibration type, as dictionaries
            caltype : gives caltype, of 'bias', 'dark', 'flat'
            reddir : directory for cal frames
            clobber= : set to True to force construction even if cal frame already exists
    """

    # we will loop over (possibly) multiple individual cal products of this type
    # These may or may not be combined, depending on "use" tag
    outcal={}
    if html is not None :
        html.write('<br><h3>{:s}</h3><br><TABLE BORDER=2>\n'.format(caltype))
    for cal in cals :
        calname = cal['id']
        try : superbias = sbias[cal['bias']]
        except KeyError: superbias = None
        try : superdark = sdark[cal['dark']]
        except KeyError: superdark = None
        try :
            print('create {:s} : {:s}'.format(caltype,calname))
            # if not clobber, try to read existing frames
            if clobber :
                make=True
            else :
                make=False
                if len(reducer.channels)==1 :
                    try : scal= Data.read(reddir+calname+'.fits',
                                             unit=u.dimensionless_unscaled)
                    except FileNotFoundError : make=True
                else :
                    scal=[]
                    for channel in reducer.channels :
                        try : scal.append(
                                Data.read(reddir+calname+'_'+channel+'.fits',
                                             unit=u.dimensionless_unscaled))
                        except FileNotFoundError : make=True
            if make :
                try :
                    # see if we are requested to make product from previous 
                    # products by seeing if dictionary entries exist for frames
                    scal=[]
                    tot=[]
                    for frame in cal['frames'] :
                        out = outcal[frame]
                        if type(out) is not list : out=[out]
                        for i in range(len(out)) :
                            print('combining: {:s}'.format(frame))
                            try:
                                scal[i] = scal[i].add(out[i].multiply(out[i].header['MEANNORM']))
                                tot[i]+=out[i].header['MEANNORM']
                            except:
                                scal.append( copy.deepcopy(out[i].multiply(out[i].header['MEANNORM'])) )
                                tot.append(out[i].header['MEANNORM'])
                    for i in range(len(scal)) : scal[i] = scal[i].divide(tot[i])
                    if len(scal) == 1 : scal= scal[0]
                except :
                    # make calibration product from raw data frames
                    if caltype == 'bias' :
                        scal = reducer.mkbias(cal['frames'],**kwargs)
                    elif caltype == 'dark' :
                        scal = reducer.mkdark(cal['frames'],bias=superbias,**kwargs)
                    elif caltype == 'flat' :
                        scal = reducer.mkflat(cal['frames'],bias=superbias,dark=superdark,**kwargs)
                        try: 
                            if cal['specflat'] : scal = reducer.mkspecflat(scal)
                        except: pass
                        reducer.scatter(scal,scat=reducer.scat,**kwargs)
                reducer.write(scal,reddir+calname+'.fits',overwrite=True,png=True)
                if html is not None :
                    html.write(
                      '<TR><TD>{:s}<TD><A HREF={:s}.png><IMG SRC={:s}.png WIDTH=500></A>\n'.
                      format(calname,calname,calname))
            else : print('  already made!')
        except RuntimeError :
            print('error processing {:s} frames'.format(caltype))
        except KeyError:
            print('no {:s} frames given'.format(caltype))
            scal=None

        outcal[calname] = scal

    if html is not None : html.write('</TABLE>\n')

    return outcal

def process_thread(pars) :

    red,id,superbias,superdark,superflat,scat,crbox,solve,reddir = pars
    frames= red.reduce(id, bias=superbias,
                          dark=superdark,
                          flat=superflat,
                          scat=red.scat,
                          crbox=red.crbox,
                          solve=solve,
                          return_list=True,display=None)
    name=frames[0].header['FILE']
    red.write(frames,reddir+name,overwrite=True,png=True)
    return frames

