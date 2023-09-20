import copy
import numpy as np
import os
import multiprocessing as mp
import pdb
import yaml
import matplotlib.pyplot as plt
from collections.abc import Iterable
from pyvista import imred, image, spectra, tv
from holtztools import plots, html
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from pyvista.dataclass import Data
from pyvista import bitmask
import scipy.signal
from scipy.ndimage import median_filter

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

    pixmask=bitmask.PixelBitMask()

    if type(groups) is not list : groups = [groups]

    # loop over multiple groups in input file
    for group in d['groups'] :

        if 'skip' in group:
            if group['skip']  : continue
        if groups[0] != 'all' and group['name'] not in groups  : continue

        # clear displays if given
        if display is not None : display.clear()

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
                            try : file = wavecal['file']
                            except KeyError : file = None
                            # extract and ID lines
                            if wavecal['wavecal_type'] == 'echelle' :
                                shift=wtrace.find(arc,plot=display) 
                                arcec=wtrace.extract(arc,plot=display)
                                wcal.identify(spectrum=arcec, rad=3, 
                                          display=display, plot=plot,file=file)

                            elif wavecal['wavecal_type'] == 'longslit' :
                                # 1d for inspection
                                wtrace.pix0 +=30
                                arcec=wtrace.extract(arc,plot=display,rad=20)
                                arcec.data = arcec.data - \
                                   scipy.signal.medfilt(arcec.data,kernel_size=[1,101])
                                wcal.identify(spectrum=arcec, rad=3, plot=plot,
                                              display=display,
                                              lags=range(-500,500),file=file)
                                # full 2D wavecal
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
                                              nskip=20,lags=range(-50,50))
   
                            if plot :
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

                    # basic reduction of frames
                    output = reduce_frames(obj,red,sbias,sdark,sflat,threads=threads,solve=solve,reddir=reddir)

                    for iframe,(id,frames) in enumerate(zip(obj['frames'],output)) : 
                        if display is not None : display.tvclear()
                        name=frames[0].header['FILE']
                        if html is not None : 
                            name=name.replace('.fits','') 
                            fhtml.write('<TR><TD>{:s}'.format(name))
                            for frame in frames :
                                fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                '</a>\n').format(name,name))
                    if html is not None : fhtml.write('</TABLE>')
                if html is not None : html.tail(fhtml) 
                plt.ion()

            elif 'echelle' in objects :
                # multi-order 
                for obj in objects['echelle'] :
                    if html is not None : 
                        try : fhtml.write('<BR><h3>{:s}</h3>\n'.format(obj['id']))
                        except KeyError : pass
                        fhtml.write('<br><TABLE BORDER=2>\n')

                    # reduction of frames and 1D flat if requested
                    if obj['flat_type'] == '1d' :
                        tmp = copy.copy(obj['flat'] )
                        obj['flat']  = 'none'
                        output = reduce_frames(obj,red,sbias,sdark,sflat,threads=threads,solve=solve,reddir=reddir)
                        obj['flat'] = tmp
                        try : superflat = sflat[obj['flat']]
                        except KeyError: 
                            raise ValueError('cannot make 1D flat without superflat specified!')
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
                    else :
                        output = reduce_frames(obj,red,sbias,sdark,None,threads=threads,solve=solve,reddir=reddir)

                    # extraction radius
                    try : rad = obj['rad'] 
                    except KeyError : rad = None

                    # retrace?
                    try : retrace = obj['retrace'] 
                    except KeyError : retrace = True

                    # wavelength solution to use
                    waves_all = wavedict[obj['wavecal']]

                    for iframe,(id,frames) in enumerate(zip(obj['frames'],output)) : 
                        if display is not None : display.clear() 
                        print("extracting object {}".format(id))

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
                                objec=tmptrace.extract(frame,rad=rad,
                                      threads=threads, plot=display)
                                w=wcal.wave(image=np.array(objec.data.shape))
                                if obj['flat_type'] == '1d' : 
                                    objecraw = copy.deepcopy(objec)
                                    objec.data/=ecflat[ichannel][iwind].data
                                    objec.uncertainty.array/= \
                                       ecflat[ichannel][iwind].uncertainty.array

                                if plot :
                                    gd=np.where((objec.bitmask & pixmask.badval()) == 0)[0]
                                    med=np.median(objec.data[gd[0],gd[1]])
                                    max=np.max([max,
                                         scipy.signal.medfilt(objec.data,[1,101]).max()])
                                    for row in range(objec.data.shape[0]) :
                                        gd=np.where((objec.bitmask[row] & pixmask.badval()) == 0)[0]
                                        plots.plotl(ax[0],w[row,gd],
                                                    objec.data[row,gd],
                                                    yr=[0,1.2*max],
                                                    xt='Wavelength',yt='Flux')
                                        plots.plotl(ax[1],w[row,gd],
                                                    objec.data[row,gd]/objec.uncertainty.array[row,gd],
                                                    xt='Wavelength',yt='S/N')
                                    plot.suptitle(objec.header['OBJNAME'])
                                    plt.draw()

                            # write individual orders/raw wavelength in any case
                            objec.add_wave(w)
                            objec.write(reddir+objec.header['FILE'].replace(
                                     '.fits','.ec.fits'),png=True)

                            # resample/combine orders if requested
                            if 'wresample' in obj :
                                print('resampling/combining')
                                wresample = np.array(obj['wresample'])
                                if len(wresample) == 1 :
                                    wnew = 10**np.linspace(np.log10(w.min()),np.log10(w.max()),wresample[0])
                                else :
                                    wnew = 10.**np.arange(*wresample)
                                if obj['flat_type'] == '1d' : 
                                    flatcomb = wcal.scomb(ecflat[ichannel][0],wnew,average=False,usemask=True)
                                    if len(ecflat[ichannel][0].data) > 1 :
                                        #remove large scale flat field intensity varation
                                        ncol = ecflat[ichannel][0].shape[1]
                                        flatcomb.data /= median_filter(flatcomb.data,[3*ncol])
                                    comb = wcal.scomb(objecraw,wnew,average=False,usemask=True)
                                    comb.data /= flatcomb.data
                                    comb.uncertainty.array = \
                                         np.sqrt(comb.uncertainty.array**2+flatcomb.uncertainty.array**2)
                                    comb.bitmask &= flatcomb.bitmask
                                    comb.write(reddir+objec.header['FILE'].replace(
                                             '.fits','.comb.fits'),png=True)
                                else :
                                    comb = wcal.scomb(objec,wnew,average=False,usemask=True)
                                    comb.write(reddir+objec.header['FILE'].replace(
                                             '.fits','.comb.fits'),png=True)
                                if plot :
                                    plots.plotl(ax[0],wnew,comb.data,color='k')
                                    plots.plotl(ax[1],wnew,
                                           comb.data/comb.uncertainty.array,
                                           color='k')
                                    plt.draw()
                                    plot.canvas.draw_idle()
                                    plt.pause(0.1)
                                    input("  hit a key to continue")

                        if html is not None : 
                            for frame in frames :
                                name=frames[0].header['FILE'].replace('.fits','')
                                fhtml.write('<TR><TD>{:s}'.format(name))
                                fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                             '</a>\n').format(name,name))
                                fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                             '</a>\n').format(name+'.ec',name+'.ec'))
                                if 'wresample' in obj :
                                    fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                                 '</a>\n').format(name+'.comb',name+'.comb'))
                    if html is not None : fhtml.write('</TABLE>')
                if html is not None : html.tail(fhtml) 

            elif 'longslit' in objects :
                # 1D spectra
                for obj in objects['longslit'] :

                    if html is not None : 
                        try : fhtml.write('<BR><h3>{:s}</h3>\n'.format(obj['id']))
                        except KeyError : pass
                        fhtml.write('<br><TABLE BORDER=2>\n')

                    # basic reduction of frames
                    output = reduce_frames(obj,red,sbias,sdark,sflat,threads=threads,solve=solve,reddir=reddir)

                    # loop through frames
                    for iframe,(id,frames) in enumerate(zip(obj['frames'],output)) : 
                        if display is not None : display.clear() 
                        print("extracting object {}".format(id))
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
                        if plot :
                            fig,ax=plots.multi(1,2,sharex=True,hspace=0.001)

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
                                obj2d=tmptrace.extract2d(frame,display=display)
                                w=wcal.wave(image=np.array(obj2d.data.shape))

                                #if plot :
                                #    gd=np.where((obj2d.bitmask & pixmask.badval()) == 0)[0]
                                #    med=np.median(obj2d.data[gd[0],gd[1]])
                                #    max=np.max([max,
                                #         scipy.signal.medfilt(obj2d.data,[1,101]).max()])
                                #    for row in range(obj2d.data.shape[0]) :
                                #        gd=np.where((obj2d.bitmask[row] & pixmask.badval()) == 0)[0]
                                #        plots.plotl(ax[0],w[row,gd],
                                #                    obj2d.data[row,gd],
                                #                    yr=[0,1.2*max],
                                #                    xt='Wavelength',yt='Flux')
                                #        plots.plotl(ax[1],w[row,gd],
                                #                    obj2d.data[row,gd]/obj2d.uncertainty.array[row,gd],
                                #                    xt='Wavelength',yt='S/N')
                                #    plot.suptitle(obj2d.header['OBJNAME'])
                                #    plt.draw()

                            # write raw wavelength image in any case
                            obj2d.add_wave(w)
                            obj2d.write(reddir+obj2d.header['FILE'].replace(
                                     '.fits','.2d.fits'),png=True,imshow=True)

                            # resample if requested
                            if 'wresample' in obj :
                                print('resampling')
                                wresample = np.array(obj['wresample'])
                                if len(wresample) == 1 :
                                    wnew = 10**np.linspace(np.log10(w.min()),np.log10(w.max()),wresample[0])
                                else :
                                    wnew = 10.**np.arange(*wresample)
                                resamp = wcal.correct(obj2d,wnew)
                                resamp.write(reddir+obj2d.header['FILE'].replace(
                                             '.fits','.resamp.fits'),png=True,imshow=True)

                        if html is not None : 
                            for frame in frames :
                                name=frames[0].header['FILE'].replace('.fits','')
                                fhtml.write('<TR><TD>{:s}'.format(name))
                                fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                             '</a>\n').format(name,name))
                                fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                             '</a>\n').format(name+'.2d',name+'.2d'))
                                if 'wresample' in obj :
                                    fhtml.write(('<TD><a href={:s}.png><IMG src={:s}.png width=500>'+
                                                 '</a>\n').format(name+'.resamp',name+'.resamp'))
                    if html is not None : fhtml.write('</TABLE>')
                if html is not None : html.tail(fhtml) 

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
    """ Process a single frame
    """
    red,id,superbias,superdark,superflat,scat,crbox,solve,reddir = pars
    frames= red.reduce(id, bias=superbias,
                          dark=superdark,
                          flat=superflat,
                          scat=red.scat,
                          crbox=red.crbox,
                          solve=solve,
                          return_list=True,display=None)
    if reddir is not None :
        name=frames[0].header['FILE']
        red.write(frames,reddir+name,overwrite=True,png=True)
    return frames

def reduce_frames(obj,red,sbias,sdark,sflat,threads=0,solve=False,reddir=None) :
    """ Reduce a set of frames, in parallel if threads>0

    """
    try : superbias = sbias[obj['bias']]
    except KeyError: superbias = None
    try : superdark = sdark[obj['dark']]
    except KeyError: superdark = None
    try : superflat = sflat[obj['flat']]
    except KeyError: superflat = None

    pars=[]
    for id in obj['frames'] :
        pars.append((red,id,superbias,superdark,
                    superflat,red.scat,red.crbox,
                    solve,reddir))

    if threads > 0  :
        # if multiprocessing, do all frames in this object 
        #    in parallel
        pool = mp.Pool(threads)
        output = pool.map_async(process_thread, pars).get()
        pool.close()
        pool.join()
    else :
        output=[]
        for par in pars :
            output.append(process_thread(par))

    return output

