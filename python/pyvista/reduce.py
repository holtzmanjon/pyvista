import copy
import numpy as np
import os
import pickle
import pdb
import yaml
import matplotlib.pyplot as plt
from pyvista import imred
from pyvista import image
from pyvista import spectra
from pyvista import tv
from tools import plots
from astropy import units
from astropy.nddata import CCDData, StdDevUncertainty
import scipy.signal

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/../../'

def all(ymlfile,display=None,plot=None,verbose=True,clobber=True,wclobber=None,groups='all') :
    """ Reduce full night(s) of data given input configuration file
    """

    # read input configuration file for reductions
    f=open(ymlfile,'r')
    d=yaml.load(f, Loader=yaml.FullLoader)
    f.close()

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
        print('Instrument: {:s}'.format(inst))
        try : red = imred.Reducer(inst=group['inst'],dir=group['rawdir'],verbose=verbose,nfowler=group['nfowler'])
        except KeyError : red = imred.Reducer(inst=group['inst'],dir=group['rawdir'],verbose=verbose)
        comb = imred.Combiner(reducer=red,verbose=verbose)
        reddir = group['reddir']+'/'
        try: os.makedirs(reddir)
        except FileExistsError : pass

        #create superbiases if biases given
        if 'biases' in group : 
            sbias = mkcal(group['biases'],'bias',comb,reddir,clobber=clobber,display=display)
        else: 
            print('no bias frames given')
            sbias = None

        #create superdarks if darks given
        if 'darks' in group : 
            sdark = mkcal(group['darks'],'dark',comb,reddir,clobber=clobber,display=display,sbias=sbias)
        else: 
            print('no dark frames given')
            sdark = None

        #create superflats if darks given
        if 'flats' in group : 
            sflat = mkcal(group['flats'],'flat',comb,reddir,clobber=clobber,display=display,sbias=sbias,sdark=sdark)
        else: 
            print('no flat frames given')
            sflat = None

        # create wavecals if arcs given
        if 'arcs' in group :
            # existing trace template
            traces=pickle.load(open(ROOT+'/data/'+inst+'/'+inst+'_traces.pkl','rb'))

            if wclobber is None : wclobber = clobber
            wavedict={}
            wavecals=group['arcs']
            for wavecal in wavecals :
                print('create wavecal : {:s}'.format(wavecal['id']))
                # existing wavecal template
                waves=pickle.load(open(ROOT+'/data/'+inst+'/'+wavecal['wref']+'.pkl','rb'))
                if wclobber :
                    make = True
                else :
                    make = False
                    try: waves_all = pickle.load(open(reddir+wavecal['id']+'.pkl','rb'))
                    except FileNotFoundError : make=True
                if make :
                    # combine frames
                    try : superbias = sbias[obj['bias']]
                    except KeyError: superbias = None
                    arcs=comb.sum(wavecal['frames'],return_list=True, superbias=superbias, crbox=[5,1], display=display)

                    print('extract wavecal')
                    # loop over channels
                    waves_all=[]
                    for arc,wave,trace in zip(arcs,waves,traces) :
                 
                        # loop over windows
                        waves_channel=[]
                        for iwind,(wcal,wtrace) in enumerate(zip(wave,trace)) :
                            # extract
                            if wavecal['wavecal_type'] == 'echelle' :
                                arcec=wtrace.extract(arc,plot=display)
                                wcal.identify(spectrum=arcec, rad=3, display=display, plot=plot)
                            elif wavecal['wavecal_type'] == 'longslit' :
                                r0=wtrace.rows[0]
                                r1=wtrace.rows[1]
                                # 1d for inspection
                                wtrace.pix0 +=30
                                arcec=wtrace.extract(arc,plot=display,rad=20)
                                arcec.data=arcec.data - scipy.signal.medfilt(arcec.data,kernel_size=[1,101])
                                wcal.identify(spectrum=arcec, rad=3, plot=plot, display=display)
                                wcal.fit()

                                print("doing 2D wavecal...")
                                arcec=wtrace.extract2d(arc)
                                print(" remove continuum and smooth in rows")
                                arcec.data=arcec.data - scipy.signal.medfilt(arcec.data,kernel_size=[1,101])
                                # smooth vertically for better S/N, then sample accordingly
                                image.smooth(arcec,[5,1])
                                wcal.identify(spectrum=arcec, rad=3, display=display, plot=plot, nskip=5)
    
                            wcal.fit()
                            w=wcal.wave(image=np.array(arcec.data.shape))
                            waves_channel.append(w)
                        waves_all.append(waves_channel)
                    pickle.dump(waves_all,open(reddir+wavecal['id']+'.pkl','wb'))
                    if display is not None : display.clear()
                    if plot is not None : plot.clf()
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
                    try : superbias = sbias[obj['bias']]
                    except KeyError: superbias = None
                    try : superdark = sdark[obj['dark']]
                    except KeyError: superdark = None
                    try : superflat = sflat[obj['flat']]
                    except KeyError: superflat = None
                    pdb.set_trace()
                    for iframe,id in enumerate(obj['frames']) : 
                        frames=red.reduce(id,superbias=superbias,superdark=superdark,superflat=superflat,scat=red.scat,
                                          return_list=True,crbox=red.crbox,display=display) 
            elif 'extract1d' in objects :
                # 1D spectra
                traces=pickle.load(open(ROOT+'/data/'+inst+'/'+inst+'_traces.pkl','rb'))
                for obj in objects['extract1d'] :
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
                        for trace in traces :
                            tmp=[]
                            for wtrace in trace :
                                shift=wtrace.find(superflat,plot=display) 
                                tmp.append(wtrace.extract(superflat,plot=display))
                            ecflat.append(tmp)
                        superflat = None
                    # now frames
                    for iframe,id in enumerate(obj['frames']) : 
                        if display is not None : display.clear() 
                        print("extracting object {}".format(id))
                        frames=red.reduce(id,superbias=superbias,superdark=superdark,superflat=superflat,scat=red.scat,
                                          return_list=True,crbox=red.crbox,display=display) 
                        if 'skyframes' in obj :
                            id = obj['skyframes'][iframe]
                            skyframes=red.reduce(id,superbias=superbias,superdark=superdark,superflat=superflat,scat=red.scat,
                                              return_list=True,crbox=red.crbox,display=display) 
                            for iframe,(frame,skyframe) in enumerate(zip(frames,skyframes)) : 
                                header = frame.header
                                frames[iframe]= frame.subtract(skyframe)
                                frames[iframe].header = header

                        # extraction radius
                        try : rad = obj['rad'] 
                        except KeyError : rad = None

                        # retrace?
                        try : retrace = obj['retrace'] 
                        except KeyError : retrace = False

                        # initialize plots
                        if plot is not None : 
                            plot.clf()
                            ax=[]
                            ax.append(plot.add_subplot(2,1,1))
                            ax.append(plot.add_subplot(2,1,2,sharex=ax[0]))
                            plot.subplots_adjust(hspace=0.001)

                        # loop over channels
                        max=0
                        for ichannel,(frame,wave,trace) in enumerate(zip(frames,waves_all,traces)) :
                            # loop over windows
                            for iwind,(wcal,wtrace) in enumerate(zip(wave,trace)) :
                                if retrace : shift=wtrace.retrace(frame,plot=display) 
                                else : shift=wtrace.find(frame,plot=display) 
                                ec=wtrace.extract(frame,plot=display,rad=rad)
                                if obj['flat_type'] == '1d' : 
                                    header=ec.header
                                    ec=ec.divide(ecflat[ichannel][iwind])
                                    ec.header=header
                                if plot is not None :
                                    gd=np.where(ec.mask == False) 
                                    med=np.median(ec.data[gd[0],gd[1]])
                                    max=np.max([max,scipy.signal.medfilt(ec.data,[1,101]).max()])
                                    for row in range(ec.data.shape[0]) :
                                        gd=np.where(ec.mask[row,:] == False)[0]
                                        plots.plotl(ax[0],wcal[row,gd],ec.data[row,gd],yr=[0,1.2*max],xt='Wavelength',yt='Flux')
                                        plots.plotl(ax[1],wcal[row,gd],ec.data[row,gd]/ec.uncertainty.array[row,gd],xt='Wavelength',yt='S/N')
                                    plot.suptitle(ec.header['OBJNAME'])
                                    plt.draw()
                                    plot.canvas.draw_idle()
                                    plt.pause(0.1)
                                    input("  hit a key to continue")
                            #ec.write(reddir+ec.header['FILE'].replace('.fits','.ec.fits'),overwrite=True)
                            out = spectra.SpecData(ec,wave=wave)
                            out.write(reddir+ec.header['FILE'].replace('.fits','.ec.fits'))
                        pdb.set_trace()
            elif 'extract2d' in objects :
                # 2D spectra
                print('extract2d')

        """      
        # 1d extractions
        # flat field for 1D first
        if group['config']['flat_type'] == '1d' :
            print("extracting 1D flat")
            ecflat=[]
            for trace in traces :
                tmp=[]
                for wtrace in trace :
                    shift=wtrace.find(sflat,plot=display) 
                    tmp.append(wtrace.extract(sflat,plot=display))
                ecflat.append(tmp)
            sflat2d=None
        elif group['config']['flat_type'] == '2d' :
            sflat2d=sflat
        else :
            sflat2d=None

        # now frames
        for id in group['objects']['extract1d']['frames'] : 
            if display is not None : display.clear() 
            print("extracting object {}",id)
            frames=red.reduce(id,superbias=sbias,superdark=sdark,superflat=sflat2d,scat=red.scat,
                              return_list=True,crbox=red.crbox,display=display) 
            try : rad = group['objects']['extract1d']['rad'] 
            except KeyError : rad = None
            if plot is not None : 
                plot.clf()
                ax=[]
                ax.append(plot.add_subplot(2,1,1))
                ax.append(plot.add_subplot(2,1,2,sharex=ax[0]))
            max=0
            for ichannel,(frame,wave,trace) in enumerate(zip(frames,waves_all,traces)) :
                for iwind,(wcal,wtrace) in enumerate(zip(wave,trace)) :
                    shift=wtrace.find(frame,plot=display) 
                    ec=wtrace.extract(frame,plot=display,rad=rad)
                    if group['config']['flat_type'] == '1d' : 
                        header=ec.header
                        ec=ec.divide(ecflat[ichannel][iwind])
                        ec.header=header
                    if plot is not None :
                        plot.subplots_adjust(hspace=0.001)
                        gd=np.where(ec.mask == False) 
                        med=np.median(ec.data[gd[0],gd[1]])
                        max=np.max([max,scipy.signal.medfilt(ec.data,[1,101]).max()])
                        for row in range(ec.data.shape[0]) :
                            gd=np.where(ec.mask[row,:] == False)[0]
                            plots.plotl(ax[0],wcal[row,gd],ec.data[row,gd],yr=[0,1.2*max],xt='Wavelength',yt='Flux')
                            plots.plotl(ax[1],wcal[row,gd],ec.data[row,gd]/ec.uncertainty.array[row,gd],xt='Wavelength',yt='S/N')
                        plot.suptitle(ec.header['OBJNAME'])
                        plt.draw()
                        plot.canvas.draw_idle()
                        plt.pause(0.1)
                        input("  hit a key to continue")
                ec.write(reddir+ec.header['FILE'].replace('.fits','.ec.fits'),overwrite=True)
            pdb.set_trace()
        """
    # 2d extractions
#    traces=pickle.load(open(group['inst']+'_trace.pkl','rb'))
#    if type(traces) is not list : traces = [traces]
#    pdb.set_trace()
#    for id in group['objects']['extract2d']['frames'] : 
#        frames=red.reduce(id,superbias=sbias,superdark=sdark,superflat=sflat,scat=red.scat,return_list=True) 
#        for frame,trace in zip(frames,traces) :
#            out=trace.extract2d(frame,plot=t)


def mkcal(cals,caltype,comb,reddir,sbias=None,sdark=None,clobber=False,**kwargs) :
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
                if len(comb.reducer.channels)==1 :
                    try : scal= CCDData.read(reddir+calname+'.fits')
                    except FileNotFoundError : make=True
                else :
                    scal=[]
                    for channel in comb.reducer.channels :
                        try : scal.append(CCDData.read(reddir+calname+'_'+channel+'.fits'))
                        except FileNotFoundError : make=True
            if make :
                try :
                    # see if we are requested to make product from previous products by seeing if dictionary entries exist for frames
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
                    if len(scal) is 1 : scal= scal[0]
                except :
                    # make calibration product from raw data frames
                    if caltype == 'bias' :
                        scal = comb.superbias(cal['frames'],**kwargs)
                    elif caltype == 'dark' :
                        scal = comb.superdark(cal['frames'],superbias=superbias,**kwargs)
                    elif caltype == 'flat' :
                        scal = comb.superflat(cal['frames'],superbias=superbias,superdark=superdark,**kwargs)
                        try: 
                            if cal['specflat'] : scal = comb.specflat(scal)
                        except: pass
                        comb.reducer.scatter(scal,scat=comb.reducer.scat,**kwargs)
                comb.reducer.write(scal,reddir+calname,overwrite=True)

#            if make :
#                scal=[]
#                tot=[]
#                for cal in cals :
#                    print(' '+cal['id'])
#                    if os.path.exists(reddir+calname+'.fits') and not clobber :
#                        out=CCDData.read(reddir++calname+'.fits')
#                    else :
#                    if cal['use'] :
#                        # we may have multiple channels to combine
#                        if type(out) is not list : out=[out]
#                        for i in range(len(out)) :
#                            try:
#                                scal[i] = scal[i].add(out[i].multiply(out[i].header['MEANNORM']))
#                                tot[i]+=out[i].header['MEANNORM']
#                            except:
#                                scal.append( copy.deepcopy(out[i].multiply(out[i].header['MEANNORM'])) )
#                                tot.append(out[i].header['MEANNORM'])
#                if darkflat is not None :
#                    if type(darkflat) is not list : darkflat=[darkflat]
#                    for i in range(len(scal)) : scal[i]=scal[i].subtract(darkflat[i])
#                for i in range(len(scal)) : scal[i] = scal[i].divide(tot[i])
#                if len(scal) is 1 : scal= scal[0]
#                comb.reducer.write(scal,reddir+calname,overwrite=True)
            else : print('  already made!')
        except RuntimeError :
            print('error processing {:s} frames'.format(caltype))
        except:
            print('no {:s} frames given'.format(caltype))
            scal=None

        outcal[calname] = scal

    return outcal

