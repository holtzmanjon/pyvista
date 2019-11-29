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

def all(ymlfile,display=None,plot=None,verbose=True,clobber=True) :
    """ Reduce full night(s) of data given input configuration file
    """
    f=open(ymlfile,'r')
    d=yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    # loop over multiple groups in input file
    for group in d['groups'] :

        try: 
            if group['skip']  : continue
        except : pass

        if display is not None : display.clear()
        if plot is not None : plot.clf()

        # set up Reducer, Combiner, and output directory
        inst = group['inst']
        print('Instrument: {:s}'.format(inst))
        try :red = imred.Reducer(inst=group['inst'],dir=group['rawdir'],verbose=verbose,nfowler=group['nfowler'])
        except: red = imred.Reducer(inst=group['inst'],dir=group['rawdir'],verbose=verbose)
        comb = imred.Combiner(reducer=red,verbose=verbose)
        reddir = group['reddir']+'/'
        try: os.makedirs(reddir)
        except: pass

        #create superbias if biases given
        try:
            frames=group['biases']
            print('create superbias')
            # if not clobber, try to read existing frames
            if clobber :
                make = True
            else :
                make=False
                if len(red.channels)==1 : 
                    try : sbias= CCDData.read(reddir+'sbias.fits')
                    except : make=True
                else :
                    sbias=[]
                    for channel in red.channels :
                        try : sbias.append(CCDData.read(reddir+'sbias_'+channel+'.fits'))
                        except : make=True
            if make :
                sbias=comb.superbias(group['biases'],scat=None,display=display)
                red.write(sbias,reddir+'sbias',overwrite=True)
            else : print('  already made!')
        except:
            print('no bias frames given')
            sbias=None

    #create superdark if darks given
        try:
            frames=group['darks']
            print('create superdark')
            # if not clobber, try to read existing frames
            if clobber :
                make = True
            else :
                make=False
                if len(red.channels)==1 :
                    try : sdark= CCDData.read(reddir+'sdark.fits')
                    except : make=True
                else :
                    sdark=[]
                    for channel in red.channels :
                        try : sdark.append(CCDData.read(reddir+'sdark_'+channel+'.fits'))
                        except : make=True
            if make :
                sdark=comb.superdark(group['darks'],scat=None,display=display)
                red.write(sdark,reddir+'sdark',overwrite=True)
            else : print('  already made!')
        except:
            print('no dark frames given')
            sdark=None

        #create dark for flat (used in IR) if darkflats given
        try:
            frames=group['darkflats']
            print('create darkflat')
            # if not clobber, try to read existing frames
            if clobber :
                make=True
            else :
                make=False
                if len(red.channels)==1 :
                    try : darkflat= CCDData.read(reddir+'darkflat.fits')
                    except : make=True
                else :
                    darkflat=[]
                    for channel in red.channels :
                        try : darkflat.append(CCDData.read(reddir+'darkflat_'+channel+'.fits'))
                        except : make=True
            if make :
                # use superbias to combine, same procedure as superdark
                darkflat=comb.superbias(group['darkflats'],scat=None,display=display)
                red.write(darkflat,reddir+'darkflat',overwrite=True)
            else : print('  already made!')
        except:
            print('no darkflat frames given')
            darkflat=None

        # create superflat. Here allow for multiple sets, which may or may not be combined
        #   depending on "use" tag
        try :
            flats=group['flats']
            print('create superflat')
            # if not clobber, try to read existing frames
            if clobber :
                make=True
            else :
                make=False
                if len(red.channels)==1 :
                    try : sflat= CCDData.read(reddir+'sflat.fits')
                    except : make=True
                else :
                    sflat=[]
                    for channel in red.channels :
                        try : sflat.append(CCDData.read(reddir+'sflat_'+channel+'.fits'))
                        except : make=True
            if make :
                sflat=[]
                tot=[]
                for flat in flats :
                    print(' '+flat['id'])
                    if os.path.exists(reddir+'flat_'+flat['id']+'.fits') and not clobber :
                        out=CCDData.read(reddir+'flat_'+flat['id']+'.fits')
                    else :
                        out = comb.superflat(flat['frames'],superbias=sbias,display=display)
                        red.scatter(out,scat=red.scat,display=display)
                        red.write(out,reddir+'flat_'+flat['id'],overwrite=True)
                    if flat['use'] : 
                        # we may have multiple channels to combine
                        if type(out) is not list : out=[out]
                        for i in range(len(out)) :
                            try: 
                                sflat[i] = sflat[i].add(out[i].multiply(out[i].header['MEANNORM']))
                                tot[i]+=out[i].header['MEANNORM']
                            except: 
                                sflat.append( copy.deepcopy(out[i].multiply(out[i].header['MEANNORM'])) )
                                tot.append(out[i].header['MEANNORM'])
                if darkflat is not None : 
                    if type(darkflat) is not list : darkflat=[darkflat]
                    for i in range(len(sflat)) : sflat[i]=sflat[i].subtract(darkflat[i])
                for i in range(len(sflat)) : sflat[i] = sflat[i].divide(tot[i])
                if len(sflat) is 1 : sflat= sflat[0]
                red.write(sflat,reddir+'sflat',overwrite=True)
            else : print('  already made!')
        except:
            print('no flat frames given')
            sflat=None

        # existing trace template
        traces=pickle.load(open(ROOT+'/data/'+inst+'/'+inst+'_traces.pkl','rb'))

        # existing wavecal template
        waves=pickle.load(open(ROOT+'/data/'+inst+'/'+group['config']['wref']+'.pkl','rb'))

        # wavecals
        try :
            wavecals=group['arcs']
            print('create wavecals')
            for wavecal in wavecals :
                if clobber :
                    make = True
                else :
                    make = False
                    try: waves_all = pickle.load(open(reddir+wavecal['id']+'.pkl','rb'))
                    except : make=True
                if make :
                    # combine frames
                    arcs=comb.sum(wavecal['frames'],return_list=True, superbias=sbias, crbox=[5,1], display=display)
                    try:
                        darks=comb.sum(wavecal['darks'],return_list=True)
                        for i,dark in enumerate(darks) :arcs[i]=arcs[i].subtract(dark)
                    except: pass
                    print('extract wavecal')

                    # loop over channels
                    waves_all=[]
                    for arc,wave,trace in zip(arcs,waves,traces) :
                 
                        # loop over windows
                        waves_channel=[]
                        for iwind,(wcal,wtrace) in enumerate(zip(wave,trace)) :
                            # extract
                            if group['config']['wavecal_type'] == 'echelle' :
                                arcec=wtrace.extract(arc,plot=display)
                                wcal.identify(spectrum=arcec, rad=3, display=display, plot=plot)
                            elif group['config']['wavecal_type'] == 'longslit' :
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
        except :
            print('no wavecal frames given')
            w=None

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
            print("extracting object {}",id)
            frames=red.reduce(id,superbias=sbias,superdark=sdark,superflat=sflat2d,scat=red.scat,
                              return_list=True,crbox=red.crbox,display=display) 
            try : rad = group['objects']['extract1d']['rad'] 
            except : rad = None
            if plot is not None : 
                plot.clf()
                ax=[]
                ax.append(plot.add_subplot(2,1,1))
                ax.append(plot.add_subplot(2,1,2,sharex=ax[0]))
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
                        for row in range(ec.data.shape[0]) :
                            gd=np.where(ec.mask[row,:] == False)[0]
                            plots.plotl(ax[0],wcal[row,gd],ec.data[row,gd],yr=[0,3*med],xt='Wavelength',yt='Flux')
                            plots.plotl(ax[1],wcal[row,gd],ec.data[row,gd]/ec.uncertainty.array[row,gd],xt='Wavelength',yt='S/N')
                        plot.suptitle(ec.header['OBJNAME'])
                        plt.draw()
                        plot.canvas.draw_idle()
                        plt.pause(0.1)
                        input("  hit a key to continue")
                ec.write(reddir+ec.header['FILE'].replace('.fits','.ec.fits'),overwrite=True)
                pdb.set_trace()
                if display is not None : display.clear() 

    # 2d extractions
#    traces=pickle.load(open(group['inst']+'_trace.pkl','rb'))
#    if type(traces) is not list : traces = [traces]
#    pdb.set_trace()
#    for id in group['objects']['extract2d']['frames'] : 
#        frames=red.reduce(id,superbias=sbias,superdark=sdark,superflat=sflat,scat=red.scat,return_list=True) 
#        for frame,trace in zip(frames,traces) :
#            out=trace.extract2d(frame,plot=t)


