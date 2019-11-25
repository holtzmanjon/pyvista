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

        # set up Reducer, Combiner, and output directory
        inst = group['inst']
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
                    try: waves_all = pickle.load(open(reddir+'/'+wavecal['id']+'.pkl','rb'))
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
                    pickle.dump(waves_all,open(reddir+'/'+wavecal['id']+'.pkl','wb'))
        except :
            print('no wavecal frames given')
            w=None

        # 1d extractions
        pdb.set_trace()
        #trace=pickle.load(open(group['inst']+'_trace.pkl','rb'))
        #display=t
        for id in group['objects']['extract1d'] : 
            frames=red.reduce(id,superbias=sbias,superdark=sdark,superflat=sflat,scat=red.scat,return_list=True,crbox=red.crbox,display=display) 
            for frame,wave,trace in zip(frames,waves_all,traces) :
                for iwind,(wcal,wtrace) in enumerate(zip(wave,trace)) :
                    shift=wtrace.find(frame,plot=display) 
                    ec=wtrace.extract(frame,plot=display)

    # 2d extractions
#    traces=pickle.load(open(group['inst']+'_trace.pkl','rb'))
#    if type(traces) is not list : traces = [traces]
#    pdb.set_trace()
#    for id in group['objects']['extract2d'] : 
#        frames=red.reduce(id,superbias=sbias,superdark=sdark,superflat=sflat,scat=red.scat,return_list=True) 
#        for frame,trace in zip(frames,traces) :
#            out=trace.extract2d(frame,plot=t)


def mktrace(ymlfile) :

    # trace from scratch
    try :
        traces=group['traces']
        print('create trace')
        if group['traces']['frame'] == 'sflat' : 
            a=sflat
        else :
            a= red.reduce(group['traces']['frame'],superbias=sbias,superdark=sdark,superflat=sflat,scat=red.scat)
        try : 
            b= red.reduce(group['traces']['dark'],superbias=sbias,superdark=sdark,superflat=sflat,scat=red.scat)
            a=a.subtract(b)
        except: pass
        if group['inst'] == 'TSPEC' :
            apers=[155,316,454,591,761]
            apers.reverse()
            traces=spectra.Trace(inst=group['inst'])
            traces.trace(a,apers,sc0=350,plot=display)
            traces=[traces]
        elif group['inst'] == 'ARCES' :
            apers=np.loadtxt('newap')[:,1]/4
            traces=spectra.Trace(inst=group['inst'])
            traces.trace(a,apers,sc0=1024,plot=display)
            traces=[[traces]]
        elif group['inst'] == 'DIS' :
            apers=[662,562]
            traces=[]
            for i,(im,ap) in enumerate(zip(a,apers)) :
                trace=spectra.Trace(inst=group['inst'],channel=i)
                trace.trace(im,ap,sc0=1024,plot=display)
                traces.append(trace)
            traces=[[traces[0]],[traces[1]]]
        pickle.dump(traces,open(group['inst']+'_trace.pkl','wb'))
    except:
        print('no trace frames given')

    # existing trace template
    trace=pickle.load(open(group['inst']+'_trace.pkl','rb'))


from pyvista import imred
import readmultispec
import os
import pickle
def dis() :
    dred=imred.Reducer(inst='DIS',dir='UT191019/DIS/')
    dcomb=imred.Combiner(reducer=dred)
    arc=dcomb.sum([1,2,3])

    # define a flat trace model
    def model(x) : return(np.zeros(len(x))+500)
    traces=Trace(rad=10,model=[model],sc0=1024)

    spec=traces.extract(arc[0])
    spec-=scipy.signal.medfilt(spec[0,:],kernel_size=101)

    wcal=WaveCal(type='chebyshev',spectrum=spec)
    f=open(os.environ['PYVISTA_DIR']+'/data/dis/dis_blue_lowres.pkl','rb')
    wcal0=pickle.load(f)
    wcal.identify(wcal0=wcal0,file=os.environ['PYVISTA_DIR']+'/data/henear.dat',rad=3)
    wcal.fit()
    w=wcal.wave(image=np.array(spec.data.shape))

    spec2d=traces.extract2d(arc[0],rows=range(200,900))
    pdb.set_trace()

    star=dred.reduce(26)
    return spec,w,wcal

def arces() :
    ered=imred.Reducer(inst='ARCES',dir='UT191020/ARCES/')
    ecomb=imred.Combiner(reducer=ered)
    flat=ecomb.sum([11,15])
    thar=ecomb.sum([19,20])

    t=tv.TV()

    apertures=np.loadtxt('newap')[:,1]
    #traces=trace(flat,apertures/4)
    traces=Trace()
    traces.trace(flat,apertures/4,sc0=1024,thresh=1000,plot=t)
    ec=traces.extract(thar,scat=True)

    wcal=spectra.WaveCal(type='chebyshev2D',orders=54+np.arange(107),spectrum=ec)
    wav=readmultispec.readmultispec('w131102.0004.ec.fits')['wavelen']
    new=ec.data*0.
    new[:,204:1855]=wav
    wcal.identify(ec,wav=new,file=os.environ['PYVISTA_DIR']+'/data/lamps/thar_arces',xmin=201,xmax=1849,thresh=200,rad=3,plot=t)
    wcal.fit()

    wcal.identify(ec,file=os.environ['PYVISTA_DIR']+'/data/lamps/thar_arces',xmin=150,xmax=1900,thresh=200,rad=3,plot=t)
    wcal.fit()

    w=wcal.wave(image=np.array(ec.data.shape))

    hr7950=ered.reduce(1,superflat=flat)
    hr7950ec=traces.extract(hr7950,scat=True)

    return flat,thar,traces,ec,wcal,w

from astropy.io import fits
def tspec() :

    tspec=imred.Reducer(inst='TSPEC',dir='UT191026/TSPEC',nfowler=8)   
    a=tspec.reduce(21)
    acr=tspec.reduce(21,crbox=[11,1])
    b=tspec.reduce(22)

    t=tv.TV()
    fig=plt.figure()

    rows=[[135,235],[295,395],[435,535],[560,660],[735,830]]
    apers=[155,316,454,591,761]    
    traces=[]
    waves=[]

    # wavelength arrays from TSpexTool reduction, note that flips wavelengths, and has K band first, not last
    wav=fits.open('tspec_wave.fits')[0].data
    #w0=[23147.049,17388.174,13927.875,11621.16,9972.335]
    #w0.reverse()
    #disp=[-2.8741264,-2.156124, -1.7261958,-1.4418244,-1.2400593]
    #disp.reverse()
    order=7
    for aper,row in zip(apers,rows) :
        trace=spectra.Trace(order=3,rows=row,lags=range(-75,75))
        trace.trace(a.subtract(b),aper,sc0=350,thresh=100,plot=t)
        traces.append(trace)
        trace.pix0 +=30
        out=trace.extract(acr,rad=20,plot=t)
        out.data=out.data - scipy.signal.medfilt(out.data,kernel_size=[1,201])
        wcal=spectra.WaveCal(type='chebyshev',orders=[order],degree=3)
        w=np.atleast_2d(wav[order-3,0,:][::-1])*1.e4
        bd=np.where(~np.isfinite(w))
        w[bd[0],bd[1]]=9000.
        #wcal.identify(out,file=os.environ['PYVISTA_DIR']+'/data/sky/OHll.dat',rad=10,wref=[wr,2048-1500],disp=d,plot=fig)
        wcal.identify(out,file=os.environ['PYVISTA_DIR']+'/data/sky/OHll.dat',rad=3,wav=w,plot=fig)
        wcal.fit()
        delattr(wcal,'ax')
        waves.append(wcal)
        order-=1

    #spec2d=traces.extract2d(a.subtract(b),rows=range(200,900))
    pickle.dump([traces],open('TSPEC_traces.pkl','wb'))
    pickle.dump([waves],open('TSPEC_waves.pkl','wb'))

    return traces,waves

def test():
    a=np.loadtxt('ecnewarc.ec')
    x=a[:,2]
    y=a[:,1]
    w=a[:,4]
    fitter=fitting.LinearLSQFitter()
    mod=models.Chebyshev2D(x_degree=3,y_degree=3,x_domain=[1,2000],y_domain=[55,160])
    fit=fitter(mod,x,y,w*y)
    plt.figure()
    plt.plot(w,w-fit(x,y)/y,'ro')

def testfit() :
    a=np.loadtxt('ecnewarc.ec')
    a[:,2]
    x=a[:,2]
    y=a[:,1]
    w=a[:,4]
    fitter=fitting.LinearLSQFitter()
    mod=models.Chebyshev2D(x_degree=3,y_degree=3,x_domain=[1,2000],y_domain=[55,160])
    fit=fitter(mod,x,y,w*y)
    plt.figure()
    plt.plot(w,w-fit(x,y)/y)
    plt.clf()
    plt.plot(w,w-fit(x,y)/y,'ro')


def echfit() :
    mod=models.custom_model(echelle_model)

def ripple(w,ord,amp=1,alpha=1,wc=5500) :
#def echelle_model(ind,amp,alpha,wc) :
#    w=ind[0,:]
#    ord=ind[1,:]
    x = ord * (1 - wc/w)
    out=amp*(np.sin(np.pi*alpha*x)/(np.pi*alpha*x))**2
    bd=np.where(np.pi*alpha*x == 0)[0]
    out[bd] = amp[bd]
    return out

def ripple2d(w,ord,amp0=1,amp1=0,amp2=0,alpha0=1,alpha1=0,alpha2=0,wc0=5500,wc1=0,wc2=0) :
#def echelle_model(ind,amp,alpha,wc) :
#    w=ind[0,:]
#    ord=ind[1,:]
    wc = wc0 + wc1*(ord-109) + wc2*(ord-109)**2
    amp = amp0 + amp1*(ord-109) + amp2*(ord-109)**2
    alpha = alpha0 + alpha1*(ord-109) + alpha2*(ord-109)**2
    #print(ord,amp,alpha,wc)
    x = ord * (1 - wc/w)
    out=amp*(np.sin(np.pi*alpha*x)/(np.pi*alpha*x))**2
    bd=np.where(np.pi*alpha*x == 0)[0]
    out[bd] = amp[bd]
    return out

def echelle_model_deriv(w,ord,amp=1,alpha=1,wc=5500) :
    x = ord * (1 - wc/w)
    f=amp*(np.sin(np.pi*alpha*x)/(np.pi*alpha*x))**2
    deriv=[]
    deriv.append(f/amp)
    deriv.append(2*amp*np.sin(np.pi*alpha*x)/(np.pi*alpha*x)**2*np.cos(np.pi*alpha*x)*np.pi*x
                 - 2*f/alpha )
    deriv.append(2*amp*np.sin(np.pi*alpha*x)/(np.pi*alpha*x)**2*np.cos(np.pi*alpha*x)*np.pi*alpha*-1*ord/w
                 + 2*f/x*ord/w)

    return np.array(deriv)

