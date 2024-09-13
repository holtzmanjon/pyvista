# routines for a pyvista display tool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets  import RectangleSelector
import matplotlib
import scipy.stats
from astropy.wcs import wcs
from astropy.nddata import support_nddata
from . import cmap
from . import mmm
from . import image
try:
#   import autopy
    import pyautogui
except:
   print('pyautogui does not seem to be available, disabling arrow key cursor moves')

import pdb

sig2fwhm = 2*np.sqrt(2*np.log(2))

 
class TV:
    """
    A "TV" figure
    
    Usage: import tv
           tv=TV()  to set up a new TV object (display window)
    """
 
    def __init__(self, figsize=(12,8.5), aspect='equal', clickzoom=True):
        """
        Initialize TV object
        """
    
        # create new figure,set title, margins, and facecolor
        tv = plt.figure(figsize=figsize)
        self.fig = tv
        tv.canvas.manager.set_window_title('Image display window')
        tv.set_facecolor('darkred')
        rect = 0., 0.05, 0.7, 0.95
        ax = tv.add_axes(rect)
        self.ax = ax
        ax.axis('off')
        self.ax = ax
        self.axis = False
        self.aspect = aspect
        self.doflip = False
        self.usezoom = clickzoom
        self.histclick = True
        self.object = None

        # set up initial img and header lists
        self.current = -1
        self.images = 0
 
        # initialize rolling buffers
        self.img = None
        self.imglist = [None, None, None, None]
        self.hdr = None
        self.hdrlist = [None, None, None, None]
        self.scale = np.array([0.,1.])
        self.scalelist = [self.scale,self.scale,self.scale,self.scale]
        self.cmap = 'Greys_r'
        self.axlist = [None, None, None, None]

        # set up colorbar
        self.cb = None
        self.cblist = [None, None, None, None]
        rect = 0.00, 0.03, 0.7, 0.06
        self.cb_ax = tv.add_axes(rect)
        #tv.subplots_adjust(left=-0.15,right=1.15,bottom=-0.10,top=1.00)
        self.bottom = 0.
        self.top = 1.

        # plot windows
        rect = 0.74, 0.6, 0.25, 0.4
        plotax = tv.add_axes(rect)
        self.plotax1 = plotax
        rect = 0.74, 0.15, 0.25, 0.4
        plotax = tv.add_axes(rect)
        self.plotax2 = plotax

        # "lights" windows
        rect = 0.85, 0.02, 0.1, 0.05
        self.lgt1 = tv.add_axes(rect)
        self.lgt1.axis('off')
        self.lgt1.add_patch(patches.Rectangle((0,0),1,1,color='r',fill=True))
        self.lgt1.text(0.5,0.5,'Asynchronous',ha='center',va='center')

        # function to show image values, etc.
        def format_coord(x, y):
            x = int(x + 0.5)
            y = int(y + 0.5)
            if x< 0 or y<0 : return " "

            try:
                self.img
                try:
                    hdr=self.hdr
                    mywcs=wcs.WCS(hdr)
                    pixcrd = np.array([[x,y]])
                    world=mywcs.wcs_pix2world(pixcrd,1)
                    try:
                       object=self.hdr['object']
                    except:
                       object=self.object
                    return "[x,y]=[%4d, %4d] val=%8.5g   [%s %s]=[%10.6f,%10.6f]   OBJECT: %s" % (x,y, self.img[y, x], mywcs.wcs.ctype[0],mywcs.wcs.ctype[1],world[0,0], world[0,1], object)
                except:
                    mywcs=None
                try:
                    return "[x,y]\n [%4i, %4i] val=%8.5g OBJECT: %s" % (x,y, self.img[y, x], object)
                except IndexError:
                    return ""
            except:
                return " [%4i, %4i]" % (x, y)

        # set this routine up for format
        ax.format_coord = format_coord

        #event handling 
        self.event = None
        # turn off default key bindings
        tv.canvas.mpl_disconnect(tv.canvas.manager.key_press_handler_id)
        # set up our event handler
        self.cid = tv.canvas.mpl_connect('key_press_event', self.__onEvent)
        self.cid2 = tv.canvas.mpl_connect('button_press_event', self.__onEvent)
        self.cid3 = tv.canvas.mpl_connect('button_release_event', self.__onEvent)
        self.cid4 = tv.canvas.mpl_connect('motion_notify_event', self.__onEvent)
        self.button = False
        self.blocking = 0

    def __onEvent(self, event):
        """
        Handler for all trapped events. 
        
        Args:
          event -- a KeyEvent
        """
        self.event = event
        subPlotNr = self.__getSubPlotNr(event)        

        if event.name == 'key_press_event' :
            # keypress events: '-', '+/=', 'r'
            self.key = event.key

            # function for computing "scale"
            def scale() :
                # compute screen pixels per image pixel
                p1 = self.ax.transData.transform((0.,0.))
                p2 = self.ax.transData.transform((100.,100.))
                return (p2[1]-p1[1])/100., (p2[0]-p1[0])/100.

            if event.key == '-' or event.key == '+' or event.key == '=':
                # rolling image buffer
                if event.key == '-' :
                    self.current = (self.current-1) % self.images
                elif event.key == '+' or event.key == '=':
                    self.current = (self.current+1) % self.images
                self.img = self.imglist[self.current]
                self.hdr = self.hdrlist[self.current]
                self.scale = self.scalelist[self.current]
                for i in range(self.images) :
                    if i == self.current :
                        self.axlist[i].set_visible(True)
                    else :
                        self.axlist[i].set_visible(False)
                self.aximage=self.axlist[self.current]
                self.cb=self.cblist[self.current]
                #self.cb.ax.clear()
                #self.cb = self.fig.colorbar(self.aximage,cax=self.cb.ax,orientation='horizontal')
                #self.cb = self.ax.get_figure().colorbar(self.aximage,cax=self.cb_ax,orientation='horizontal')
                #cm=cmap.remap(self.cmap,self.bottom,self.top)
                #self.aximage.set_cmap(cm)
                plt.draw()
                try:
                    #x,y= autopy.mouse.location()
                    #autopy.mouse.move(int(x),int(y))
                    x,y=pyautogui.position()
                    pyautogui.moveTo(int(x),int(y))
                except: pass

            elif (event.key == 'p' or event.key == 'v') and subPlotNr == 0 :
                # find peak or valley near cursor position and move mouse there
                n=7
                xdata=int(round(event.xdata))
                ydata=int(round(event.ydata))
                if event.key == 'p' :
                    py, px = np.unravel_index(np.argmax(self.img[ydata-n:ydata+n,xdata-n:xdata+n]),
                                              self.img[ydata-n:ydata+n,xdata-n:xdata+n].shape)
                else :
                    py, px = np.unravel_index(np.argmin(self.img[ydata-n:ydata+n,xdata-n:xdata+n]),
                                              self.img[ydata-n:ydata+n,xdata-n:xdata+n].shape)
                px-=n
                py-=n
                try:
                    xs,ys = scale()
                    #x,y= autopy.mouse.location()
                    #autopy.mouse.move(int(x+px*xs),int(y-py*ys))
                    x,y= pyautogui.position()
                    pyautogui.moveTo(int(x+px*xs),int(y-py*ys))
                except: pass

            elif event.key == 'r' and subPlotNr == 0 :
                # in display window, redraw image at original zoom
                dim=np.shape(self.img)
                size=np.max([dim[0],dim[1]])
                self.ax.set_xlim(dim[1]/2.-size/2.,dim[1]/2.+size/2.)
                if self.doflip :self.ax.set_ylim(dim[0]/2.+size/2.,dim[0]/2.-size/2.)
                else :self.ax.set_ylim(dim[0]/2.-size/2.,dim[0]/2.+size/2.)
                #self.ax.set_xlim(-0.5,dim[1]-0.5)
                #self.ax.set_ylim(-0.5,dim[0]-0.5)
                plt.draw()

            elif event.key == 'r' and subPlotNr == 1 :
                # in color bar, redraw image at original color scale
                self.bottom=0.
                self.top=1.
                cm=cmap.remap(self.cmap,self.bottom,self.top)
                self.aximage.set_cmap(cm)
                plt.draw()

            elif event.key == 'x' and subPlotNr == 0 :
                # row and column plots
                xdata=int(round(event.xdata))
                ydata=int(round(event.ydata))
                self.plotax1.cla()
                self.plotax1.plot(self.img[ydata,:])
                self.plotax1.set_xlabel('X',color='c')
                self.plotax1.tick_params(axis='x',colors='c')
                self.plotax1.tick_params(axis='y',colors='c')
                self.plotax1.set_xlim(self.ax.get_xlim())
                self.histclick = False
                self.plotax2.cla()
                self.plotax2.plot(self.img[:,xdata])
                self.plotax2.set_xlabel('Y',color='c')
                self.plotax2.tick_params(axis='x',colors='c')
                self.plotax2.tick_params(axis='y',colors='c')
                self.plotax2.set_xlim(self.ax.get_ylim())
                plt.draw()

            elif event.key == 'left' and subPlotNr == 0 :
                # move cursor
                xs,ys = scale()
                try:
                    #x,y= autopy.mouse.location()
                    x,y= pyautogui.position()
                    if xs < 1. :
                        #autopy.mouse.move(int(x-1),int(y))
                        pyautogui.moveTo(int(x-1),int(y))
                    else :
                        #autopy.mouse.move(int(x-xs),int(y))
                        pyautogui.moveTo(int(x-xs),int(y))
                except:
                    print('autopy error')

            elif event.key == 'right' and subPlotNr == 0 :
                # move cursor
                xs,ys = scale()
                try:
                    #x,y= autopy.mouse.location()
                    x,y= pyautogui.position()
                    if xs < 1. :
                        #autopy.mouse.move(int(x+1),int(y))
                        pyautogui.moveTo(int(x+1),int(y))
                    else :
                        #autopy.mouse.move(int(x+xs),int(y))
                        pyautogui.moveTo(int(x+xs),int(y))
                except: pass

            elif event.key == 'up' and subPlotNr == 0 :
                # move cursor
                xs,ys = scale()
                try:
                    #x,y= autopy.mouse.location()
                    x,y= pyautogui.position()
                    if ys < 1. :
                        #autopy.mouse.move(int(x),int(y-1))
                        pyautogui.moveTo(int(x),int(y-1))
                    else :
                        #autopy.mouse.move(int(x),int(y-ys))
                        pyautogui.moveTo(int(x),int(y-ys))
                except: pass

            elif event.key == 'down' and subPlotNr == 0 :
                # move cursor
                xs,ys = scale()
                try:
                    #x,y= autopy.mouse.location()
                    x,y= pyautogui.position()
                    if ys < 1. :
                        #autopy.mouse.move(int(x),int(y+1))
                        pyautogui.moveTo(int(x),int(y+1))
                    else :
                        #autopy.mouse.move(int(x),int(y+ys))
                        pyautogui.moveTo(int(x),int(y+ys))
                except: pass

            elif event.key == 'a' and subPlotNr == 0 :
                # toggle axes on and off
                if self.axis :
                    rect = 0., 0.05, 0.7, 0.95
                    self.ax.axis('off')
                else :
                    rect = 0.05, 0.15, 0.65, 0.85
                    self.ax.axis('on')
                self.ax.set_position(rect)
                self.axis = not self.axis
                plt.draw()

            elif event.key == 'z' and subPlotNr == 0 :
                self.usezoom = not self.usezoom

            elif event.key == '#' and subPlotNr == 0 :
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                if (xlim[1]-xlim[0]) > 32 or (ylim[1]-ylim[0]) > 32 :
                    print('too many pixels to label, zoom in further first')
                    return
                for ix in range(int(xlim[0]),int(xlim[1])) :
                    for iy in range(int(ylim[0]),int(ylim[1])) :
                        self.ax.text(ix,iy,'{:d}'.format(int(self.aximage.get_array()[iy,ix])),ha='center',va='center')
                plt.draw()
                print('use $ to remove pixel label values')

            elif event.key == '$' and subPlotNr == 0 :
                for text in self.ax.texts : text.set_visible(False)
                plt.draw()

            elif event.key == '%' and subPlotNr == 0 :
                self.tvclear()

            elif event.key == 'h' or event.key == '?' :
                # print help
                print('Asynchronous commands: ')
                print('Image window: ')
                print('  mouse (with clickzoom enabled):')
                print('    left mouse  : zoom in, centered on cursor')
                print('    center mouse: zoom out, centered on cursor')
                print('    right mouse : pan, center to cursor')
                print('  keys:')
                print('    r           : redraw at default zoom')
                print('    +/=         : toggle to next image in stack')
                print('    -           : toggle to previous image in stack')
                print('    arrow keys  : move single image pixels')
                print('    x           : cross-section plots at cursor position')
                print('    p/v         : find nearest peak/valley')
                print('    a           : toggle axes on/off')
                print('    z           : toggle zoom with mouse (clickzoom) on/off')
                print('    #           : label pixels with values')
                print('    $           : clear text ')
                print('    %           : clear patches ')
                print('    h/?         : print this help')

            if self.blocking == 1 : self.__stopBlock()

        elif event.name == 'button_press_event' :
            if self.usezoom and subPlotNr == 0 :
                # button press in image window to zoom/pan
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                if event.button == 1 :
                    # zoom in
                    xsize = ( xlim[1]-xlim[0] )/ 2.
                    ysize = ( ylim[1]-ylim[0] )/ 2.
                elif event.button == 3 :
                    # zoom out
                    xsize = ( xlim[1]-xlim[0] )* 2.
                    ysize = ( ylim[1]-ylim[0] )* 2.
                else :
                    # pan
                    xsize = xlim[1]-xlim[0]
                    ysize = ylim[1]-ylim[0]
                size=max([xsize,ysize])
                self.ax.set_xlim(event.xdata-size/2.,event.xdata+size/2.)
                if self.doflip:self.ax.set_ylim(event.ydata+size/2.,event.ydata-size/2.)
                else : self.ax.set_ylim(event.ydata-size/2.,event.ydata+size/2.)
                plt.draw()
            elif subPlotNr == 1 :
                # flag button press in colorbar
                self.button = True
                disp = self.fig.axes[1].transData.transform([event.ydata,event.xdata])
                #xstart,xend = self.fig.axes[1].transAxes.inverted().transform(disp)
                ystart,xstart = self.fig.axes[1].transAxes.inverted().transform(disp)
                #self.xstart = event.xdata
                self.xstart = xstart

            elif subPlotNr == 2 and self.histclick :
                # mouse click in plotax1 changes limits 
                ylim=self.plotax1.get_ylim()
                if event.button == 1 :
                    self.scale[0] = event.xdata
                    self.lowline[0].remove()
                    self.lowline = self.plotax1.plot(
                              [self.scale[0],self.scale[0]],
                              [ylim[0],ylim[1]*0.8],
                              ls=':',color='r')
                elif event.button == 3 :
                    self.scale[1] = event.xdata
                    self.highline[0].remove()
                    self.highline = self.plotax1.plot(
                              [self.scale[1],self.scale[1]],
                              [ylim[0],ylim[1]*0.8],
                              ls=':',color='r')
                self.aximage = self.ax.imshow(self.img,
                                   vmin=self.scale.min(),vmax=self.scale.max(),
                                   cmap=self.cmap, 
                                   interpolation='nearest',aspect=self.aspect)
                self.plotax1.set_ylim(ylim)
                plt.draw()

        elif event.name == 'button_release_event' :
            self.button = False

        elif event.name == 'motion_notify_event'  and self.button :
            disp = self.fig.axes[subPlotNr].transData.transform([event.ydata,event.xdata])
            yend,xend = self.fig.axes[subPlotNr].transAxes.inverted().transform(disp)
            # if motion in colorbar with key pressed in colorbar, adjust colorbar
            if subPlotNr == 1 :
                if event.button == 2 :
                    diff = (xend - self.xstart)
                    self.top = self.top + diff
                    self.bottom = self.bottom + diff
                    self.xstart = xend
                else :
                    if self.xstart > 0.5 :
                      if xend > self.bottom :
                          self.top = xend
                      else :
                          self.top = self.bottom
                    else :
                      if xend < self.top :
                          self.bottom = xend
                      else :
                          self.bottom = self.top
                cm=cmap.remap(self.cmap,self.bottom,self.top)
                self.aximage.set_cmap(cm)
                plt.draw()

        
    def __getSubPlotNr(self, event):
    
        """
        Get the nr of the subplot that has been clicked
        
        Args::
          event -- an event
        
        Returns:
          A number or None if no subplot has been clicked
        """
    
        i = 0
        axisNr = None
        for axis in self.fig.axes:
            if axis == event.inaxes:
                axisNr = i        
                break
            i += 1
        return axisNr
        
    def __stopBlock(self) :
        """
        stops blocking for keypress event
        """
        if self.blocking == 1 :
            self.fig.canvas.stop_event_loop()

    def __startBlock(self) :
        """
        starts blocking for keypress event
        """
        self.blocking = 1
        self.fig.canvas.start_event_loop(1000)

    def flip(self) :
        """ toggle display flip
        """
        self.doflip = not self.doflip
        ylim = self.ax.get_ylim()
        if self.doflip : self.ax.set_ylim(np.max(ylim),np.min(ylim))
        else : self.ax.set_ylim(np.min(ylim),np.max(ylim))
        plt.draw()

    def tv(self,img,min=None,max=None,same=False,cmap=None,sn=False,object=None, draw=True) :
        """
        main display routine: displays image with optional scaling

        Args:
          img: pyvista Data object, numpy array, or fits HDU
          min=, max= : optional scaling arguments
          same= : bool, if True use display scaling from previous image
          cmap= : specify different color map
        """
        # load data array depending on input type
        if sn :
            try : data = img.data / img.uncertainty.array
            except: raise ValueError('with sn, input must be Data type')
        elif isinstance(img, (np.ndarray)) :
            data = img
        elif isinstance(img.data, (np.ndarray)) :
            data = img.data
        else :
            print('input must be numpy array or have data attribute that is')
            return

        # if object is explicitly specified, load it
        self.object = object

        # set figure and axes
        plt.figure(self.fig.number)
        plt.axes(self.ax)
        #self.clear()

        # make last image not visible so we don't see anything 
        #   if new image is smaller
        if self.axlist[self.current] is not None: 
            self.axlist[self.current].set_visible(False)
        if same :
            min = self.scale[0]
            max = self.scale[1]

        # load new image data onto rolling stack
        current= (self.current+1) % 4
        self.images += 1
        if self.images > 4 : self.images = 4
        self.current = current
        self.imglist.pop(current)
        self.imglist.insert(current,data)
        self.img = data

        # save the header if we have one
        if hasattr(img,'header') :
            self.hdrlist.pop(current)
            self.hdrlist.insert(current,img.header)
            self.hdr=img.header
        else :
            self.hdr=None
      
        # get autodisplay parameters if needed, and save display params
        if min is None : 
           min = 0.
        if max is None : 
           min,max = image.minmax(data)
        self.scale = np.array([min,max])
        self.scalelist.pop(current)
        self.scalelist.insert(current,self.scale)

        if cmap != None :
           self.cmap = cmap
 
        # display image and new colorbar 
        dim=np.shape(self.img)
        size=np.max([dim[0],dim[1]])
        self.ax.set_xlim(dim[1]/2.-size/2.,dim[1]/2.+size/2.)
        if self.doflip : self.ax.set_ylim(dim[0]/2.+size/2.,dim[0]/2.-size/2.)
        else : self.ax.set_ylim(dim[0]/2.-size/2.,dim[0]/2.+size/2.)

        self.aximage = self.ax.imshow(data,vmin=min,vmax=max,cmap=self.cmap, 
                                      interpolation='nearest',aspect=self.aspect)
        old=self.axlist.pop(current)
        #self.tvclear()

        # if we had a previous image, reload the data with a single value
        # so we don't continually accumulate memory (matplotlib doesn't
        # describe how memory can be released
        z=np.zeros([1,1])
        if old is not None : old.set_data(z)
        self.axlist.insert(current,self.aximage)
        if self.cb is None :
            #self.cb = self.fig.colorbar(self.aximage,orientation='horizontal',shrink=0.7,pad=0)
            self.cb = self.fig.colorbar(self.aximage,cax=self.cb_ax,orientation='horizontal')
            #plt.subplots_adjust(left=-0.15,right=1.15,bottom=-0.10,top=1.00)
        else :
            self.cb.ax.clear()
            self.cb = self.fig.colorbar(self.aximage,cax=self.cb.ax,orientation='horizontal')
        self.cblist.pop(current)
        self.cblist.insert(current,self.cb)

        # instead of redraw color, could replace data, but not if sizes change?
        # img.set_data()
        # img.changed()
        # plt.draw()

        # image histogram in plotax1
        self.plotax1.cla()
        self.plotax1.hist(data.flatten(),bins=np.linspace(data.min(),data.max(),100))
        ylim=self.plotax1.get_ylim()
        self.lowline = self.plotax1.plot([self.scale[0],self.scale[0]],
                                         [ylim[0],ylim[1]*0.8],
                                         ls=':',color='r')
        self.highline = self.plotax1.plot([self.scale[1],self.scale[1]],
                                         [ylim[0],ylim[1]*0.8],
                                         ls=':',color='r')
        self.plotax1.text(0.05,0.95,'Image histogram : ',
                          transform=self.plotax1.transAxes)
        self.plotax1.text(0.1,0.90,'left click for new lower scale',
                          transform=self.plotax1.transAxes)
        self.plotax1.text(0.1,0.85,'right click for new higher scale',
                          transform=self.plotax1.transAxes)
        self.plotax1.text(0.1,0.8,'(or use min= and/or max= in tv())',
                          transform=self.plotax1.transAxes)
        self.plotax1.set_ylim(ylim)
        self.histclick = True

        if draw: plt.draw()
        try :
          x,y=pyautogui.position()
        except:
          x,y=(500,500)
        # following fails on Windows QT
        try: self.fig.canvas.motion_notify_event(x,y)
        except : pass
        self.fig.canvas.flush_events()

    def tvtext(self,x,y,text,color='m',ha='center',va='center') :
        """ Annotates with text
        """
        self.ax.text(x,y,text,ha=ha,va=va,color=color)

    def tvcirc(self,x,y,rad=3,color='m',ls=None,lw=None) :
        """
        displays a circle on an image

        Args:
          x,y : center position of patch

        Keyword args :
          size= :  patch size
          color= :  patch color
        """
        self.ax.add_patch(patches.Circle((x,y),rad,fill=False,color=color,
                          ls=ls,lw=lw))
        self.fig.canvas.flush_events()
        plt.draw()

    def intbox(self) :
        def box_callback(eclick,erelease) :
            global x1, y1, x2, y2
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            #rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2),fill=False )
            #self.ax.add_patch(rect)
            self.fig.canvas.stop_event_loop()

        usezoom = self.usezoom
        self.usezoom=False
        rs = RectangleSelector(self.ax, box_callback,
                       useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=False)
        self.fig.canvas.start_event_loop(0)
        self.usezoom = usezoom
        return image.BOX(sc=int(min(x1,x2)),sr=int(min(y1,y2)),nc=int(np.abs(x1-x2)),nr=int(np.abs(y1-y2)))


    def tvbox(self,x=0,y=0,box=None,size=3,color='m',ls=None,lw=None) :
        """
        displays a patch (box by default) on an image

        Args:
          x,y : center position of patch

        Keyword args :
          size= :  patch size
          color= :  patch color
        """
        plt.figure(self.fig.number)
        if box is not None :
            x0=box.xmin
            x1=box.xmax
            y0=box.ymin
            y1=box.ymax
            xsize=(x1-x0)
            ysize=(y1-y0)
        else :
            x0=x-size/2
            xsize=size
            y0=y-size/2
            ysize=size
        self.ax.add_patch(patches.Rectangle((x0,y0),xsize,ysize,fill=False,color=color,ls=ls,lw=lw))
        self.fig.canvas.flush_events()
        plt.draw()

    def clear(self) :
        """  Clear image
        """
        self.ax.cla()
        self.ax.axis('off')
        self.plotax1.cla()
        self.plotax2.cla()

    def tvclear(self) :
        """
        clears patches from image
        """
        plt.figure(self.fig.number)
        for i in range(len(self.ax.patches)) : self.ax.patches[0].remove()
        plt.draw()

    def tvmark(self) :
        """
        Blocking input: waits for key press in display and returns key 
        that was pressed and data pixel location of the keypress

        Args:
          none

        Returns:
          key pressed, x data position, y data position
        """
        self.light(self.lgt1,'Input','g')
        self.__startBlock()
        reserved=['r','p','v','left','right','up','down','-','+','=','%','#','$'] 
        if self.event.key in reserved : self.tvmark()
        self.light(self.lgt1,'Asynchronous','r')
        key,x,y = self.event.key,self.event.xdata,self.event.ydata
        self.fig.canvas.flush_events()
        return key,x,y

    def light(self,ax,text,color) :
        ax.cla()
        ax.axis('off')
        ax.add_patch(patches.Rectangle((0,0),1,1,color=color,fill=True))
        ax.text(0.5,0.5,text,ha='center',va='center')
        plt.draw()
 
    def fill(self) :
        y,x=np.mgrid[0:100,0:100]
        self.tv(x)
        self.tv(y)
        self.tv(x+y)
        self.tv(x-y)

    def imexam(self,size=11,fwhm=5,scale=1,pafixed=False) :
        """ Fit gaussian and show radial profile of stars marked interactively
        """
        key=''
        rect = 0.74, 0.15, 0.25, 0.4
        plotax = self.fig.add_axes(rect,projection='3d')
        print('Hit key near star center, "q" to quit')
        while key != 'q' :
            key,x,y=self.tvmark()
            if key == 'q' : 
                self.plotax2 = self.fig.add_axes(rect)
                return
            self.plotax1.cla()
            amp,xcen,ycen,xfwhm,yfwhm,theta,back= \
                    image.gfit2d(self.img,x,y,size=size,fwhm=fwhm,scale=scale,plot=self.plotax1,
                            sub=False,pafixed=pafixed,astropy=False)
            self.tvcirc(xcen,ycen,np.sqrt(xfwhm*yfwhm)/2.)
            self.histclick = False

            #3D plot
            x=int(x)
            y=int(y)
            yg,xg=np.mgrid[y-size:y+size,x-size:x+size]
            plotax.cla()
            plotax.plot_surface(xg, yg, self.img[y-size:y+size,x-size:x+size],cmap='jet')
            self.fig.canvas.flush_events()

    def savefig(self,name) :
        """ hardcopy of only display Axes
        """
        extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(name, bbox_inches=extent)

