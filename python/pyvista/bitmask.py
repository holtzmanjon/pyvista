#Routines for handling bitmasks

import numpy as np
import pdb
import sys

class BitMask():
    '''
    Base class for bitmasks, define common methods
    
    At a minimum, a BitMask will have a set of name, level, descrip

    BitMask provides 3 methods:
        getname(val,level=level) : returns name(s) of all set bits
                                  (optionally, of requested level)
        getval(name) : returns value of bit with input name
        badval()     : returns value of all bits that are marked bad (level=1)
        warnval()    : returns value of all bits that are marked warn (level=2)
    '''

    def getname(self,val,level=0,strip=True):
        '''
        Given input value, returns names of all set bits, optionally of a given level
        '''
        strflag=''
        for ibit,name in enumerate(self.name) :
            if (name != 'RESERVED' ) :
                try:
                    if ( val & 2**ibit ) > 0 and ( level == 0 or self.level == level ) :
                      strflag = strflag + name +','
                except: 
                    print('bit problem: ', ibit)
                    pdb.set_trace()
        if strip : return strflag.strip(',')
        else : return strflag

    def getval(self,name) :
        """
        Get the numerical bit value of a given character name(s)
        """
        if type(name) is str :
            name = [name]
        bitval = np.int64(0)
        for n in name :
            try:
                j=self.name.index(n.strip())
                bitval|=np.int64(2**j)
            except :
                print('WARNING: undefined name: ',n)
        return bitval


    def badval(self) :
        """
        Return bitmask value of all bits that indicate BAD in input bitmask
        """
        val=np.int64(0)
        for i,level in enumerate(self.level) :
            if level == 1 :
                try: val=val | np.int64(2**i)
                except: pdb.set_trace()
        return val

    def warnval(self) :
        """
        Return bitmask value of all bits that indicate BAD in input bitmask
        """
        val=np.int64(0)
        for i,level in enumerate(self.level) :
            if level == 2 :
                val=val | np.int64(2**i)
        return val

    def print(self,fmt='txt',fp=sys.stdout) :
        """ Formatted output of bit definitions
        """
        if fmt == 'txt' :
            fp.write('{:25s}{:>6s}  {:s}\n'.format('Name','Bit','Description'))
        elif fmt == 'wiki' :
            fp.write('||{:25s}||{:>6s}||{:s}||'.format('Name','Bit','Description'))
        elif fmt == 'latex' :
            fp.write('{:25s}&{:>6s}&{:s}\\\\'.format('Name','Bit','Description'))
        elif fmt == 'par' :
            fp.write('masktype {:s} {:d}\n'.format(self.flagname,len(self.name)))
        elif fmt == 'html' or fmt == 'sdsshtml' :
            if fmt == 'html' :
                fp.write("[SDSS_GROUP TITLE='<h2 id={:s}>{:s}']\n".format(self.shorttitle,self.title))
                fp.write("{:s}\n".format(self.blurb))
            else :
                fp.write('<div id="{:s}" class="panel panel-default">\n'.format(self.flagname))
                fp.write('<div class="panel-heading">\n')
                fp.write('<h3 class="panel-title"><a class="accordion-toggle" href="#collapse{:s}" data-toggle="collapse" data-parent="#accordion-bitmask">{:s}&nbsp; </a></h3>\n'.format(self.shorttitle,self.title))
                fp.write('</div>\n')
                fp.write('<div id="collapse{:s}" class="panel-collapse collapse">\n'.format(self.flagname))
                fp.write('<div class="panel-body">\n')

            fp.write('<table class="table table-bordered table-condensed"\n')
            fp.write('<thead>\n')
            fp.write('<tr><th style="white-space:nowrap;">Bit&nbsp;Name</th><th style="white-space:nowrap;">Binary&nbsp;Digit</th><th>Description</th></tr>\n')
            fp.write('</thead>\n')
            fp.write('<tbody>\n')
        for ibit,name in enumerate(self.name) :
            if (name != 'RESERVED' and name != '' ) :
                if fmt == 'txt' :
                    fp.write('{:25s}{:6d}  {:s}\n'.format(name,ibit,self.descrip[ibit]))
                elif fmt == 'wiki' :
                    fp.write('||{:25s}||{:6d}||{:s}||'.format(name,ibit,self.descrip[ibit]))
                elif fmt == 'latex' :
                    fp.write('{:25s}&{:6d}&{:s}\\\\'.format(name,ibit,self.descrip[ibit]))
                elif fmt == 'par' :
                    fp.write('maskbits {:s} {:d} {:s} "{:s}"\n'.format(self.flagname,ibit,name,self.descrip[ibit]))
                elif fmt == 'html' :
                    fp.write('<tr><td style="white-space:nowrap;">{:s}<td>{:d}<td>{:s}\n'.format(name,ibit,self.descrip[ibit]))
        if fmt == 'html' or fmt == 'sdsshtml' :
            fp.write('</tbody>\n')
            fp.write('</table>\n')
            if fmt == 'html' :
                fp.write("[/SDSS_GROUP]\n")
            else :
                fp.write("</div>\n")
                fp.write("</div>\n")
                fp.write("</div>\n")

class PixelBitMask(BitMask) :
    '''
    BitMask class for APOGEE pixel bitmask (APOGEE_PIXMASK)
    '''
    flagname='PIXMASK'
    shorttitle='PixelBitMask'
    title='PIXMASK : Abitmask for individual pixels'
    blurb='This bitmask is used to provide information associated with individual pixels in a one-dimensional spectrum. At the visit level, <code>PIXMASK</code> refers to the individual spectrum. At the combined level, PIXMASK attempts to appropriately combine the PIXMASKs of the visit spectrum level. '
    name=(['BADPIX','CRPIX','SATPIX','UNFIXABLE','BADDARK','BADFLAT','BADERR','NOSKY',
          'LITTROW_GHOST','PERSIST_HIGH','PERSIST_MED','PERSIST_LOW','SIG_SKYLINE','SIG_TELLURIC','NOT_ENOUGH_PSF','',
          'INACTIVE_PIXEL','','','','','','','',
          'BAD_EXTRACTION','','','','','','','RESERVED'])

    level=([1,1,1,1,1,1,1,1,
            0,0,0,0,0,0,1,0,
            1,0,0,0,0,0,0,0,
            1,0,0,0,0,0,0,0])

    maskcontrib=([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
                 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.0,
                 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.0,
                 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.0])

    descrip=([
     'Pixel marked as BAD in bad pixel mask or from strong persistence jump',
     'Pixel marked as cosmic ray',
     'Pixel marked as saturated',
     'Pixel marked as unfixable',
     'Pixel marked as bad as determined from dark frame',
     'Pixel marked as bad as determined from flat frame',
     'Pixel set to have very high error (not used)',
     'No sky available for this pixel from sky fibers',
     'Pixel falls in Littrow ghost, may be affected',
     'Pixel falls in high persistence region, may be affected',
     'Pixel falls in medium persistence region, may be affected',
     'Pixel falls in low persistence region, may be affected',
     'Pixel falls near sky line that has significant flux compared with object',
     'Pixel falls near telluric line that has significant absorption',
     'Less than 50 percent PSF in good pixels',
     '',
     'Pixel masked by FERRE mask < 0.001',
     '',
     '',
     '',
     '',
     '',
     '',
     '',
     'Bad extraction',
     '',
     '',
     '',
     '',
     '',
     '',
     ''
    ])

def print_bitmasks(fmt='html',out=None) :
  
    if out is not None : fp = open(out,'w') 
    else : fp  = sys.stdout

    for mask in [ PixelBitMask() ] :
        mask.print(fmt=fmt,fp=fp)
