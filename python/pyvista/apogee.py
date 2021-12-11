from astropy.io import fits
import numpy as np
import pdb

def unzip(file,dark=None,cube=False,cds=True) :
    """ Read APOGEE .apz file, get CDS image
    """
    # open file and confirm checksums
    hd=fits.open(file, do_not_scale_image_data = True, uint = True, checksum = True)

    # file has initial header, avg_dcounts, then nreads
    nreads = len(hd)-2
    try:
        avg_dcounts=hd[1].data
    except:
        # fix header if there is a problem (e.g., MJD=55728, 01660046)
        hd[1].verify('fix')
        avg_dcounts=hd[1].data

    # first read is in extension 2
    ext = 2

    # loop over reads, processing into raw reads, and appending
    for read in range(1,nreads+1) :
        header = hd[ext].header
        try:
          raw = hd[ext].data
        except:
          hd[ext].verify('fix')
          raw = hd[ext].data
        if read == 1 :
          data = np.copy(raw)
          if cube : 
              data3d=np.zeros([nreads,2048,2048],dtype=np.int16)
              data3d[0]=data[0:2048,0:2048]
        else :
          data = np.add(data,raw,dtype=np.int16)
          data = np.add(data,avg_dcounts,dtype=np.int16)
          if read == 2 : first = data
          if cube :
              data3d[read-1]=data[0:2048,0:2048]

        ext += 1

      # compute and add the cdsframe, subtract dark if we have one
    cds = (data[0:2048,0:2048] - first[0:2048,0:2048] ).astype(float)
    if dark is not None :
        # if we don't have enough reads in the dark, do nothing
        try :
            cds -= (dark[nreads-1,:,:] - dark[2,:,:])
        except:
            print('not halting: not enough reads in dark, skipping dark subtraction for mjdcube')
            pass

    if cube : return data3d
    else : return cds
