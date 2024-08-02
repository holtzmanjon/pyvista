# Here we define a Data class which is very similar to astropy CCDData and uses
# much of the astropy code, but defines a class that can also include a wavelength
# extension. We also add a bitmask attribute that is an integer rather than
# a boolean to allow it to be used as a bitmask. We also remove the requirement
# of including units. Finally, the name CCDData is not used, since we may 
# be using digital data that may not come from a CCD!

# Unfortunately, without redoing nddata, the methods provided by nddata, such
# as slicing and arithmetic, seem to drop the new attributes. So beware
# using those

from astropy.io import registry, fits
from astropy.nddata import ccddata, CCDData
import astropy.units as u
from astropy import log
from holtztools import plots
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pdb
from pyvista import bitmask
try : from linetools.spectra.xspectrum1d import XSpectrum1D
except: pass

from astropy.nddata.nduncertainty import (StdDevUncertainty, NDUncertainty, VarianceUncertainty, InverseVariance)
_known_uncertainties = (StdDevUncertainty, VarianceUncertainty, InverseVariance)
_unc_name_to_cls = {cls.__name__: cls for cls in _known_uncertainties}
_unc_cls_to_name = {cls: cls.__name__ for cls in _known_uncertainties}

class Data(CCDData) :
    """ Class to include a wavelength array on top of CCDData

    Parameters
    ----------

    Attributes
    ----------

    """
    def __init__(self, *args, **kwd):

        # Add bitmask attribute
        if 'bitmask' in kwd :
            self.bitmask = kwd['bitmask']
            kwd.pop('bitmask')
        else :
            self.bitmask = None

        # Add wavelength attribute
        if 'wave' in kwd :
            self.wave = kwd['wave']
            kwd.pop('wave')
        else :
            self.wave = None

        # Add response attribute
        if 'response' in kwd :
            self.response = kwd['response']
            kwd.pop('response')
        else :
            self.response = None

        # Add sky attribute
        if 'sky' in kwd :
            self.sky = kwd['sky']
            kwd.pop('sky')
        else :
            self.sky = None

        # Add sky attribute
        if 'skyerr' in kwd :
            self.skyerr = kwd['skyerr']
            kwd.pop('skyerr')
        else :
            self.skyerr = None


        ccddata._config_ccd_requires_unit = False
        super().__init__(*args, **kwd)

    def add_wave(self,wave) :
        """ Add a wavelength attribute to Data object

        Parameters
        ----------
        wave : float, array-like
               Wavelength array to add
        """
        self.wave = wave

    def add_bitmask(self,bitmask) :
        """ Add a bitmask attribute to Data object

        Parameters
        ----------
        bitmask : int, array-like
                   Bitmask array to add
        """
        self.bitmask = bitmask

    def add_response(self,response) :
        """ Add a response attribute to Data object

        Parameters
        ----------
        response : float, array-like
                   Response array to add
        """
        self.response = response

    def add_sky(self,sky) :
        """ Add a sky attribute to Data object

        Parameters
        ----------
        sky : flat, array-like
               Sky array to add
        """
        self.sky = sky

    def add_skyerr(self,skyerr) :
        """ Add a skyerr attribute to Data object

        Parameters
        ----------
        skyerr : flat, array-like
               Sky error array to add
        """
        self.skyerr = skyerr

    def to_hdu(self, hdu_bitmask='BITMASK', hdu_uncertainty='UNCERT',
               hdu_wave='WAVE', hdu_response='RESPONSE', hdu_sky='SKY', hdu_skyerr='SKYERR',
               wcs_relax=True, key_uncertainty_type='UTYPE', as_image_hdu=True):
        """Creates an HDUList object from a CCDData object.

        Parameters
        ----------
        hdu_bitmask, hdu_uncertainty, hdu_flags : str or None, optional
            If it is a string append this attribute to the HDUList as
            `~astropy.io.fits.ImageHDU` with the string as extension name.
            Flags are not supported at this time. If ``None`` this attribute
            is not appended.
            Default is ``'BITMASK'`` for mask, ``'UNCERT'`` for uncertainty

        wcs_relax : bool
            Value of the ``relax`` parameter to use in converting the WCS to a
            FITS header using `~astropy.wcs.WCS.to_header`. The common
            ``CTYPE`` ``RA---TAN-SIP`` and ``DEC--TAN-SIP`` requires
            ``relax=True`` for the ``-SIP`` part of the ``CTYPE`` to be
            preserved.

        key_uncertainty_type : str, optional
            The header key name for the class name of the uncertainty (if any)
            that is used to store the uncertainty type in the uncertainty hdu.
            Default is ``UTYPE``.

        as_image_hdu : bool
            If this option is `True`, the first item of the returned
            `~astropy.io.fits.HDUList` is a `~astropy.io.fits.ImageHDU`, instead
            of the default `~astropy.io.fits.PrimaryHDU`.

        Raises
        ------
        ValueError
            - If ``self.mask`` is set but not a `numpy.ndarray`.
            - If ``self.uncertainty`` is set but not a astropy uncertainty type.
            - If ``self.uncertainty`` is set but has another unit then
              ``self.data``.

        NotImplementedError
            Saving flags is not supported.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
        """
        if isinstance(self.header, fits.Header):
            # Copy here so that we can modify the HDU header by adding WCS
            # information without changing the header of the CCDData object.
            header = self.header.copy()
        else:
            # Because _insert_in_metadata_fits_safe is written as a method
            # we need to create a dummy CCDData instance to hold the FITS
            # header we are constructing. This probably indicates that
            # _insert_in_metadata_fits_safe should be rewritten in a more
            # sensible way...
            dummy_ccd = CCDData([1], meta=fits.Header(), unit="adu")
            for k, v in self.header.items():
                dummy_ccd._insert_in_metadata_fits_safe(k, v)
            header = dummy_ccd.header
        if self.unit is not u.dimensionless_unscaled and self.unit is not None :
            header['bunit'] = self.unit.to_string()
        if self.wcs:
            # Simply extending the FITS header with the WCS can lead to
            # duplicates of the WCS keywords; iterating over the WCS
            # header should be safer.
            #
            # Turns out if I had read the io.fits.Header.extend docs more
            # carefully, I would have realized that the keywords exist to
            # avoid duplicates and preserve, as much as possible, the
            # structure of the commentary cards.
            #
            # Note that until astropy/astropy#3967 is closed, the extend
            # will fail if there are comment cards in the WCS header but
            # not header.
            wcs_header = self.wcs.to_header(relax=wcs_relax)
            header.extend(wcs_header, useblanks=False, update=True)

         
        if as_image_hdu:
            hdus = [fits.PrimaryHDU(header=header)]
            hdus.append(fits.ImageHDU(self.data))
        else:
            hdus = [fits.PrimaryHDU(self.data, header)]

        if hdu_uncertainty and self.uncertainty is not None:
            # We need to save some kind of information which uncertainty was
            # used so that loading the HDUList can infer the uncertainty type.
            # No idea how this can be done so only allow StdDevUncertainty.
            uncertainty_cls = self.uncertainty.__class__
            if uncertainty_cls not in _known_uncertainties:
                raise ValueError('only uncertainties of type {} can be saved.'
                                 .format(_known_uncertainties))
            uncertainty_name = _unc_cls_to_name[uncertainty_cls]

            hdr_uncertainty = fits.Header()
            hdr_uncertainty[key_uncertainty_type] = uncertainty_name

            # Assuming uncertainty is an StdDevUncertainty save just the array
            # this might be problematic if the Uncertainty has a unit differing
            # from the data so abort for different units. This is important for
            # astropy > 1.2
            if (hasattr(self.uncertainty, 'unit') and
                    self.uncertainty.unit is not None):
                if not ccddata._uncertainty_unit_equivalent_to_parent(
                        uncertainty_cls, self.uncertainty.unit, self.unit):
                    raise ValueError(
                        'saving uncertainties with a unit that is not '
                        'equivalent to the unit from the data unit is not '
                        'supported.')

            hduUncert = fits.ImageHDU(self.uncertainty.array, hdr_uncertainty,
                                      name=hdu_uncertainty)
            print('appending uncertainty')
            hdus.append(hduUncert)

        if hdu_bitmask and self.bitmask is not None:
            # Always assuming that the mask is a np.ndarray (check that it has
            # a 'shape').
            if not hasattr(self.bitmask, 'shape'):
                raise ValueError('only a numpy.ndarray mask can be saved.')

            # Convert boolean mask to uint since io.fits cannot handle bool.
            hduMask = fits.ImageHDU(self.bitmask, name=hdu_bitmask)
            print('appending bitmask')
            hdus.append(hduMask)


        if hdu_wave and self.wave is not None :
            print('appending wave')
            hdus.append(fits.ImageHDU(self.wave,name=hdu_wave))

        if hdu_response and self.response is not None :
            print('appending response')
            hdus.append(fits.ImageHDU(self.response,name=hdu_response))

        if hdu_sky and self.sky is not None :
            print('appending sky')
            hdus.append(fits.ImageHDU(self.sky,name=hdu_sky))

        if hdu_skyerr and self.skyerr is not None :
            print('appending skyerr')
            hdus.append(fits.ImageHDU(self.skyerr,name=hdu_skyerr))


        hdulist = fits.HDUList(hdus)

        return hdulist

    def write(self,file,overwrite=True,png=False,imshow=False) :
        """  Write Data to file
        """
        self.to_hdu().writeto(file,overwrite=overwrite)

        if png :
            #backend=matplotlib.get_backend()
            #matplotlib.use('Agg')
            if imshow :
                fig=plt.figure(figsize=(12,9))
                vmin,vmax=minmax(self.data)
                plt.imshow(self.data,vmin=vmin,vmax=vmax,
                       cmap='Greys_r',interpolation='nearest',origin='lower')
                plt.colorbar(shrink=0.8)
                plt.axis('off')
                plt.savefig(file.replace('.fits','.png'))
            else : 
                fig,ax=plots.multi(1,1,figsize=(18,6))
                self.plot(ax)
                fig.savefig(file.replace('.fits','.png'))

            plt.close()
            #matplotlib.use(backend)

    def plot(self,ax,rows=None,**kwargs) :
        pixmask = bitmask.PixelBitMask()
        if self.data.ndim == 1 :
            gd = np.where((self.bitmask & pixmask.badval()) == 0)[0]
            plots.plotl(ax,self.wave[gd],self.data[gd],**kwargs)
        else :
            if rows is None : rows=range(self.wave.shape[0])
            for row in rows :
                try :  
                    gd = np.where((self.bitmask[row,:] & pixmask.badval()) == 0)[0]
                    plots.plotl(ax,self.wave[row,gd],self.data[row,gd],**kwargs)
                except :
                    plots.plotl(ax,self.wave[row,:],self.data[row,:],**kwargs)
        try :
            gd = np.where((self.bitmask & pixmask.badval()) == 0)[0]
            med=np.nanmedian(self.data[gd])
        except :
            med=np.nanmedian(self.data)
        ax.set_ylim(0,2*med)

    def to_linetools(self) :
        try: 
            if len(self.wave) == 1 and len(self.data) > 1 :
                wav = np.tile(self.wave,(len(self.data),1))
            else :
                wav = self.wave
            return XSpectrum1D(wav,self.data,self.uncertainty.array)
        except :
            print('linetools not available')


def fits_data_reader(filename, hdu=0, unit=None, hdu_uncertainty='UNCERT',
                        hdu_bitmask='BITMASK', hdu_wave='WAVE',
                        hdu_response='RESPONSE',hdu_sky='SKY', hdu_skyerr='SKYERR',
                        key_uncertainty_type='UTYPE', **kwd):
    """
    Generate a Data object from a FITS file. Modified from astropy fits_ccddata_reader

    Parameters
    ----------
    filename : str
        Name of fits file.

    hdu : int, str, tuple of (str, int), optional
        Index or other identifier of the Header Data Unit of the FITS
        file from which CCDData should be initialized. If zero and
        no data in the primary HDU, it will search for the first
        extension HDU with data. The header will be added to the primary HDU.
        Default is ``0``.

    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a
        unit for the image in the FITS header (the keyword ``BUNIT`` is used
        as the unit, if present), this argument is used for the unit.
        Default is ``None``.

    hdu_uncertainty : str or None, optional
        FITS extension from which the uncertainty should be initialized. If the
        extension does not exist the uncertainty of the CCDData is ``None``.
        Default is ``'UNCERT'``.

    hdu_bitmask : str or None, optional
        FITS extension from which the bitmask should be initialized. If the
        extension does not exist the bitmask of the CCDData is ``None``.
        Default is ``'BITMASK'``.

    hdu_flags : str or None, optional
        Currently not implemented.
        Default is ``None``.

    key_uncertainty_type : str, optional
        The header key name where the class name of the uncertainty  is stored
        in the hdu of the uncertainty (if any).
        Default is ``UTYPE``.

    kwd :
        Any additional keyword parameters are passed through to the FITS reader
        in :mod:`astropy.io.fits`; see Notes for additional discussion.

    Notes
    -----
    FITS files that contained scaled data (e.g. unsigned integer images) will
    be scaled and the keywords used to manage scaled data in
    :mod:`astropy.io.fits` are disabled.
    """
    unsupport_open_keywords = {
        'do_not_scale_image_data': 'Image data must be scaled.',
        'scale_back': 'Scale information is not preserved.'
    }
    for key, msg in unsupport_open_keywords.items():
        if key in kwd:
            prefix = f'unsupported keyword: {key}.'
            raise TypeError(' '.join([prefix, msg]))
    with fits.open(filename, **kwd) as hdus:
        hdr = hdus[hdu].header

        if hdu_uncertainty is not None and hdu_uncertainty in hdus:
            unc_hdu = hdus[hdu_uncertainty]
            stored_unc_name = unc_hdu.header.get(key_uncertainty_type, 'None')
            # For compatibility reasons the default is standard deviation
            # uncertainty because files could have been created before the
            # uncertainty type was stored in the header.
            unc_type = ccddata._unc_name_to_cls.get(stored_unc_name, StdDevUncertainty)
            uncertainty = unc_type(unc_hdu.data)
        else:
            uncertainty = None

        if hdu_bitmask is not None and hdu_bitmask in hdus:
            # Mask is saved as uint 
            bitmask = hdus[hdu_bitmask].data.astype(np.uintc)
        else:
            bitmask = None

        if hdu_wave is not None and hdu_wave in hdus:
            # Wavelength is saved as float
            wave = hdus[hdu_wave].data.astype(np.float32)
        else:
            wave = None

        if hdu_response is not None and hdu_response in hdus:
            # float
            response = hdus[hdu_response].data.astype(np.float32)
        else:
            response = None

        if hdu_sky is not None and hdu_sky in hdus:
            # float
            sky = hdus[hdu_sky].data.astype(np.float32)
        else:
            sky = None

        if hdu_skyerr is not None and hdu_skyerr in hdus:
            # float
            skyerr = hdus[hdu_skyerr].data.astype(np.float32)
        else:
            skyerr = None

        # search for the first instance with data if
        # the primary header is empty.
        if hdu == 0 and hdus[hdu].data is None:
            for i in range(len(hdus)):
                if (hdus.info(hdu)[i][3] == 'ImageHDU' and
                        hdus.fileinfo(i)['datSpan'] > 0):
                    hdu = i
                    comb_hdr = hdus[hdu].header.copy()
                    # Add header values from the primary header that aren't
                    # present in the extension header.
                    comb_hdr.extend(hdr, unique=True)
                    hdr = comb_hdr
                    log.info(f"first HDU with data is extension {hdu}.")
                    break

        if 'bunit' in hdr:
            fits_unit_string = hdr['bunit']
            # patch to handle FITS files using ADU for the unit instead of the
            # standard version of 'adu'
            if fits_unit_string.strip().lower() == 'adu':
                fits_unit_string = fits_unit_string.lower()
        else:
            fits_unit_string = None

        if fits_unit_string:
            if unit is None:
                # Convert the BUNIT header keyword to a unit and if that's not
                # possible raise a meaningful error message.
                try:
                    kifus = CCDData.known_invalid_fits_unit_strings
                    if fits_unit_string in kifus:
                        fits_unit_string = kifus[fits_unit_string]
                    fits_unit_string = u.Unit(fits_unit_string)
                except ValueError:
                    raise ValueError(
                        'The Header value for the key BUNIT ({}) cannot be '
                        'interpreted as valid unit. To successfully read the '
                        'file as CCDData you can pass in a valid `unit` '
                        'argument explicitly or change the header of the FITS '
                        'file before reading it.'
                        .format(fits_unit_string))
            else:
                log.info("using the unit {} passed to the FITS reader instead "
                         "of the unit {} in the FITS file."
                         .format(unit, fits_unit_string))

        use_unit = unit or fits_unit_string
        hdr, wcs = ccddata._generate_wcs_and_update_header(hdr)
        data = Data(hdus[hdu].data, meta=hdr, unit=use_unit,
                    bitmask=bitmask, uncertainty=uncertainty, wave=wave, wcs=wcs,
                    response=response, sky=sky, skyerr=skyerr)

    return data

registry.register_reader('fits', Data, fits_data_reader,force=False)
registry.register_reader('fit', Data, fits_data_reader,force=False)

def transpose(im) :
    """ Transpose a Data object
    """
    return Data(im.data.T,header=im.header,
                   uncertainty=StdDevUncertainty(im.uncertainty.array.T),
                   bitmask=im.bitmask.T,unit=u.dimensionless_unscaled)


def minmax(data,mask=None, low=3,high=10):
    """ Return min,max scaling factors for input data using median, and MAD

        Args:
            img : input CCDData
            low : number of MADs below median to return
            high : number of MADs above median to retunr

        Returns:
            min,max : low and high scaling factors
    """
    if mask is not None :
        gd = np.where(np.isfinite(data) & ~mask)
    else :
        gd = np.where(np.isfinite(data))
    std=np.median(np.abs(data[gd]-np.median(data[gd])))
    min = np.median(data[gd])-low*std
    max = np.median(data[gd])+high*std
    return min,max

