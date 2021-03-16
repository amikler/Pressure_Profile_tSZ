import numpy as np
def rad_profile(image, pixel_size_arcmin, return_k=False):
    '''Computes the azimuthally-averaged radial profile of a given map.
    Parameters
    ----------
    image: 2D float or complex array
      Input image.
    pixel_size_arcmin: float
      Pixel size in arcmin.
    return_k: bool, optional
      If set to True, the provided image is assumed to be a power spectrum
      and the x-coordinate of the returned data is converted to the
      two-dimensional spatial frequency k. Default: None    
    Returns
    -------
    kradialprofile: float array
      x-coordinate of the radial profile. Either the radial separation
      from the image center or spatial frequency k, depending on the
      value of the variable return_k.
    radialprofile: float or complex array
      azimuthally averaged radial profile.
    '''

    nxpix, nypix = float(image.shape[0]),float( image.shape[1])

    YY, XX = np.indices((image.shape))
    r = np.sqrt((XX - nxpix//2)**2 + (YY - nypix//2.)**2)

    if return_k is True:
        k = 360/(pixel_size_arcmin/60.)*np.sqrt(((XX-nxpix//2)/nxpix)**2 + ((YY-nypix//2)/nypix)**2)
    else:
        k = np.copy(r)

    #r = np.round(r).astype(np.int)
    r_int = r.astype(np.int)

    weight = np.bincount(r_int.ravel())
    kradialprofile = np.bincount(r_int.ravel(), k.ravel()) / weight
    radialprofile_real = np.bincount(r_int.ravel(), np.real(image.ravel())) / weight
    radialprofile_imag = np.bincount(r_int.ravel(), np.imag(image.ravel())) / weight
    radialprofile = radialprofile_real + radialprofile_imag*1j

    return(kradialprofile, radialprofile)


def power_spec(image, pixel_size_arcmin, return_k=False, mask=None):
    '''Computes the azimuthally-averaged power spectrum of a given map.
    Parameters
    ----------
    image: 2D float array
        Input image
    pixel_size_arcmin: float
        Pixel size in arcmin.
    return_k: bool, optional
        If set to True, the provided image is assumed to be a power spectrum
        and the x-coordinate of the returned data is converted to the
        two-dimensional spatial frequency k. Default: None    
    Returns
    -------
    k: float array
        x-coordinate of the radial profile. Either the radial separation
        from the image center or spatial frequency k, depending on the
        value of the variable return_k.
    Pk: float or complex array
        azimuthally-averaged power spectrum
    '''
    # if mask is not None:
    #     image[mask] = -np.inf

    # rows, cols = np.where(image != -np.inf)
    # rows.sort(); cols.sort()
    # image =  image[rows[0]:rows[-1], cols[0]:cols[-1]]


    npix = image.shape[0] * image.shape[1]
    
    Fk = np.fft.fftshift((np.fft.fft2(image,  norm=None))) / npix

    ps=(np.absolute((Fk))**2)
    
    k, Pk = rad_profile(ps, pixel_size_arcmin, return_k=return_k)
    
    return(k, Pk)
