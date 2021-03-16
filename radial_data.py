import pdb
import math

def radial_data(data,annulus_width=1,working_mask=None, weight = None, x=None,y=None,rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    :INPUT:
      data   - whatever data you are radially averaging.  Data is
              binned into a series of annuli of width 'annulus_width'
              pixels.

      annulus_width - width of each annulus.  Default is 1.

      working_mask - array of same size as 'data', with zeros at
                        whichever 'data' points you don't want included
                        in the radial data computations.
      weight - array of same size as 'data', 
               with wieght values for each point in 'data'.

      x,y - coordinate system in which the data exists (used to set
               the center of the data).  By default, these are set to
               integer meshgrids

      rmax -- maximum radial value over which to compute statistics
    
    :OUTPUT:
        r - a data structure containing the following
                   statistics, computed across each annulus:

          .r      - the radial coordinate used (outer edge of annulus)

          .mean   - mean of the data in the annulus

          .sum    - the sum of all enclosed values at the given radius

          .std    - standard deviation of the data in the annulus

          .median - median value in the annulus

          .max    - maximum value in the annulus

          .min    - minimum value in the annulus

          .numel  - number of elements in the annulus

    :EXAMPLE:        
      ::
        
        import numpy as np
        import pylab as py
        import radial_data as rad

        # Create coordinate grid
        npix = 50.
        x = np.arange(npix) - npix/2.
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2)
        fake_psf = np.exp(-(r/5.)**2)
        noise = 0.1 * np.random.normal(0, 1, r.size).reshape(r.shape)
        simulation = fake_psf + noise

        rad_stats = rad.radial_data(simulation, x=xx, y=yy)

        py.figure()
        py.plot(rad_stats.r, rad_stats.mean / rad_stats.std)
        py.xlabel('Radial coordinate')
        py.ylabel('Signal to Noise')
    """
    
# 2012-02-25 20:40 IJMC: Empty bins now have numel=0, not nan.
# 2012-02-04 17:41 IJMC: Added "SUM" flag
# 2010-11-19 16:36 IJC: Updated documentation for Sphinx
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    import numpy as np

    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None
            self.rmean = None
            self.meannan = None 
            self.meanweight = None
    #---------------------
    # Set up input parameters
    #---------------------
    data = np.array(data)
    
    if working_mask is None:
        working_mask = np.ones(data.shape,bool)

    if weight is not None:
      weight_data = data#*weight 
    else:
    	weight_data = data
    
    npix, npiy = data.shape
    if x==None or y==None:
        x1 = np.arange(-npix/2.,npix/2.) #x1=np.arange(npix-npix,npix)
        y1 = np.arange(-npiy/2.,npiy/2.) #y1=np.arange(npiy-npiy,npiy)
        x,y = np.meshgrid(y1,x1)

    r =  abs(x+1j*y) #abs(np.hypot(1*x,1*y)) #distance from center for each point
    #print (r[0,0])
    #print (r[540,540])
    if rmax==None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container - empty
    #---------------------
    
    dr = np.abs([x[0,0] - x[0,1]]) * annulus_width #width (rmax of the bin)
    radial = np.arange(rmax/dr)*dr + dr/2. #makes the radial coordinate - half point on the bin
    
    
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = np.zeros(nrad)
    radialdata.sum = np.zeros(nrad)
    radialdata.std = np.zeros(nrad)
    radialdata.median = np.zeros(nrad)
    radialdata.numel = np.zeros(nrad, dtype=int)
    radialdata.max = np.zeros(nrad)
    radialdata.min = np.zeros(nrad)
    radialdata.r = radial # gives you the middle point of the bin
    radialdata.rmean = np.zeros(nrad)
    radialdata.meannan = np.zeros(nrad)
    radialdata.meanweight = np.zeros(nrad)
    
    #---------------------
    # Loop through the bins
    #---------------------
    #bin23=np.zeros(shape=(1,9428))
    for irad in range(nrad): #= 1:numel(radial)
      

      minrad = irad*dr #lower edge of bin
      maxrad = minrad + dr # upper edge of bin - excluded

      thisindex = (r>=minrad) * (r<maxrad) * working_mask #true or false about the statement
      #import pylab as py
      #pdb.set_trace() #debbuger
      #print data[irad,irad]

      #if not math.isnan(data[irad, irad]):
      #  continue
      if not thisindex.ravel().any(): #if not true statements
        #continue

        radialdata.mean[irad] = np.nan
        radialdata.sum[irad] = np.nan
        radialdata.std[irad]  = np.nan
        radialdata.median[irad] = np.nan
        radialdata.numel[irad] = 0
        radialdata.max[irad] = np.nan
        radialdata.min[irad] = np.nan

      else:

        nonzero= np.count_nonzero(data[thisindex])
        #if nonzero ==0:
        

        if nonzero > 0: #if nonzero = 0 it means no values in the bin
          radialdata.meannan[irad] = data[thisindex].sum()/nonzero
          if weight is not None:
            if np.all(weight == 1):
              radialdata.meanweight[irad] =radialdata.meannan[irad]
            else:
          	  radialdata.meanweight[irad] = weight_data[thisindex].sum()/weight[thisindex].sum()
          #print 'nonzero',nonzero
        else: #meaning nonzero = 0 all thew values are 0 which means they were nan before
           radialdata.meannan[irad] = 0
           radialdata.meanweight[irad] =0

        radialdata.mean[irad] = data[thisindex].mean()
        radialdata.sum[irad] = (np.abs(data[thisindex])).sum() #data[r<maxrad] gives you the sum up to that radii; data[thisindex].sum - gives you the sum only on the bin 
        radialdata.std[irad]  = np.nanstd(data[thisindex])
        radialdata.median[irad] = np.median(data[thisindex])
        radialdata.numel[irad] = data[thisindex].size #number of points per bin
        radialdata.max[irad] = data[thisindex].max()
        radialdata.min[irad] = data[thisindex].min()
        radialdata.rmean[irad] = ((r[thisindex].sum())/data[thisindex].size)
        #print 'real_size', data[thisindex].size, 'r_sum', r[thisindex].sum()

        # if nonzero > 0: #if nonzero = 0 it means no values in the bin
        #   numzeros = data[thisindex].size - nonzero
        #   actual = data[thisindex].size - numzeros
        #   #print r[thisindex].sum(), nonzero
        #   radialdata.rmeannan[irad] = ((r[thisindex].sum())/actual)

        # else: #meaning nonzero = 0 all thew values are 0 which means they were nan 
        #    radialdata.rmeannan[irad] = 0
        # #  radialdata.rmeannan[irad] = ((r[thisindex].sum())/data[thisindex].size)
       
        #if irad==(nrad-1)  :
           #pass irad==23:
          #+= data[thisindex] 
          #print data[thisindex]
    if weight is not None:
      #print ('!!meannan is the weighted mean since a weight map was given!')
      radialdata.meannan = radialdata.meanweight
        
    #---------------------
    # Return with data
    #---------------------
    if __name__ == '__main__':
      main()
    
    return radialdata


def radialdata_applied (data, annulus_width, rmax=None, working_mask=None, weight=None):
  import numpy as np
  rad_stats_data =radial_data(data, annulus_width=annulus_width, rmax=rmax, working_mask=working_mask, weight=weight) #radial profile with model convolved
  r_rad_data = np.array([rad_stats_data.rmean])
  meannan_rad_data = np.array([rad_stats_data.meannan])

  return r_rad_data, meannan_rad_data



