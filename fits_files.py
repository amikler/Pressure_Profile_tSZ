from astropy.io import fits
import numpy as np


def open_fits(path, data_a=False, sim_a= False, TF_a=False, other=None, weight =False, extension=0, **kwargs):

	import sys

	#### High S/N Simulation ####
	if sim_a ==True:
		if 'simnum' in kwargs:
			simnum = kwargs['simnum']
		else:
			simnum = 0 #sys.argv[2]	
		#print ("simulation number", simnum)
		
		hdulist=fits.open(str(path)+'/sim.reducednoise/'+str(simnum)+'.fits') #for new reduce simulations - sim 5/sim6

		

	elif data_a == True:
		#### Data ####
		hdulist=fits.open(str(path)+'/data/image.fits') #for new reduce data - sim 5/sim6

	elif TF_a == True:
		######## TF #######
		hdulist=fits.open(str(path)+'/data/tcentral.fits') #TF0 - new reductions sim 5 /6
	elif other is not None:
		hdulist=fits.open(path+other) #TF0 - new reductions sim 5 /6

	if extension != 0:
		print ("Reading extension %0.0f of the fits file" %float(extension))
	data= hdulist[extension].data
	header = hdulist[extension].header

	#mask =np.logical_not(np.isnan(data))
	#mask =np.isnan(data)
	data = np.ma.masked_array(data, np.isnan(data)).data
	mask = np.ma.masked_array(data, np.isnan(data)).mask
	
	data[np.isnan(data)] = 0 #We convert the nans to 0 because the radial code deals with this 0 values but cannot handle the nan
	if weight == True:
		weight_map = hdulist[1].data
		return data, header, weight_map
	##########

	hdulist.close()
	return data, header, mask

def write_fits(filename, data, header=None):

	hdu = fits.PrimaryHDU(data, header=header)
	hdu.writeto(filename, overwrite=True)

	return None

