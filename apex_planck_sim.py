import numpy as np 
import GNFW_functions as fun 
import fits_files as ff


reload(fun)
reload(ff)

def sim2d( theta, ndim,  A10, apex = False, planck = False, zznoise =False, **kwargs):

	#kwargs = {'z':z,'pixindeg':pixindeg_apex,'freq_obs':freq_obs, 'rmax':rmax, 'm500':M500, 'r500':r500}

	#kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500
	z=kwargs['z']
	pixindeg =kwargs['pixindeg']
	pixel_to_meters, pixeltoarcsec = fun.pix_con (z, pixindeg)#[0]
	path = kwargs['path']
	#noise_factor = kwargs['noise_factor']
	noise_factor = kwargs.get('noise_factor', 0)
	name = kwargs['name']

	if 'path' in kwargs: del kwargs['path']
	if 'noise_factor' in kwargs: del kwargs['noise_factor']
	if 'name' in kwargs: del kwargs['name']
	
	
	if apex == True:


		################ CREATE GRID ###################
		rmax = 270
		xx, yy= np.meshgrid(
						np.linspace(0,2*rmax+0,2*rmax+1),
						np.linspace(0,2*rmax+0,2*rmax+1))


		xx=xx*pixel_to_meters #m
		yy=yy*pixel_to_meters #m

		if 'fitcenter' in kwargs:
			xcenter = int(round(theta[3]))
			ycenter = int(round(theta[4]))
		else:
			xcenter = rmax*pixel_to_meters 
			ycenter =  rmax*pixel_to_meters
			zcenter = rmax*pixel_to_meters
		rr=np.sqrt( (xx-xcenter)**2 + (yy-ycenter)**2) #m 
				#
		###############################
		if 'hotpix' in kwargs:
			#print ('hotpix')
			zz = np.zeros(rr.shape)
			zz[zz.shape[0]//2+1,zz.shape[1]//2+1] = theta*10
			if 'smoothing' in kwargs:
				from scipy import ndimage
				#print ('Gaussian kernel')
				if 'smoothing' == True:
					beam_size = 60
				else:
					beam_size = kwargs.get('smoothing')
				d = beam_size/(pixeltoarcsec*2.35482)
				zz_con=ndimage.filters.gaussian_filter(zz, d, mode='constant')
			elif 'smoothing05'  in kwargs:
				from scipy import ndimage
				d = 30/(pixeltoarcsec*2.35482)
				zz_con=ndimage.filters.gaussian_filter(zz, d, mode='constant')
			else:
				from scipy import signal
				#print ('TF')
				zz_con = signal.fftconvolve(zz,kwargs['trans_fun'], mode='same')
			if 'y_prior' in kwargs:
				Y_vals = np.inf 

		elif 'y_prior' in kwargs:
			zz, zz_con, Y_vals = fun.gnfw_function_apex(theta=theta,r=rr, ndim = ndim, A10=A10,**kwargs)
		else:
			#print (kwargs)
			zz, zz_con= fun.gnfw_function_apex(theta=theta,r=rr, ndim = ndim, A10=A10,   **kwargs)#*10# temp map [mK CMB]


		if zznoise ==True:
			map_size = int(zz_con.shape[0]/2.)
			if 'white' in kwargs:
				print ('White Noise Map')
				sigma = 1#1.55 #sigma obtained from the bootstrap maps
				noise_map = np.random.normal(0,sigma, size=(map_size*2 + 1, map_size*2 +1))
				if 'smoothing' in kwargs:
					from scipy import ndimage
					print ('Gaussian kernel')
					if 'smoothing' == True:
						beam_size = 60
					else:
						beam_size = kwargs.get('smoothing')
					
					d = beam_size/(pixeltoarcsec*2.35482)
					noise_map=ndimage.filters.gaussian_filter(noise_map, d, mode='constant')
				elif 'smoothing05' in kwargs:
					from scipy import ndimage
					print ('Gaussian kernel 0.5 arcmin')
					d = 30/(pixeltoarcsec*2.35482)
					noise_map=ndimage.filters.gaussian_filter(noise_map, d, mode='constant')
				else:
					from scipy import signal
					print ('TF')
					noise_map = signal.fftconvolve(noise_map,kwargs['trans_fun'], mode='same')
			else:
				ii = kwargs.get('simnum',np.random.randint(100)) # 0 
				print ('noise map number,', ii)
				try:
					noise_map = ff.open_fits(path, other='/noise.images/'+str(ii)+'.fits')[0]
				except:
					ii = int(ii) +1
					noise_map = ff.open_fits(path, other='/'+str(ii)+'.fits')[0] 
			noise_map = noise_map *noise_factor#*0.01#0.1#*0.01
			noise_map =noise_map[rmax-map_size:rmax+map_size+1,rmax-map_size:rmax+map_size+1]
			zz_noise = zz_con +noise_map

		else:
			zz_noise = np.zeros(zz.shape)

	if planck ==True:
		import random

		rmax = kwargs['rmax']
		if 'sample_olda' in kwargs:del kwargs['sample_olda']
		if 'mymcmc' in kwargs:del kwargs['mymcmc']

		################ CREATE GRID ###################
		xx, yy= np.meshgrid(
						np.linspace(0,2*rmax+0,2*rmax+1),
						np.linspace(0,2*rmax+0,2*rmax+1))


		xx=xx*pixel_to_meters #m
		yy=yy*pixel_to_meters #m
		if 'fitcenter' in kwargs:
			xcenter = int(round(theta[3]))
			ycenter = int(round(theta[4]))
		else:
			xcenter = rmax*pixel_to_meters 
			ycenter =  rmax*pixel_to_meters
			zcenter = rmax*pixel_to_meters
		rr=np.sqrt( (xx-xcenter)**2 + (yy-ycenter)**2) #m 

		################################

		if 'rmax' in kwargs: del kwargs['rmax']

		zz, zz_con = fun.gnfw_function_planck(theta, rr, ndim, A10, **kwargs) #y-map

		
		if zznoise == True:
			map_size = int(zz_con.shape[0]/2.)
			if 'white' in kwargs:
				from scipy import ndimage
				print ('White Noise')
				resolution = kwargs['resolution']
				d = resolution/(pixindeg*3600*2.35482)
				sigma = 1.445e-6 * noise_factor #sigma obtained from the half ring maps
				noise_map = np.random.normal(0,sigma, size=(map_size*2 + 1, map_size*2 +1))
				noise_map=ndimage.filters.gaussian_filter(noise_map, d, mode='constant')
			else:
				i = kwargs.get('simnum', np.random.randint(999))#random.randint(0,999)
				#i = 916
				#print ("Simnum", i)
				if 'patches' in kwargs:
					print ('Cluster Surroundings')
					name = name+'_patches'

				if 'cov_mod_ymap' in kwargs:
					print ('Cluster Surroundings')
					name = name+'_patches'
					
				noise_map = ff.open_fits(path, other='noise_maps_MILCA/'+str(name)+'_noise.fits')[0]
				if i > noise_map.shape[0]:
					i = random.randint(0,noise_map.shape[0])
				noise_map = noise_map[i,:,:]*noise_factor
				xcent, ycent = noise_map.shape
				xcent, ycent = xcent/2, ycent/2
				noise_map =noise_map[ycent-map_size:ycent+map_size+1,xcent-map_size:xcent+map_size+1]
			zz_noise = zz_con +noise_map

		else:
			zz_noise = np.zeros(zz.shape)

	if 'y_prior' in kwargs:
		return zz, zz_con, Y_vals, zz_noise
	else:
		return zz, zz_con, zz_noise
	

def rmax_cal(highsn_model, data, width, rmax, working_mask=None, **kwargs):

	import radial_data as rad
	import numpy as np
	import GNFW_functions as fun 

	reload(rad)
	reload(fun)

	#set minimun r as 1.3*r500
	m500 = kwargs['m500']
	r500_arcsec = kwargs['r500']
	pixindeg = kwargs['pixindeg']
	if m500 is not None:
		z = kwargs['z']
		r500_arcsec =fun.mr500 (z, m500=m500)[2]

	r500 = r500_arcsec/(3600*pixindeg) 
	print ('R500 [arcmin] %0.2f' %(r500_arcsec/60))


	#Determine where the nans start
	npix, npiy = data.shape
	x1 = np.arange(-npix/2.,npix/2.) #x1=np.arange(npix-npix,npix)
	y1 = np.arange(-npiy/2.,npiy/2.) #y1=np.arange(npiy-npiy,npiy)
	x,y = np.meshgrid(y1,x1)

	r =  abs(x+1j*y) #abs(np.hypot(1*x,1*y)) #distance from center for each point
    #print (r[0,0])
	rmax_nan = r[working_mask].min()
	
	#convert mask so shows True the signal and False the nans
	working_mask = np.logical_not(working_mask)

	#Radial profile of  real data
	rad_stats_data =rad.radial_data(data, annulus_width=width, rmax=rmax, working_mask=working_mask) #radial profile with data
	r_rad_data = np.array([rad_stats_data.rmean])
	meannan_rad_data = np.array([rad_stats_data.meannan])

	#Radial profile of   high sn sim
	rad_stats_sim =rad.radial_data(highsn_model, annulus_width=width, rmax=rmax, working_mask=working_mask) #radial profile with data
	r_rad_sim= np.array([rad_stats_sim.r])
	meannan_rad_sim = np.array([rad_stats_sim.meannan])

	diff_o = np.inf
	#diff_array = np.zeros(shape=(r_rad_sim.shape[1]))
	for i in range (meannan_rad_data.shape[1]):
		area_data = np.trapz(abs(meannan_rad_data[0,0:i]))
		area_sim = np.trapz(abs(meannan_rad_sim[0,0:i]))
		diff = abs(area_data - area_sim)
		#diff_array[i] =diff
		if 0<diff <= diff_o and 2*r500<r_rad_data[0,i]+width< rmax_nan:
			diff_o = diff
			rmax = r_rad_data[0,i]+width
			#print (diff, r_rad_data[0,i])

	# diff_array[np.where(diff_array==0)]=np.inf
	# rmax2 = r_rad_data[0,np.where(abs(diff_array).min()==diff_array)]
	# print abs(diff_array).min(), rmax2
	rmax = (rmax+width)  #pixels
	return int(rmax)

def radectodeg(ra,dec):
	deg_ra = 0
	deg_dec = 0
	if len(ra)==1 and len(dec)==1:
		return [ra,dec]
	else:
		for i in ra[::-1]:
			if i != ra[0]:
				deg_ra = (i+deg_ra)/60.
			else:
				deg_ra = (deg_ra +i)*15
		for i in dec[::-1]:
			if i != dec[0]:
				deg_dec = (i+deg_dec)/60.
			else:
				deg_dec = deg_dec +abs(i)
				if i <0:
					deg_dec = deg_dec *-1
		return [deg_ra,deg_dec]


