def comb_data(list_clus, comb_state, path_apex= None, path_planck=None, simulation = False, intermediate_simulation = False, high_simulation = False, **kwargs):
	'''Kwargs: 
			*other_units: boolean. Used for the convertion from r500 to M500.  The function assumes the r500 in arcsec. However, if other unit is used add 'unit' key in kwargs with the unit. Default False.
		kwargs_apex{width}'''
	

	import GNFW_functions as fun 
	import fits_files as ff
	import apex_planck_sim as apsim
	import covariance as cov
	import radial_data as rad

	from astropy.wcs import WCS
	import numpy as np
	import matplotlib.pyplot as plt
	import pylab
	import sys
	from scipy import ndimage
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	from scipy import signal

	reload(fun)
	reload(ff)
	reload(apsim)
	reload(cov)
	reload(rad)
	pylab.ion()

	lc = list_clus

	#Reading the table
	data = list_clus
	name = []
	zlist = []
	state = []
	r500list = []
	ra_deg = []
	dec_deg =[]
	m500list = []
	m500err = []


	for i in range (len(lc)):
		name.append(lc[i][0])
		zlist.append(lc[i][1])
		state.append(lc[i][7])
		if kwargs.get('r500_inlist', False):
			other_units = kwargs.get('other_units', False)
			M500, r500_m, r500 = fun.mr500(lc[i][1], r500=lc[i][2], m500=None, other_units=other_units, **{'unit':kwargs.get('unit', '')})
			r500list.append(r500)
			m500list.append(M500/1e14)
			#r500list.append(lc[i][2])
			#m500list.append(fun.mr500(lc[i][1], r500=lc[i][2], m500=None, other_units=other_units, **{'unit':kwargs.get('unit', '')})[0])
		else:
			m500list.append(lc[i][2])
			r500list.append(fun.mr500(lc[i][1], m500=lc[i][2], r500=None)[2])
		m500err.append(np.average([lc[i][3], lc[i][4]])) #if r500 is given on the list instead of M500 the errors are 999 - meaning no errors calculated
		ra, dec = apsim.radectodeg(lc[i][5],lc[i][6])
		ra_deg.append(ra)
		dec_deg.append(dec)

	namelist = np.array(name)
	zlist = np.array(zlist)
	state = np.array(state)
	r500list = np.array(r500list)
	ra_deg = np.array(ra_deg)
	dec_deg= np.array(dec_deg)

	if comb_state == 'All':
		i = np.array([np.arange(state.shape[0])])
	elif comb_state =='Random':
		i = np.array([np.random.randint(0,len(state),5)])
	else:	
		i = np.array(np.where(state==comb_state)) #list indexes

	class Dat:
		"""z_selec
			r500arcsec
			ra
			dec
			rmax_selec [arcsec]
			cov_selec (inverse)
			data
			meannan_rad
			name_selec
			kwargs
		"""
		def __init__(self): 
			self.z_selec = None
			self.r500arcsec = None
			self.ra = None
			self.dec = None
			self.rmax_selec = None
			self.cov_selec = None
			self.data = None
			self.meannan_rad = None
			self.name_selec = None
			self.kwargs = None
	
	collection = Dat()
	collection.name_selec = []#np.zeros(len(i[0,:]))
	collection.z_selec = np.zeros(len(i[0,:]))
	collection.pixelsize= np.zeros(len(i[0,:])) 
	collection.r500arcsec = np.zeros(len(i[0,:]))
	collection.ra = np.zeros(len(i[0,:]))
	collection.dec = np.zeros(len(i[0,:]))
	collection.rmax_selec = np.zeros(len(i[0,:]))
	collection.cov_selec = []
	collection.meannan_rad = []
	collection.data = []
	collection.kwargs = []
	num = len(i[0,:])
	kwargs['num'] = num

	############## GENERAL INFO ######################
	A10=kwargs.get('A10',[8.403 ,1.177, 1.0510, 5.4905,0.3081]) #GNFW values
	ndim = kwargs.get('ndim', 3)
	if ndim==1:
		theta_m = A10[0]
	if ndim==2:
		theta_m = A10[0], A10[1]
	if ndim ==3:
		theta_m = A10[0], A10[1], A10[3]
	if ndim ==4:
		theta_m = A10[0], A10[1], A10[2], A10[3]
	if ndim ==5:
		theta_m =  A10[0], A10[1], A10[4], A10[2], A10[3]

	noise_factor = 1
	if intermediate_simulation:
		noise_factor = 0.1
	if high_simulation == True:
		noise_factor = 0.01
	M500 = None

	############## INFO APEX ###################

	ndim_a = kwargs.get('ndim', 3)
	theta_ma = kwargs.get('theta_ma', theta_m) #[0:ndim_a]
	width_a = kwargs.get('width_a', 6.0)
	
	
	#if 'ndim_a' is not  kwargs:
	#	ndim_a =3 
	#else:
	#	ndim_a = kwargs['ndim_a']

	#if 'theta_ma' is not kwargs:
	#	theta_ma = theta_m[0:ndim_a]
	#else:
	#	theta_ma = kwargs['theta_ma']

	#if 'width_a' is not kwargs:
	#	width_a = 6.0#pixel #0.2#6#0.2#*pixel_to_meters
	#else:
	#	width_a = kwargs['width_a']

	freq_obs=kwargs.get('freq_obs', 150*10**9) #Hz 152?
	############################################################################

	################ INFO PLANCK ###########
	
	
	ndim_p = kwargs.get('ndim', 3)
	theta_mp = kwargs.get('theta_mp', theta_m)#[0:ndim_p]
	resolution = kwargs.get('resolution',10*60) # arcsec #10 arcmin per pixel
	############################################################################


	print ('-------------------------------------------------------------------')
	print("Dynamical state = %s, Number of combined clusters = %i " %(str(comb_state), int(num)))
	print ('-------------------------------------------------------------------')

	#plt.figure()

	for index in range (num): 

		z = zlist[i[0,index]]
		r500 = r500list[i[0,index]]
		M500 = m500list[i[0,index]]
		M500_err = m500err[i[0,index]]
		name = namelist[i[0,index]]
		print ('* %s'%name)
		ra = ra_deg[i[0,index]]
		dec = dec_deg[i[0,index]]

		collection.z_selec[index]= z
		collection.r500arcsec[index]= r500
		collection.ra[index] = ra
		collection.dec[index] = dec
		collection.name_selec.append(name)


		if simulation == True:
			print ('SIMULATION')
		elif high_simulation ==True:
			print ("High S/N Simulation")
		else:
			print ('Data')	

		if path_apex is not None:

			temp_map, header, mask = ff.open_fits(path_apex+str(name),data_a=True)
			sim_a = ff.open_fits(path_apex+str(name), sim_a= True)[0]
			if 'repeat' in kwargs:
				path_apex_TF = kwargs['repeat']
			else:
				path_apex_TF = path_apex

			trans_fun = np.array(ff.open_fits(path_apex_TF+str(name), TF_a=True)[0])

			if simulation == True:
				if 'repeat' in kwargs:
					import random
					simnum_a =random.randint(0,99)
				else:
					simnum_a = 0
				print ("Simnum Apex", simnum_a)#[index])
				temp_map = ff.open_fits(path_apex+str(name),other= '/sim.noise/'+str(simnum_a)+'.fits')[0] #not necesarry to read the header because when cutting the simulation map we use the center of the map
				 #low S/N simulation

			##
			#if high_simulation == True or intermediate_simulation ==True or simulation ==True:
				#xcent, ycent = temp_map.shape[0]/2, temp_map.shape[1]/2
			try:
				w= WCS(header)
				if ra != header['CRVAL1'] or dec != header['CRVAL2']:
					print ('RA and Dec from list different than header. Using now RA and Dec from header')
					ra, dec = header['CRVAL1'], header['CRVAL2']
				xcent, ycent =w.wcs_world2pix(ra,dec,0)#fig.world2pixel(c[0],c[1])
				xcent, ycent  = int(round(xcent)), int(round(ycent))
				#xcent, ycent  = xcent -1, ycent-1
			except:
				xcent, ycent = temp_map.shape[0]/2, temp_map.shape[1]/2
			###
			
			pixindeg_apex =  header['CDELT2'] #0.002777778
			pixel_to_meters_a, pixeltoarcsec_a = fun.pix_con (z, pixindeg_apex)
			rmax_0a = int(sim_a.shape[0]/2)#map_size[l]#126#30#12#30#126#int(rmax)#90

			kwargs_apex = {'z':z,'pixindeg':pixindeg_apex,'freq_obs':freq_obs, 'rmax':rmax_0a, 'm500':M500, 'm500err':M500_err, 'r500':r500, 'trans_fun':trans_fun, 'path':path_apex, 'noise_factor': noise_factor, 'name':name, 'alphap':kwargs.get('alphap', False)}
			
			########################## CREATE A MODEL  - APEX #######################
			zz_con_a = apsim.sim2d(theta=theta_ma, ndim=ndim_a,  A10=A10, apex = True,  **kwargs_apex)[1]
			##########################################################################
						
			############################ CALCULATE RMAX -APEX ########################

			rmax_a = apsim.rmax_cal(highsn_model=zz_con_a, data=sim_a, width=width_a,  working_mask=mask, **kwargs_apex) #90
			rmax_a_arcsec = rmax_a*pixeltoarcsec_a
			collection.rmax_selec[index] = (rmax_a_arcsec)

			print ('Map Size Apex= %0.2f [arcmin]' %float(2*rmax_a_arcsec/60.))
			print ('Map Size Apex = %0.2f [R500]' %float(2*rmax_a_arcsec/r500))
			##########################################################################

			################### CMB Maps ##################################
			print ('Adding Convolved CMB Map to Filtered Simulations')

			n = np.random.randint(100)
			cmb_map = ff.open_fits('/vol/aibn160/data1/amikler/Documents/APEX_SZ/CMB_maps/', other='maps_3600/cmb_realization_'+str(n)+'.fits')[0] #in K
			cmb_map = cmb_map*1e3 #in mK
			cmb_map = signal.fftconvolve(cmb_map,trans_fun, mode='same')
			xcent_cmb, ycent_cmb = int(cmb_map.shape[0]/2.),int(cmb_map.shape[1]/2.) 
			cmb_map =cmb_map[ycent_cmb-rmax_a:ycent_cmb+rmax_a+1,xcent_cmb-rmax_a:xcent_cmb+rmax_a+1]
			##########################################################################

			######################## CUT APEX DATA TO DESIRED SIZE ################
			if simulation ==True:
				data_a = temp_map[rmax_0a-rmax_a+0:rmax_0a+rmax_a+1,rmax_0a-rmax_a+0:rmax_0a+rmax_a+1]
			else:
				data_a =temp_map[ycent-rmax_a+0:ycent+rmax_a+1,xcent-rmax_a+0:xcent+rmax_a+1]

			sim_a =sim_a[rmax_0a-rmax_a+0:rmax_0a+rmax_a+1,rmax_0a-rmax_a+0:rmax_0a+rmax_a+1]

			if high_simulation ==True:
				sim_a = sim_a #+ cmb_map
				collection.data.append(sim_a)
			if simulation == True:
				data_a = data_a + cmb_map
				collection.data.append(data_a)
			else:
				collection.data.append(data_a)
			#plt.imshow(data_a)
			#pylab.pause(0.01)
			##########################################################################

			############################## NOISE COVARIANCE - APEX #################

			cov_a, cov_i_a, error_a = cov.cov(name=name, rmax=rmax_a, xcent=xcent, ycent=ycent, width=width_a, path=path_apex+str(name),noise_factor = noise_factor, apex=True, **{'trans_fun':trans_fun} )[0:3]

			#collection.cov_selec += cov_a
			collection.cov_selec.append(cov_i_a)
			##########################################################################

			############################## RADIAL DATA - APEX #################

			#Radial profile of  high S/N simulation
			r_rad_sima, meannan_rad_sima = rad.radialdata_applied(sim_a, annulus_width=width_a, rmax=rmax_a)
			
			#Radial profile of  low S/N simulation or data
			r_rad_dataa, meannan_rad_dataa = rad.radialdata_applied(data_a, annulus_width=width_a, rmax=rmax_a)

					

			if high_simulation == True:
				collection.meannan_rad.append(meannan_rad_sima)
				sn = abs(meannan_rad_sima/error_a)[0,0]

			else:
				collection.meannan_rad.append(meannan_rad_dataa)
				sn = abs(meannan_rad_dataa/error_a)[0,0]

			print ('S/N - Apex = %0.2f '%sn)
			# if simulation == True:
			# 	print ("SIMULATION")
			# elif high_simulation == True:
			# 	print ("High S/N Simulation")
			# else:
			# 	print ("Data")


			r500 = None
			kwargs_apex = {'z':z,'pixindeg':pixindeg_apex,'freq_obs':freq_obs, 'rmax':rmax_a, 'm500':M500, 'm500err':M500_err, 'r500':r500, 'trans_fun':trans_fun, 'path':path_apex, 'noise_factor': noise_factor, 'name':name, 'width':width_a, 'num':num, 'sn':sn, 'alphap':kwargs.get('alphap', False), 'xcent':xcent, 'ycent':ycent}
			collection.kwargs.append(kwargs_apex)
			print ('%%%%%%%%%%%%%%%%%%%%%%%%')
			#sys.exit(0)

			##########################################################################

		if path_planck is not None:
			
			resolution = kwargs['resolution']

			if 'nilc' in kwargs:
				print ('Using NILC data')
				path_extension='y_maps_NILC/'
			else:
				print ('Using MILCA data')
				path_extension='y_maps/'

			data_p, header = ff.open_fits(path_planck, other=path_extension+str(name)+'.fits')[0:2]
				## y_maps_new
			try:
				pixindeg_planck = header['CDELT2'] #deg
			except:
				pixindeg_planck = 0.02500 #deg
				
			pixel_to_meters_p, pixeltoarcsec_p = fun.pix_con (z, pixindeg_planck)

			if 'width_p' not in kwargs:
				width_p = round(6*60/pixeltoarcsec_p)#bining in n arcmins and converting to pixels #0.2#6#0.2#*pixel_to_meters
			else:
				if type(kwargs['width_p']) is list:
					width_p = kwargs['width_p'][index]
				else:
					width_p = kwargs['width_p']

			#print('Width in arcsecs for Planck',  width_p*pixeltoarcsec_p)
			print('Width = %0.2f [pix]' %width_p)
			################## RMAX #######################################

			if 'rmax_a_arcsec' in kwargs:
				try:
					rmax_a_arcsec= kwargs['rmax_a_arcsec'][0][index]
					rmaxp_factor = kwargs['rmax_a_arcsec'][1]
				except:
					rmax_a_arcsec= kwargs['rmax_a_arcsec'][index]
					rmaxp_factor = 2

				rmax_p = int(round(rmaxp_factor*rmax_a_arcsec/pixeltoarcsec_p)) #planck pixels
				if rmax_p > data_p.shape[0]/2:
					rmax_p = data_p.shape[0]/2 -1
			elif 'rmax_p_arcsec' in kwargs:
				rmax_p = int(round(kwargs['rmax_p_arcsec'][index]/pixeltoarcsec_p))#planck pixels
			else:
				rmax_p = data_p.shape[0]/2 -1

			rmax_p_arcsec = rmax_p * pixeltoarcsec_p
			collection.rmax_selec[index] = (rmax_p_arcsec)
			
			print ('Rmax = %0.2f [pix]' %rmax_p)
			print ('Map Size Planck = %0.2f [arcmin]' %float(2*rmax_p_arcsec/60.))
			print ('Map Size Planck = %0.2f [R500]' %float(2*rmax_p_arcsec/r500))
			######################################################################

			######################## CREATE A MODEL  - PLANCK #######################
			r500 = None
			kwargs_planck = {'z':z,'pixindeg':pixindeg_planck,'resolution':resolution, 'rmax':rmax_p, 'm500':M500, 'm500err':M500_err, 'r500':r500, 'path':path_planck, 'noise_factor': noise_factor, 'name':name, 'width':width_p, 'num':num}


			#collection.kwargs.append(kwargs_planck)

			if simulation == True or high_simulation ==True:
				import random
				simnum_p = random.randint(0,999)
				
				if 'patches' in kwargs:
					kwargs_planck['patches'] = True
					simnum_p = np.random.randint(99)

				if 'random' in kwargs:
					print ('Random GNFW Parameters for Planck Simulations')
					
					ref_file = 'Results/multi_lowsn_sim7/Relaxed_w400_i1500_maf0.278_tau251.46_ndim4_PLANCK_newtfXL_lowsn_simulation7_GNFW_17multi.npz'
					ref_chain = np.load(ref_file)['chain'][:,0:ndim_p]
					cov_model = np.cov(ref_chain.T)
					mean = [A10[0],A10[1],A10[3]] if ndim_p ==3 else [A10[0],A10[1],A10[2],A10[3]]
					theta_mp = np.random.multivariate_normal (mean,cov_model)					
					print([float("%.2f"%item) for item in theta_mp])# for printing purposes
				if 'theta_fromfile' in kwargs:
					simnum_p = random.randint(0,99)
					from astropy.io import ascii
					print ('GNFW Parameters from %s' %str(kwargs['theta_fromfile']) )
					params_table = ascii.read(kwargs['theta_fromfile'])
					theta_mp = params_table[np.where(name==params_table['Name'])]['P0','c_500', 'alpha', 'beta']
					try:
						theta_mp = theta_mp[0][0], theta_mp[0][1], theta_mp[0][2], theta_mp[0][3]
					except:
						theta_mp = theta_m
					print (name, theta_mp)

					
				else:
					print ('A10 GNFW Parameters for Planck Simulations')
					print (theta_mp)


				
				print ('Simulation # Planck', simnum_p)
				kwargs_planck['simnum'] =  simnum_p

				#print (gNFW_params)

				data_p = apsim.sim2d(theta=theta_mp, ndim=ndim_p,  A10=A10, planck = True, zznoise= True, **kwargs_planck)[2]
					

			############################################################################
			#print ('data_p shape', data_p.shape)
			################ CUT PLANCK DATA TO DESIRED SIZE ################
			if simulation ==True or high_simulation == True:
				xcent, ycent = data_p.shape[0]/2, data_p.shape[1]/2
			else:
				try:
					header['CRVAL1']
					w= WCS(header)
					#print (ra, dec)
					# if ra != header['CRVAL1'] or dec != header['CRVAL2']:
					# 	print ('RA and Dec from list different than header. Using now RA and Dec from header')
					# 	ra, dec = header['CRVAL1'], header['CRVAL2']
					# else:
					print('Using the RA and Dec from list')
					xcent, ycent =w.wcs_world2pix(ra,dec,0)#fig.world2pixel(c[0],c[1])
					xcent, ycent  = int(round(xcent)), int(round(ycent))
					#print ('Forcing center to be the physical center of the map!!!!')
					#xcent, ycent = data_p.shape[0]/2, data_p.shape[1]/2
					# print (ra,dec)
					#print (xcent, ycent)
	 			except: 
					xcent, ycent = data_p.shape[0]/2, data_p.shape[1]/2
				data_p =data_p[ycent-rmax_p:ycent+rmax_p+1,xcent-rmax_p:xcent+rmax_p+1]
			# plt.figure()
			# plt.imshow(data_p)
			# pylab.pause(0.01)
			if 'tempmap' in kwargs:
				print ("Converting Planck's y-map to temperature map [mk CMB] observed at %0.02f GHz"%float(freq_obs/1e9))
				
				f_x_nu = fun.fxnu(freq_obs)
				tcmb = fun.const['tcmb']
				data_p = data_p*f_x_nu*tcmb*1e3 #temp map mK cmb
				kwargs_planck['freq_obs']=freq_obs
				kwargs_planck['tempmap']=True
			collection.data.append(data_p)

			############################################################################

			############################## NOISE COVARIANCE - PLANCK #################

			if 'cov_mod' in kwargs or 'cov_mod_ymap' in kwargs or 'stack' in kwargs:
				print ('Using modified covariance. Same covariance for all clusters ')
				try:
					path_cov = kwargs['cov_mod'] 
					print ('half ring covariance')
				except:
					path_cov = kwargs['stack']
					print ('stack covariance - half ring')
				else:
					path_cov = kwargs['cov_mod_ymap']
					print ('ymap - covariance')
				cov_p = ff.open_fits(path= path_cov, other = '')[0]
				bins= int(rmax_p / width_p) + (rmax_p % width_p > 0)
				cov_p = cov_p[0:bins, 0:bins]
				cov_i_p = np.linalg.inv(cov_p)
				error_p = np.sqrt(np.diag(cov_p))
			else:
				if 'patches' in kwargs:
					print ('Using cluster surroundings as noise maps')
					name_cov = name+'_patches'
				elif 'masky_noise' in kwargs:
					print ('Using noise  masked y-map same covariance for all')
					name_cov = 'ymapmilca_masked'
				elif 'hrall_noise' in kwargs:
					print ('Using noise  half-ring random maps same covariance for all')
					name_cov = kwargs['hrall_noise']
				else:
					name_cov = name
				cov_p, cov_i_p, error_p = cov.cov(name=name_cov, rmax=rmax_p, xcent=xcent, ycent=ycent, width=width_p, path=path_planck,noise_factor = noise_factor, planck=True, **kwargs )[0:3]

			collection.cov_selec.append(cov_i_p)
			#collection.cov_selec.append(error_p)
			############################################################################

			############################# RADIAL DATA - PLANCK #################
			
			#Radial profile of  real data
			r_rad_datap, meannan_rad_datap = rad.radialdata_applied(data_p, annulus_width=width_p, rmax=rmax_p)

			collection.meannan_rad.append(meannan_rad_datap)

			sn = abs(meannan_rad_datap/error_p)[0,0]
			kwargs_planck['sn'] =  sn

			print ('S/N - Planck= %0.2f '%sn)

			# if simulation == True:
			# 	print ('SIMULATION')
			# elif high_simulation ==True:
			# 	print ("High S/N Simulation")
			# else:
			# 	print ('Data')	
			kwargs_planck['xcent'] = xcent
			kwargs_planck['ycent'] = ycent
			collection.kwargs.append(kwargs_planck)
			print ('%%%%%%%%%%%%%%%%%%%%%%%%')
			############################################################################	
		
	return collection





# def combl (theta, data, apex=False, planck= False, parallel = False, **kwargs):
# 	#kwargs{ndim, A10} - #kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500, resolution

# 	import likelihood_functions as like 
# 	import multiprocessing as mu
# 	import numpy as np
# 	from joblib import Parallel, delayed
# 	import multiprocessing as mu


# 	reload(like)

# 	ndim = kwargs['ndim']
# 	#if 'ndim' in kwargs: del kwargs['ndim']
# 	A10 = kwargs['A10']
# 	#if 'A10' in kwargs: del kwargs['A10']

# 	njobs =mu.cpu_count()
	
# 	prob=[]

# 	prob = Parallel(verbose=1, n_jobs=njobs)(delayed(like.lnlike)( theta, data.meannan_rad[j], data.cov_selec[j], ndim, A10, apex=apex, planck= planck,**data.kwargs[j]) for j in np.arange (len(data.z_selec)))

	
# 	big_likelihood = np.sum(prob)

# 	return big_likelihood
	
