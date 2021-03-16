
import numpy as np 
import cosmo_cal as cc
import radial_data as rad
from scipy.integrate import quad, simps
from scipy.ndimage import interpolation
from scipy import signal
from scipy import ndimage
reload(cc)
reload(rad)

################ DICTIONARIES #######################

const={'cl':2.9979246*10**8,'G': 6.67408e-11, 'croos_section':6.6524e-29, 'me': 9.1093829e-31, 'e_charge':1.6021766e-19, 'H_0': 70, 'alpha_p':0.12, 'k_b': 1.38064852e-23, 'tcmb':2.73, 'h_planck':6.62607004e-34, 'k_b': 1.38064852e-23 } #units = cl:m, G:m^3/kgs^2, croos_section:m^2, me:kg, e_charge:C,  H_0:h Km s-1 Mpc -1, alpha_p:None,  k_b:m2 kg s-2 K-1, tcmb:K, h_planck:m2 kg / s, k_b:m2 kg s-2 K-1

conv={'mpc_to_meters':3.085677581e22, 'solarmtokg':1.9889200011445836e30, 'mktomicrok':1e3, 'conv_factor':1e6*1e3*const['e_charge'] } #conversion_factor converts from keV/cm^3 to J/m^3 = kg/(m*s^2)
################ FUNCTIONS #########################


def GNFW_params_file (theta, ndim, A10, sigmap, simulation =False):
	if theta == None:
		Pe0, c_500, a, b, c = A10[0:-1]
	else:	
		if ndim ==1:
			Pe0 = theta
			c_500, a, b, c = A10[1], A10[2], A10[3], A10[4]

		elif ndim ==2:
			Pe0, c_500 = theta
			a, b, c = A10[2], A10[3], A10[4]

		elif ndim ==3:
			Pe0, c_500, b = theta
			a,  c = A10[2],  A10[4]

		elif ndim ==5:
			Pe0, c_500, c, a, b = theta

	final_params = [Pe0, c_500, a, b, c, sigmap]
	if simulation == True:
		return np.save('GNFWsim_params', final_params)
	else:
		return np.save('GNFW_params', final_params)


def cosmo_dict (z, H_0=const['H_0']):

	cosmo_dist = cc.cosmo_cal(z=z, H_0=H_0)
	DA=cosmo_dist.da #Mpc
	Ez = cosmo_dist.ez
	h=H_0/100.
	dicti ={'DA':DA, 'Ez':Ez, 'h':h }
	return dicti

def rho_c ( z, H_0=const['H_0']):
	Ez=cosmo_dict(z, H_0)['Ez']
	rho_c =(3*H_0**2 *Ez**2)/(8*np.pi*const['G'])*(1e6/(conv['mpc_to_meters'])**2) #kg/m^3
	return rho_c

def pix_con (z, pixindeg, H_0=const['H_0']):
	"""
	Gives the converstion factor between pixels in degrees and pixels in meters and arcsec.

	Parameters
	-----------
	z: redshift
	pixindeg : -
	H_0: Default 70.

	Returns
	-------
	Pixel in meters
	Pixel in arcseconds
		"""
	DA=cosmo_dict(z, H_0)['DA']
	pixeltoarcsec = pixindeg*3600
	pixel_to_meters = pixindeg* np.pi / 180. * DA *conv['mpc_to_meters']
	return pixel_to_meters, pixeltoarcsec

def gnfw_profile (x, ndim, A10, theta, **kwargs ):

	if 'dclevel' in kwargs and ndim != 4:
		theta = theta[0:-1]
	if ndim ==1:
		Pe0 = theta
		c_500, a, b, c = A10[1], A10[2], A10[3], A10[4]

	elif ndim ==2:
		if 'betac500' in kwargs:
			c_500, b = theta
			Pe0,a,c = A10[0],A10[2],A10[4]
		else:
			Pe0, c_500 = theta
			a, b, c = A10[2], A10[3], A10[4]
		#print (Pe0, c_500, b)		

	elif ndim ==3:

		Pe0, c_500, b = theta
		a,  c = A10[2],  A10[4]
		if kwargs.get('m500fit'):
			Pe0 = A10[0]


	elif ndim ==4:
		if 'dclevel' in kwargs:
			Pe0, c_500, b, dc_level= theta
			a,  c = A10[2],  A10[4]
		else:
			Pe0, c_500, a, b= theta
			c = A10[4]

	elif ndim ==5:
		if 'fitcenter' in kwargs:
			Pe0, c_500, b, xc, yc = theta
			a,  c = A10[2],  A10[4]
		else:
			Pe0, c_500, c, a, b = theta

	#print (Pe0, c_500, a,b,c)
	return Pe0/((c_500*x)**c * (1+((c_500*x)**a))**((b-c)/a))

def alpha_prime(x, alpha_p=const['alpha_p']):
	return 0.10-(alpha_p+0.10)*((x/0.5)**3)/(1+(x/0.5)**3)

def h_z(z, H_0=const['H_0']):
	#H_0 = 70 #h Km s-1 Mpc -1
	Ez=cosmo_dict(z, H_0)['Ez']
	return (H_0*Ez)/H_0

def fxnu (freq_obs):
	h_planck= const['h_planck']
	tcmb = const['tcmb']
	k_b = const['k_b']
	x_freq =h_planck*freq_obs/(k_b*tcmb)
	f_x_nu = x_freq *((np.exp(x_freq) +1)/(np.exp(x_freq) -1))- 4
	return f_x_nu

def mr500 (z, m500=None, r500=None, H_0=const['H_0'], other_units=False, **kwargs):
	
	"""Input:
			z
			M500 in Solar masses/1e14 
			r500 in arcsec [default]. If r500 in kpc or Mpc, then other_units=True and include kwargs['unit'] with the corresponding unit.
		Returns:
			M500 in solar masses
			r500 in meters
			r500 in arcsecs"""

	rhoc =rho_c(z)
	DA=cosmo_dict(z, H_0)['DA']
	#print (DA, rhoc)

	if m500 is not None:
		M500 = m500 *1e14 #solar masses
		M500_kg = M500*conv['solarmtokg']
		r500_m = ((M500_kg *3)/(500*np.pi*rhoc*4))**(1/3.)# m 
		r500 = r500_m/conv['mpc_to_meters'] /DA * 180/np.pi *3600 #arcsec

	if r500 is not None:
		if other_units:
			if kwargs['unit']=='kpc':
				r500_mpc= r500/1e3

			elif kwargs['unit']=='Mpc':
				r500_mpc = r500

			r500= r500_mpc/DA *180/np.pi *3600 #arcsec
			
			
		else: #if r500 given in arcsecs
			r500_mpc = r500/3600. * np.pi/180.*DA #Mpc 

		r500_m = r500_mpc*conv['mpc_to_meters']
		M500_kg = r500_m**3 * 500*np.pi*rhoc*4/3. #kg
		M500 = M500_kg/conv['solarmtokg'] #solar masses

	return M500, r500_m, r500


def p_profile(theta,  ndim, A10, z, pixindeg, m500,r, H_0=const['H_0'], **kwargs ):
	if ndim ==1:
		Pe0 = theta
	elif ndim ==2:
		if 'betac500' in kwargs:
			c_500, b = theta
		else:
			Pe0, c_500 = theta
	elif ndim==3:
		Pe0, c_500, b = theta
		if 'm500fit' in kwargs:
			Pe0=A10[0]
	elif ndim ==4:
		Pe0, c_500, a, b = theta

	conv_factor = conv['conv_factor']
	alpha_p=const['alpha_p']
	h=cosmo_dict(z, H_0)['h']
	#pixel_to_meters = pix_con(z, pixindeg)[0]

	if kwargs.get('m500fit'):
		m500  = theta[0]
		#print ('p', m500)
		M500, r500_m = mr500(z,m500=m500, r500=None, H_0=H_0) [0:2]

	else:
		M500, r500_m, r500_arcsec = mr500(z,m500=m500, r500=None, H_0=H_0) 

	#print (M500/1e14, r500_arcsec/60.)
	#M500, r500_m = mr500(z,m500=m500, r500=None, H_0=H_0) [0:2]

	#rg =np.linspace(0.01*r500_m/pixel_to_meters,1.5*r500_m/pixel_to_meters,200, endpoint=True)*pixel_to_meters
	#rg =np.linspace(0.01*r500_m,1.5*r500_m,200, endpoint=True)

	#rg =np.logspace(20,23,200, endpoint=True)
	rg = r

	P500_a =(1.65*10**-3)*h_z(z)**(8/3.)*(M500/(0.7/h*3e14))**(2/3.+alpha_p) *(h/0.7)**2 *conv_factor
	profile = P500_a*gnfw_profile(np.sqrt(rg**2)/r500_m, ndim, A10, theta, **kwargs)

	#profile = P500_a*((M500/(0.7/h*3e14))**(alpha_prime(np.sqrt(rg**2)/r500_m))*gnfw_profile(np.sqrt(rg**2)/r500_m, ndim, A10, theta, **kwargs))
	
	

	profile_norm = profile/(P500_a)
	profile = profile/conv_factor

	return profile, profile_norm


def gnfw_function_apex (theta, r, ndim, trans_fun, A10, z, pixindeg, freq_obs, rmax, m500, r500, H_0=const['H_0'], y_prior=False, **kwargs  ):

	if 'dclevel' in kwargs:
		dc_level = theta[-1]

	# if ndim ==1:
	# 	Pe0 = theta
	# elif ndim ==2:
	# 	Pe0, c_500 = theta
	# elif ndim==3:
	# 	Pe0, c_500, b = theta
	# elif ndim ==4:
	# 	Pe0, c_500, b, dc_level= theta
	# elif ndim ==5:
	# 	Pe0, c_500, c, a, b = theta



 	################## DEFINING VARIABLES ##################	

	pixel_to_meters, pixeltoarcsec = pix_con(z, pixindeg) #[0]
	h=cosmo_dict(z, H_0)['h']
	alpha_p=const['alpha_p']
	conv_factor = conv['conv_factor']
	croos_section = const['croos_section']
	me = const['me']
	cl = const['cl']
	tcmb = const['tcmb']
	f_x_nu = fxnu(freq_obs)
	alphap_term = kwargs.get('alphap', False)

	if kwargs.get('m500fit'):

		m500  = theta[0]
		#print ('a', m500)
		M500, r500_m = mr500(z,m500=m500, r500=None, H_0=H_0) [0:2]

	else:
		M500, r500_m = mr500(z,m500=m500, r500=r500, H_0=H_0) [0:2]
	#print ('a', M500, r500_m)
	low_lim=1e-3*r500_m#0
	up_lim=5*r500_m

	################ 1D distance ####################

	rg =np.logspace(0,2.5,500, endpoint=True)*pixel_to_meters
	rg[0]=0#10e19 #must be 0. Otherwise the central value will correspond to 1*pixel_to_meters
	shape0 = rg.shape[0]
	
	model=np.zeros(rg.shape)

	#print ("M500_a = ", M500)
	###################################################################

	####################### M500 Unceratinties ####################

	# if 'm500err' in kwargs:
	# 	err = kwargs['m500err']
	# 	M500_rand = np.random.normal(M500, err*1e14)
	# 	P500_a =(1.65*10**-3)*h_z(z)**(8/3.)*(M500_rand/(0.7/h*3e14))**(2/3.+alpha_p) *(h/0.7)**2 *conv_factor
	# else:
	# 	P500_a =(1.65*10**-3)*h_z(z)**(8/3.)*(M500/(0.7/h*3e14))**(2/3.+alpha_p) *(h/0.7)**2 *conv_factor

	
	#################### Integral 1D ####################

	P500_a =(1.65*10**-3)*h_z(z)**(8/3.)*(M500/(0.7/h*3e14))**(2/3.+alpha_p) *(h/0.7)**2 *conv_factor

	#for i in range(shape0):
		# model[i]= P500_a*quad(lambda z:((M500/(0.7/h*3e14))**(alpha_prime(np.sqrt(rg[i]**2 + z**2)/r500_m))*gnfw_profile(np.sqrt(rg[i]**2 + z**2)/r500_m, ndim, A10, theta)),low_lim,up_lim)[0]
	# for i in range(shape0): #without alpha_pp - only for sim7
	# 	model[i]= P500_a*quad(lambda z:(gnfw_profile(np.sqrt(rg[i]**2 + z**2)/r500_m, ndim, A10, theta)),low_lim,up_lim)[0]
		

	for i, yy in enumerate(rg):
		if yy <= up_lim:
		 	if yy < low_lim: 
		 		x_min = low_lim 
		 	else: 
		 		x_min = 0 
		 	#x = np.linspace(x_min,np.sqrt(up_lim**2-yy**2),200)
			x= np.linspace(x_min,up_lim,370)
		 	z= np.sqrt(yy**2. + x**2.) 
		if alphap_term:
			model[i]=P500_a*simps((M500/(0.7/h*3e14))**(alpha_prime(z/r500_m))*gnfw_profile(z/r500_m, ndim, A10, theta, **kwargs), x)
		else:
	 		model[i]=P500_a*simps(gnfw_profile(z/r500_m, ndim, A10, theta, **kwargs), x) #without alpha_pp - only for sim7
	 	#


	if y_prior ==True:
		mpc_to_meters = conv['mpc_to_meters']
		if alphap_term:
			Y_SPH=quad(lambda r:(P500_a*(M500/(0.7/h*3*10**14))**(alpha_prime(r/r500_m))*r**2*gnfw_profile(r/r500_m, ndim,A10, theta, **kwargs)),0,r500_m)[0] #m^2
		else:
			Y_SPH=quad(lambda r:(P500_a*r**2*gnfw_profile(r/r500_m, ndim,A10, theta, **kwargs)),0,r500_m)[0] #m^2
		# x= np.linspace(1e-5*r500_m,r500_m,500)
		# z= np.sqrt(rg**2. + x**2.) 
		# Y_SPH=(P500_a*simps((M500/(0.7/h*3*10**14))**(alpha_prime(z/r500_m))*z**2*gnfw_profile(z/r500_m, ndim,A10, theta))) #m^2
		#print (Y_SPH)
		
		Y_SPH= (4*np.pi*croos_section/(me*cl**2))*Y_SPH #m^2
		Y_SPH_mpc = Y_SPH/mpc_to_meters**2 #Mpc^2
		#Y_SPH = Y_SPH_mpc/(D_A**2) *(180/np.pi *60)**2 #arcmin^2
		Y_vals = Y_SPH_mpc/1e-5
		
		

	
			
	model = (2*croos_section/(me*cl**2))*model
	model = model*f_x_nu*tcmb*1e3 #temp map mK cmb
	
	################ CONVERT MODEL FROM 1D TO 2D #########################

	extrapol = np.array([rg.reshape(-1), model.reshape(-1)])
	extrapol= np.sort(extrapol.T, axis=0)
	model= np.interp(r,extrapol[:,0], extrapol[:,1])
		
	############### CONVOLVED MODEL ###################

	if 'dclevel' in kwargs:
		#print (dc_level)
		model = model + dc_level

	if 'smoothing' in kwargs:
		#print ('smoothing with gaussian kernel')
		if 'smoothing' == True:
			beam_size = 60
		else:
			beam_size = kwargs.get('smoothing')
		
		if beam_size != 0:
			d = beam_size/(pixeltoarcsec*2.35482)
			model_con=ndimage.filters.gaussian_filter(model, d, mode='constant')
		else:
			model_con=model

	elif 'smoothing05' in kwargs:
		#print ('smoothing with gaussian kernel')
		d = 30/(pixeltoarcsec*2.35482)
		model_con=ndimage.filters.gaussian_filter(model, d, mode='constant')
	else:
		model_con = signal.fftconvolve(model,trans_fun, mode='same')

	################ CUT MODEL TO DESIRED SIZE ###############
	if 'fitcenter' in kwargs:
		shape0, shape1 = [int(round(i)) for i in theta_m[-2:]]
	else:
		shape0 = model.shape[0]//2
		shape1 = model.shape[0]//2

	model_con =model_con[shape1-rmax:shape1+rmax+1,shape0-rmax:shape0+rmax+1]
	model =model[shape1-rmax:shape1+rmax+1,shape0-rmax:shape0+rmax+1]
	if y_prior == True:
		return (model, model_con, Y_vals)
	else:
		return(model, model_con)


def gnfw_function_planck (theta, r, ndim, A10, z, pixindeg, resolution,  m500, r500, H_0=const['H_0'], **kwargs  ):

	from scipy import interpolate
	
	#if ndim ==1:
	#	Pe0 = theta
	#elif ndim ==2:
	#	Pe0, c_500 = theta
	#elif ndim==3:
	#	Pe0, c_500, b = theta
	#elif ndim ==4:
	#	Pe0, c_500, b, dc_level= theta
	#elif ndim ==5:
	#	Pe0, c_500, c, a, b = theta

 	################## DEFINING VARIABLES ##################	

	pixel_to_meters, pixeltoarcsec = pix_con(z, pixindeg)
	h=cosmo_dict(z, H_0)['h']
	alpha_p=const['alpha_p']
	conv_factor = conv['conv_factor']
	croos_section = const['croos_section']
	me = const['me']
	cl = const['cl']


	if kwargs.get('m500fit'):
		m500  = theta[0]
		#print ('p', m500)
		M500, r500_m = mr500(z,m500=m500, r500=None, H_0=H_0) [0:2]

	else:
		M500, r500_m = mr500(z,m500=m500, r500=r500, H_0=H_0) [0:2]
	#print ('p', M500, r500_m)
	low_lim=1e-3*r500_m#0
	up_lim=6*r500_m
	d = resolution/(pixeltoarcsec*2.35482)


	################ 1D distance ####################

	rg =np.logspace(0,2.5,700, endpoint=True)*pixel_to_meters

	#rg =np.logspace(0,2.5,500, endpoint=True)*pixel_to_meters
	rg[0]=0
	shape0 = rg.shape[0]
	model=np.zeros(rg.shape)

	#print ("M500_p = ", M500)
	#################### Integral 1D ####################

	
	P500_a =(1.65*10**-3)*h_z(z)**(8/3.)*(M500/(0.7/h*3e14))**(2/3.+alpha_p) *(h/0.7)**2 *conv_factor
	# for i in range(shape0):
	# 	model[i]= P500_a*quad(lambda z:((M500/(0.7/h*3e14))**(alpha_prime(np.sqrt(rg[i]**2 + z**2)/r500_m))*gnfw_profile(np.sqrt(rg[i]**2 + z**2)/r500_m, ndim, A10, theta, **kwargs)),low_lim,up_lim)[0]
	# for i in range(shape0): #without alpha_pp
	# 	model[i]= P500_a*quad(lambda z:(gnfw_profile(np.sqrt(rg[i]**2 + z**2)/r500_m, ndim, A10,theta, **kwargs)),low_lim,up_lim)[0]


	for i, yy in enumerate(rg):
		if yy <= up_lim:
		 	if yy < low_lim: 
		 		x_min = low_lim 
		 	else: 
		 		x_min = 0 
		 	#x = np.linspace(x_min,np.sqrt(up_lim**2-yy**2),200)
			x= np.linspace(x_min,up_lim,370)
		 	z= np.sqrt(yy**2. + x**2.) 
		#model[i]=simps(gnfw_profile(z/r500_m, ndim, A10, theta, **kwargs), x) #without alpha_pp and P500 
		model[i]=P500_a*simps(gnfw_profile(z/r500_m, ndim, A10, theta, **kwargs), x) #without alpha_pp - only for sim7
	 	#model[i]=P500_a*simps((M500/(0.7/h*3e14))**(alpha_prime(z/r500_m))*gnfw_profile(z/r500_m, ndim, A10, theta, **kwargs), x)

	
			
	model = (2*croos_section/(me*cl**2))*model

	if 'tempmap' in kwargs:
		tcmb = const['tcmb']
		f_x_nu = fxnu(kwargs.get('freq_obs', 150*10**9))
		model = model*f_x_nu*tcmb*1e3 #temp map mK cmb


	################ CONVERT MODEL FROM 1D TO 2D #########################

	f = interpolate.interp1d(rg.reshape(-1), model.reshape(-1))
	model = f(r.reshape(-1)) #this works for linear log 

	model = model.reshape(r.shape) #
	############### CONVOLVED MODEL ###################

	model_con = ndimage.filters.gaussian_filter(model, d, mode='constant')

	return(model, model_con)


def calc_Y(pos, A10, ndim, Y_true=None, H_0=const['H_0'], **kwargs):
	"""Input:
			pos: values for the GNFW parameters at which to evaluate Y
			A10: Preassure profile values to be assumed constant
			ndim: Number of parameters fit
			Y_true: True value of Y if known. Default None.
			H_0: Default (70)
			**kwargs:z, m500, r500, uplim (in m, default r500_m) 
		Output: 
			new_pos: values of the GNFW parameters at which Y was evaluated. Same as the input pos plus the Y_sph_R500 value for each position. [Y_sph units --> Mpc^2/1e-5]
			y_mcmc: mode of the Y_sph_R500 distribution if len(pos) >1. [Y_sph units --> Mpc^2/1e-5]
			results: mode value of the distribution of pos and Y_sph_R500 and the 68% error - calculated with the most compact 68%. [Y_sph units --> Mpc^2/1e-5]
			"""
	print ('Calculating Y')

	# if 'dclevel' in kwargs:
	# 	pos = pos [:,0:-1]

	#kwargs = z, m500, r500

	#Y_true = 29.423#12.18

	################## DEFINING VARIABLES ##################	
	m500 = kwargs.pop('m500', None) #in solar masses/1e14
	r500 = kwargs.pop('r500', None) #in arcsec
	z = kwargs.pop('z', None)
	mpc_to_meters = conv['mpc_to_meters']
	h=cosmo_dict(z, H_0)['h']
	alpha_p=const['alpha_p']
	conv_factor = conv['conv_factor']
	croos_section = const['croos_section']
	me = const['me']
	cl = const['cl']


	M500, r500_m = mr500(z,m500=m500, r500=r500, H_0=H_0, **kwargs) [0:2]
	low_lim=0
	up_lim = kwargs.get('uplim',r500_m)
	
	#print (M500)

	if type(pos) is tuple:
		pos = np.array([pos])
		
	Y_vals = np.zeros(shape=(pos.shape[0],1))

	for ii in range(pos.shape[0]):
		theta = abs(pos[ii,:])
		if 'm500fit' in kwargs:
			M500, r500_m = mr500(z,m500=theta[0], r500=None, H_0=H_0, **kwargs) [0:2]


		P500_a =(1.65*10**-3)*h_z(z)**(8/3.)*(M500/(0.7/h*3e14))**(2/3.+alpha_p) *(h/0.7)**2 *conv_factor
		Y_SPH=quad(lambda r:(P500_a*(M500/(0.7/h*3*10**14))**(alpha_prime(r/r500_m))*r**2*gnfw_profile(r/r500_m, ndim,A10, theta, **kwargs)),low_lim,r500_m)[0] #m^2
		

		Y_SPH= (4*np.pi*croos_section/(me*cl**2))*Y_SPH #m^2
		Y_SPH_mpc = Y_SPH/mpc_to_meters**2 #Mpc^2
		#Y_SPH = Y_SPH_mpc/(D_A**2) *(180/np.pi *60)**2 #arcmin^2
		Y_vals[ii,0] = Y_SPH_mpc/1e-5

	new_pos = np.column_stack((pos, Y_vals))

	#y_mcmc, results = peak_unc_cal(new_pos, A10, ndim, Y_true, verbose=False, H_0=const['H_0'], **kwargs)
	y_mcmc, results = mean_unc_cal(new_pos, A10, ndim, Y_true, H_0=const['H_0'], **kwargs)
	return new_pos, y_mcmc, results


def mean_unc_cal(pos, A10, ndim, Y_true, H_0=const['H_0'], **kwargs):
	pos = abs(pos)

	
	if ndim ==1:
		if 'dclevel' in kwargs:
			# Compute the quantiles.
			Pei_mcmc, dc_mcmc, y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
									 zip(*np.percentile(pos, [16, 50, 84],
		 												axis=0))) #, a_mcmc, b_mcmc, c_mcmc
			print("""MCMC result:
			Pei = {0[0]} +{0[1]} -{0[2]} (truth: {1})
			DC_Level = {2[0]} +{2[1]} -{2[2]} (truth: {3})
			Y_sph(R500) (10^-5 Mpc^2) = {4[0]} + {4[1]} -{4[2]} (truth:{5})
			""".format(Pei_mcmc, A10[0], dc_mcmc, 0, y_mcmc, Y_true))
			results = np.array([Pei_mcmc, dc_mcmc, y_mcmc ])

		else:
			# Compute the quantiles.
			Pei_mcmc, y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
									 zip(*np.percentile(pos, [16, 50, 84],
		 												axis=0))) #, a_mcmc, b_mcmc, c_mcmc
			print("""MCMC result:
			Pei = {0[0]} +{0[1]} -{0[2]} (truth: {1})
			Y_sph(R500) (10^-5 Mpc^2) = {2[0]} + {2[1]} -{2[2]} (truth:{3})
			""".format(Pei_mcmc, A10[0], y_mcmc, Y_true))
			results = np.array([Pei_mcmc, y_mcmc ])

	elif ndim ==2:
		if 'dclevel' in kwargs:
			# Compute the quantiles.
			Pei_mcmc,c500_mcmc, dc_mcmc, y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
										 zip(*np.percentile(pos, [16, 50, 84],
															axis=0))) 

			if 'betac500' in kwargs:
				print("""MCMC result:
			c_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
			beta = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
			DC_level = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
			Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+ {6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
			""".format(Pei_mcmc, A10[1] , c500_mcmc, A10[3], dc_mcmc, 0, y_mcmc, Y_true))
			else:

				print("""MCMC result:
				Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				DC_level = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
				Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+ {6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
				""".format(Pei_mcmc, A10[0] , c500_mcmc, A10[1], dc_mcmc, 0, y_mcmc, Y_true))
			
			results = np.array([Pei_mcmc,c500_mcmc, dc_mcmc, y_mcmc ])
		else:
			
			# Compute the quantiles.
			Pei_mcmc,c500_mcmc, y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
										 zip(*np.percentile(pos, [16, 50, 84],
															axis=0))) 

			if 'betac500' in kwargs:
				print("""MCMC result:
			c_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
			beta = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
			Y_sph(R500) (10^-5 Mpc^2) = {4[0]:0.2f} ^{{+ {4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth:{5})
			""".format(Pei_mcmc, A10[1] , c500_mcmc, A10[3], y_mcmc, Y_true))
			else:

				print("""MCMC result:
				Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				Y_sph(R500) (10^-5 Mpc^2) = {4[0]:0.2f} ^{{+ {4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth:{5})
				""".format(Pei_mcmc, A10[0] , c500_mcmc, A10[1], y_mcmc, Y_true))
			
			results = np.array([Pei_mcmc,c500_mcmc, y_mcmc ])

	elif ndim ==3:
		if 'dclevel' in kwargs:


				# Compute the quantiles.
			Pei_mcmc,c500_mcmc, b_mcmc, dc_mcmc, y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
										 zip(*np.percentile(pos, [16, 50, 84],
															axis=0))) 
			if 'm500fit' in kwargs:
				print("""MCMC result:
				M_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
				DC_level = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth: {7})
				Y_sph(R500) (10^-5 Mpc^2) = {8[0]:0.2f} ^{{+{8[1]:0.2f}}} _{{-{8[2]:0.2f}}} (truth:{9})
				""".format(Pei_mcmc, kwargs['m500'] , c500_mcmc, A10[1], b_mcmc, A10[3],dc_mcmc, 0, y_mcmc, Y_true)) 
				

			else:
				print("""MCMC result:
				Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
				DC_level = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth: {7})
				Y_sph(R500) (10^-5 Mpc^2) = {8[0]:0.2f} ^{{+{8[1]:0.2f}}} _{{-{8[2]:0.2f}}} (truth:{9})
				""".format(Pei_mcmc, A10[0] , c500_mcmc, A10[1], b_mcmc, A10[3], dc_mcmc, 0, y_mcmc, Y_true)) 
			results = np.array([Pei_mcmc,c500_mcmc,  b_mcmc, dc_mcmc, y_mcmc ])

		else:

			
				# Compute the quantiles.
			Pei_mcmc,c500_mcmc, b_mcmc, y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
										 zip(*np.percentile(pos, [16, 50, 84],
															axis=0))) 
			if 'm500fit' in kwargs:
				print("""MCMC result:
				M_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
				Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
				""".format(Pei_mcmc, kwargs['m500'] , c500_mcmc, A10[1], b_mcmc, A10[3], y_mcmc, Y_true)) 
				

			else:
				print("""MCMC result:
				Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
				Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
				""".format(Pei_mcmc, A10[0] , c500_mcmc, A10[1], b_mcmc, A10[3], y_mcmc, Y_true)) 
			results = np.array([Pei_mcmc,c500_mcmc,  b_mcmc, y_mcmc ])

	elif ndim ==4:
			
			# Compute the quantiles.
			Pei_mcmc,c500_mcmc, b_mcmc, dc_mcmc, y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
										 zip(*np.percentile(pos, [16, 50, 84],
															axis=0))) 
			if 'm500fit' in kwargs:
				print("""MCMC result:
				M_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
				dc_level = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth: {7})
				Y_sph(R500) (10^-5 Mpc^2) = {8[0]:0.2f} ^{{+{8[1]:0.2f}}} _{{-{8[2]:0.2f}}} (truth:{9})
				""".format(Pei_mcmc, kwargs['m500'] , c500_mcmc, A10[1], b_mcmc, A10[3],dc_mcmc, 0,  y_mcmc, Y_true)) 
			else:
				print("""MCMC result:
				Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
				dc_level = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth: {7})
				Y_sph(R500) (10^-5 Mpc^2) = {8[0]:0.2f} ^{{+{8[1]:0.2f}}} _{{-{8[2]:0.2f}}} (truth:{9})
				""".format(Pei_mcmc, A10[0] , c500_mcmc, A10[1], b_mcmc, A10[3],dc_mcmc, 0,  y_mcmc, Y_true)) 
			results = np.array([Pei_mcmc,c500_mcmc,  b_mcmc, dc_mcmc, y_mcmc ])

	elif ndim ==5:
		if 'dclevel' in kwargs:
				# Compute the quantiles.
			Pei_mcmc,c500_mcmc, c_mcmc, a_mcmc, b_mcmc, dc_mcmc,  y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
										 zip(*np.percentile(pos, [16, 50, 84],
															axis=0))) 
			print("""MCMC result:
			Pei = {0[0]} +{0[1]} -{0[2]} (truth: {1})
			c_500 = {2[0]} +{2[1]} -{2[2]} (truth: {3})
			c = {4[0]} +{4[1]} -{4[2]} (truth: {5})
			a= {6[0]} +{6[1]} -{6[2]} (truth: {7})
			b = {8[0]} +{8[1]} -{8[2]} (truth: {9})
			DC_level = {10[0]} +{10[1]} -{10[2]} (truth: {11})
			Y_sph(R500) (10^-5 Mpc^2) = {12[0]} + {12[1]} -{12[2]} (truth:{13})
			""".format(Pei_mcmc, A10[0] , c500_mcmc, A10[1], c_mcmc, A10[4], a_mcmc, A10[2], b_mcmc, A10[3], dc_mcmc,0, y_mcmc, Y_true)) 
			results = np.array([Pei_mcmc,c500_mcmc, c_mcmc, a_mcmc, b_mcmc,dc_mcmc, y_mcmc ])
		if 'fitcenter' in kwargs:
			xc, yc = kwargs['fitcenter']
			# Compute the quantiles.
			Pei_mcmc,c500_mcmc, b_mcmc, xc_mcmc, yc_mcmc, y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
										 zip(*np.percentile(pos, [16, 50, 84],
															axis=0))) 
			print("""MCMC result:
			Pei = {0[0]} +{0[1]} -{0[2]} (truth: {1})
			c_500 = {2[0]} +{2[1]} -{2[2]} (truth: {3})
			b = {4[0]} +{4[1]} -{4[2]} (truth: {5})
			xc= {6[0]} +{6[1]} -{6[2]} (truth: {7})
			yc = {8[0]} +{8[1]} -{8[2]} (truth: {9})
			Y_sph(R500) (10^-5 Mpc^2) = {10[0]} + {10[1]} -{10[2]} (truth:{11})
			""".format(Pei_mcmc, A10[0] , c500_mcmc, A10[1], b_mcmc, A10[3], xc_mcmc, xc, yc_mcmc, yc, y_mcmc, Y_true)) 
			results = np.array([Pei_mcmc,c500_mcmc, b_mcmc, xc_mcmc, yc_mcmc, y_mcmc ])

		else:
			# Compute the quantiles.
			Pei_mcmc,c500_mcmc, c_mcmc, a_mcmc, b_mcmc, y_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
										 zip(*np.percentile(pos, [16, 50, 84],
															axis=0))) 
			print("""MCMC result:
			Pei = {0[0]} +{0[1]} -{0[2]} (truth: {1})
			c_500 = {2[0]} +{2[1]} -{2[2]} (truth: {3})
			c = {4[0]} +{4[1]} -{4[2]} (truth: {5})
			a= {6[0]} +{6[1]} -{6[2]} (truth: {7})
			b = {8[0]} +{8[1]} -{8[2]} (truth: {9})
			Y_sph(R500) (10^-5 Mpc^2) = {10[0]} + {10[1]} -{10[2]} (truth:{11})
			""".format(Pei_mcmc, A10[0] , c500_mcmc, A10[1], c_mcmc, A10[4], a_mcmc, A10[2], b_mcmc, A10[3], y_mcmc, Y_true)) 
			results = np.array([Pei_mcmc,c500_mcmc, c_mcmc, a_mcmc, b_mcmc, y_mcmc ])


	return y_mcmc[0], results


def peak_unc_cal(pos, A10, ndim, Y_true, verbose=False, H_0=const['H_0'], **kwargs):

	print ('Peak uncertainty and Most compact 68%')

	pos = abs(pos)
	nb = 45#int(round(len(pos)**(1/3.)))*10



	chain = np.sort(pos, axis=0)
	lenchain = len(chain)
	chainshape1 = chain.shape[1]
	seper = int(0.68*len(chain))
	results = np.zeros(shape=(ndim+1, 3))

	for i in range (ndim+1):
		up_old = chain[seper, i]
		low_old = chain[0, i]
		range_old = chain[seper, i] -chain[0, i] 

		for ii in range(int(0.32*len(chain))):
			up = chain[seper+ii, i]
			low = chain[ii, i]
			ran =  up -low

			if ran < range_old:
				range_old = ran
				up_old = up
				low_old = low

			# if ran > range_old:
			# 	range_old = range_old
			# 	up_old = up_old
			# 	low_old = low_old

		index = np.where(chain[:,i] == low_old)[0][0]
		y, bins = np.histogram(chain[index:seper+index, i], bins=nb,  density=True)
		bins_width = bins[0:-1] + abs(bins[0:-1]-bins[1::])/2
		maximun_old = bins_width[y.argmax()]
		results[i,:] = maximun_old, up_old -maximun_old, maximun_old - low_old
		

	y_mcmc = maximun_old, up_old -maximun_old, maximun_old - low_old
	if verbose:
		if ndim ==1:
			print("""MCMC result:
				Pei = {0[0]} +{0[1]} -{0[2]} (truth: {1})
				Y_sph(R500) (10^-5 Mpc^2) = {2[0]} + {2[1]} -{2[2]} (truth:{3})
				""".format(results[0], A10[0], results[1], Y_true))
			

		if ndim ==2:
			if 'betac500' in kwargs:
				print("""MCMC result:
					c_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
					beta = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
					Y_sph(R500) (10^-5 Mpc^2) = {4[0]:0.2f} ^{{+ {4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth:{5})
					""".format(results[0], A10[1] ,results[1], A10[3],results[2], Y_true))
			else:
				print("""MCMC result:
				Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				Y_sph(R500) (10^-5 Mpc^2) = {4[0]:0.2f} ^{{+ {4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth:{5})
				""".format(results[0], A10[0] ,results[1], A10[1],results[2], Y_true))
		if ndim ==3:
			if 'm500fit' in kwargs:
				print("""MCMC result:
				M_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
				Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
				""".format(results[0], kwargs['m500'] , results[1], A10[1],results[2], A10[3], results[3], Y_true)) 
				
			else:
				print("""MCMC result:
				Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
				Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
				""".format(results[0], A10[0] , results[1], A10[1],results[2], A10[3], results[3], Y_true)) 

	return y_mcmc, results


def peak_unc_cal_eqp(pos, A10, ndim, Y_true, H_0=const['H_0'], **kwargs):

	pos = abs(pos)

	from scipy.interpolate import interp1d
	from scipy.integrate import trapz
	import numpy as np
	import pylab
	import matplotlib.pyplot as plt

	nb = int(round(len(pos)**(1/3.)))#*5
	diff_int_old = np.inf
	results = np.zeros(shape=(pos.shape[1], 3))

	def hline(m,x,b):
		return m*x+b


	for i in range(pos.shape[1]):
		diff_int_old = np.inf
		int_lim_final = np.inf
		values, bins = np.histogram(pos[:,i], bins=nb, density=True)
		values = np.hstack((0,values,0))
		binsm = ((bins[1::] - bins[:-1])/2) + bins[:-1]
		binsm = np.hstack((bins[0] -(bins[-1] - bins[-2]) ,binsm, bins[-1]+(bins[-1] - bins[-2])))
		fh = interp1d( binsm, values)

		x = np.linspace(binsm.min(),binsm.max(), 1e4)
		y = fh(x)
		int_tot = round(trapz(y,x))
		binsm = x
		values = y 
		maximun = binsm[np.where(values==values.max())[0][0]]
		yy = np.linspace(values.min(),values.max(), 1e4)
		yy = np.linspace(0,values.max(), 1e4)
		# if i==1:
		# 	plt.figure()
		# 	plt.plot(binsm,values)
		for val in yy:
			f = hline(0,binsm,val)
			idx = np.argwhere(np.diff(np.sign(f-values)) != 0)
			d = binsm[idx]
			#print (d)
			#x = np.linspace(min(d),max(d),1e4)
			x = np.linspace(binsm[idx[0]],binsm[idx[-1]],1e4) #original one
			y =fh(x) #np.interp(x, binsm, values)
			int_lim = (trapz(y,x))
			# if i ==1:
			# 	plt.axhline(val, c='grey', alpha=0.5)
			# 	plt.plot(x,y)
			# 	pylab.pause(0.1)
			if 0.675<int_lim <0.685:
			#if int_lim <0.6827:
				#print (int_lim, val)
				diff_int = abs(abs(int_tot -int_lim) - 0.32)
				if diff_int < diff_int_old and binsm[idx[0]] < maximun and binsm[idx[-1]] > maximun:
					val_min = val
					lim_low, lim_up = binsm[idx[0]],binsm[idx[-1]]
					diff_int_old = diff_int
					int_lim_final = int_lim
				#print (int_lim_final)
			if int_lim_final == np.inf:
				lim_low, lim_up = np.nan, np.nan
				int_lim_final = np.nan
		print (int_lim_final, lim_low, lim_up)
		results[i,:] = maximun, lim_up -maximun, maximun - lim_low
	y_mcmc = maximun, lim_up -maximun, maximun - lim_low

	if ndim ==1:
		print("""MCMC result:
			Pei = {0[0]} +{0[1]} -{0[2]} (truth: {1})
			Y_sph(R500) (10^-5 Mpc^2) = {2[0]} + {2[1]} -{2[2]} (truth:{3})
			""".format(results[0], A10[0], results[1], Y_true))

		

	if ndim ==2:
		if 'betac500' in kwargs:
			print("""MCMC result:
				c_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				beta = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				Y_sph(R500) (10^-5 Mpc^2) = {4[0]:0.2f} ^{{+ {4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth:{5})
				""".format(results[0], A10[1] ,results[1], A10[3],results[2], Y_true))
		else:
			print("""MCMC result:
				Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				Y_sph(R500) (10^-5 Mpc^2) = {4[0]:0.2f} ^{{+ {4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth:{5})
				""".format(results[0], A10[0] ,results[1], A10[1],results[2], Y_true))
	if ndim ==3:
		if 'm500fit' in kwargs:
			print("""MCMC result:
			M_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
			c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
			b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
			Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
			""".format(results[0], kwargs['m500'] , results[1], A10[1],results[2], A10[3], results[3], Y_true)) 
		else:
			print("""MCMC result:
			Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
			c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
			b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
			Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
			""".format(results[0], A10[0] , results[1], A10[1],results[2], A10[3], results[3], Y_true)) 

	return y_mcmc, results

def peak_unc_cal_eqp2(pos, A10, ndim, Y_true, H_0=const['H_0'], **kwargs):

	pos = abs(pos)

	from scipy.interpolate import interp1d
	from scipy.integrate import trapz
	import numpy as np
	import pylab
	import matplotlib.pyplot as plt

	nb = int(round(len(pos)**(1/3.)))*5
	diff_int_old = np.inf
	results = np.zeros(shape=(pos.shape[1], 3))

	def hline(m,x,b):
		return m*x+b


	for i in range(pos.shape[1]):
		diff_int_old = np.inf
		int_lim_final = np.inf
		values, bins = np.histogram(pos[:,i], bins=nb, density=True)
		#values = np.hstack((0,values,0))
		binsm = ((bins[1::] - bins[:-1])/2) + bins[:-1]
		#binsm = np.hstack((bins[0] -(bins[-1] - bins[-2]) ,binsm, bins[-1]+(bins[-1] - bins[-2])))
		fh = interp1d( binsm, values)

		x = np.linspace(binsm.min(),binsm.max(), 1e4)
		y = fh(x)
		int_tot = round(trapz(y,x))
		binsm = x
		values = y 
		maximun = binsm[np.where(values==values.max())[0][0]]
		yy = np.linspace(values.min(),values.max(), 1e4)
		#yy = np.linspace(0,values.max(), 1e4)
		# if i==1:
		# 	plt.figure()
		# 	plt.plot(binsm,values)
		for val in yy:
			f = hline(0,binsm,val)
			idx = np.argwhere(np.diff(np.sign(f-values)) != 0)

			x = np.linspace(binsm[idx[0]],binsm[idx[-1]],1e4)
			y =fh(x)
			int_lim = (trapz(y,x))
			# if i ==1:
			# 	plt.axhline(val, c='grey', alpha=0.5)
			# 	plt.plot(x,y)
			# 	pylab.pause(0.1)
			if 0.675<int_lim <0.685:
				#print (int_lim, val)
				diff_int = abs(abs(int_tot -int_lim) - 0.32)
				if diff_int < diff_int_old and binsm[idx[0]] < maximun and binsm[idx[-1]] > maximun:
					val_min = val
					lim_low, lim_up = binsm[idx[0]],binsm[idx[-1]]
					diff_int_old = diff_int
					int_lim_final = int_lim
					#print (int_lim_final)
			if int_lim_final == np.inf:
				lim_low, lim_up = np.nan, np.nan
				int_lim_final = np.nan
		#print (int_lim_final)
		results[i,:] = maximun, lim_up -maximun, maximun - lim_low
	y_mcmc = maximun, lim_up -maximun, maximun - lim_low

	if ndim ==1:
		print("""MCMC result:
			Pei = {0[0]} +{0[1]} -{0[2]} (truth: {1})
			Y_sph(R500) (10^-5 Mpc^2) = {2[0]} + {2[1]} -{2[2]} (truth:{3})
			""".format(results[0], A10[0], results[1], Y_true))

	if ndim ==2:
		print("""MCMC result:
				Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				Y_sph(R500) (10^-5 Mpc^2) = {4[0]:0.2f} ^{{+ {4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth:{5})
				""".format(results[0], A10[0] ,results[1], A10[1],results[2], Y_true))
	if ndim ==3:
		print("""MCMC result:
			Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
			c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
			b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
			Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
			""".format(results[0], A10[0] , results[1], A10[1],results[2], A10[3], results[3], Y_true)) 

	return y_mcmc, results


def grid_unc_cal(pos, lnlike, A10, ndim, Y_true, H_0=const['H_0'], **kwargs):

	#ERRORS IN Y ARE CALCULATED WORNG -- IGNORE
	pos = abs(pos)

	results = np.zeros(shape=(pos.shape[1], 3))

	for n in range(pos.shape[1]):
		mx = pos[(np.where(lnlike == max(lnlike))),n]
		err = (pos[1::,n] - pos[0:-1,n]).max()
		results[n,:] = mx, err, err

	y_mcmc = mx, err, err
	
	if ndim ==1:
		print("""MCMC result:
			Pei = {0[0]} +{0[1]} -{0[2]} (truth: {1})
			Y_sph(R500) (10^-5 Mpc^2) = {2[0]} + {2[1]} -{2[2]} (truth:{3})
			""".format(results[0], A10[0], results[1], Y_true))
		

	if ndim ==2:
		if 'betac500' in kwargs:
			print("""MCMC result:
				c_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
				beta = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
				Y_sph(R500) (10^-5 Mpc^2) = {4[0]:0.2f} ^{{+ {4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth:{5})
				""".format(results[0], A10[1] ,results[1], A10[3],results[2], Y_true))
		else:
			print("""MCMC result:
			Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
			c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
			Y_sph(R500) (10^-5 Mpc^2) = {4[0]:0.2f} ^{{+ {4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth:{5})
			""".format(results[0], A10[0] ,results[1], A10[1],results[2], Y_true))
	if ndim ==3:
		if 'm500fit' in kwargs:
			print("""MCMC result:
			M_500 = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
			c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
			b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
			Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
			""".format(results[0], kwargs['m500'] , results[1], A10[1],results[2], A10[3], results[3], Y_true)) 
			
		else:
			print("""MCMC result:
			Pei = {0[0]:0.2f} ^{{+{0[1]:0.2f}}} _{{-{0[2]:0.2f}}} (truth: {1})
			c_500 = {2[0]:0.2f} ^{{+{2[1]:0.2f}}} _{{-{2[2]:0.2f}}} (truth: {3})
			b = {4[0]:0.2f} ^{{+{4[1]:0.2f}}} _{{-{4[2]:0.2f}}} (truth: {5})
			Y_sph(R500) (10^-5 Mpc^2) = {6[0]:0.2f} ^{{+{6[1]:0.2f}}} _{{-{6[2]:0.2f}}} (truth:{7})
			""".format(results[0], A10[0] , results[1], A10[1],results[2], A10[3], results[3], Y_true)) 


	return y_mcmc[0], results



# def peak_unc_cal_eqp(pos, A10, ndim, Y_true, H_0=const['H_0'], **kwargs):

# 	pos = abs(pos)

# 	from scipy.interpolate import interp1d
# 	from scipy.integrate import trapz
# 	import numpy as np

# 	nb = int(round(len(pos)**(1/3.)))
# 	diff_int_old = np.inf


# 	for i in range(pos.shape[1]):
# 		values, bins = np.histogram(pos[:,i], bins=nb, density=True)
# 		binsm = ((bins[1::] - bins[:-1])/2) + bins[:-1]
# 		fh = interp1d( binsm, values)
# 		x = np.linspace(binsm.min(),binsm.max(), 1e5)
# 		y = fh(x)
# 		int_tot = round(trapz(y,x))
# 		binsm = x
# 		values = y 
# 		values_mx_index = np.where(values==values.max())[0][0]
# 		first_half = values[0:values_mx_index+1]
# 		second_half = values[values_mx_index+1:
# 		for ii, val in enumerate(first_half):
# 			diff =abs(val -second_half)
# 			minp = np.where(diff ==diff.min())[0][-1]
# 			index = np.where(second_half[minp]==values)[0][-1]
# 			x = np.linspace(binsm[ii],binsm[index],1e5)
# 			y =fh(x)
# 			int_lim = (trapz(y,x))
# 			#print (int_tot -int_lim)
# 			diff_int = abs((int_tot -int_lim) - 0.32)
# 			if diff_int < diff_int_old:
# 				val_min = val
# 				lim_low, lim_up = binsm[ii],binsm[index]
# 				diff_int_old = diff_int
# 		return (val_min,binsm[values_mx_index],lim_low,lim_up)
	#return -np.inf, -np.inf
 

