import GNFW_functions as fun 
import numpy as np
import radial_data as rad


tst = [0,0]

def gaussprior (theta, ndim, A10, **kwargs):

	if 'dclevel' in kwargs:
		ndim = ndim-1
		theta = theta[0:-1]
	if A10[-1] ==-np.inf:
		return 0,0
	elif any(val <0 for val in theta) and 'mirror_effect' not in kwargs:
		return -np.inf, -np.inf
	else:
		if ndim ==3:
			Pe0, c_500, b = theta
		if 'c500prior' in kwargs:
			mu = A10[1] #values from Planck fit
			s = A10[5]#value chosen from literature and experience :P
			param = abs(theta[1])#abs(c_500) 		
		if 'm500prior' in kwargs:
			mu = kwargs['m500']
			s =A10[5]
			param = theta[0]#Pe0 #in this case this is the m500 that was pick 
		else:
			mu = A10[0] #values from Planck fit
			s = A10[5]#0.0075 #- A1835 from Planck - for super S/N 0.0075, for SN=5 - 0.78
			param= theta[0]#Pe0
			#s = 0.62 # RXJ1347 from Planck
			#mu = 10.316
			#s = 1.05
		gauss = (1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*(float(param-mu)/s)**2)) / (1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*(float(mu-mu)/s)**2))
		
		loggauss = np.log(gauss)
		return gauss, loggauss

def yprior (yin, ndim, yplanck, ysigma):
	mu = yplanck #values from Planck fit
	s = ysigma#0.0075 #- A1835 from Planck - for super S/N 0.0075, for SN=5 - 0.78
	#s = 0.62 # RXJ1347 from Planck
	#mu = 10.316
	#s = 1.05
	gauss = (1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*(float(yin-mu)/s)**2)) / (1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*(float(mu-mu)/s)**2))
	loggauss = np.log(gauss)
	return gauss, loggauss

def exponprior (theta, ndim, A10, **kwargs):
	from scipy.stats import expon

	if 'dclevel' in kwargs:
		ndim = ndim-1
		theta = theta[0:-1]
	if A10[-1] ==-np.inf:
		return 0.0,0.0
	elif any(val <0 for val in theta) and 'mirror_effect' not in kwargs:
		return -np.inf, -np.inf
	else:
		if ndim ==3:
			Pe0, c_500, b = theta
			c_500 = (c_500)

		if  -A10[1]-1.5 < c_500 < A10[1]+1.5:
			return 1.0,np.log(1.0)

		loc_exp = A10[1]+1.5

		prob = expon.pdf(abs(c_500), loc=loc_exp, scale=A10[-1]) #scale is 1/lambda

		return prob, np.log(prob)


def lnprior(theta, ndim, A10, **kwargs):

	if 'dclevel' in kwargs and ndim !=4:
		ndim = ndim-1
		theta = theta[0:-1]

	if ndim ==1:
		Pe0 = theta #a, b, c,
		
		if 0< Pe0< 15*A10[0]: 
			return 0.0

	elif ndim ==2:
		if 'betac500' in kwargs:
			c_500, b = theta
			if  0 < c_500 <15*A10[1] and 0*A10[3]< b< 15*A10[3]: 
				return 0.0
		else:
			Pe0, c_500= theta
			if 0< Pe0< 15*A10[0] and 0 < c_500 <15*A10[1]:
				return 0.0
	
	elif ndim ==3 and A10[-1] != -np.inf:
		Pe0, c_500, b = theta
		if 'c500prior' in kwargs or 'exprior' in kwargs:
			if 0< Pe0< 15*A10[0] and 0< b< 15*A10[3]: 
				return 0.0
		else:
			if 0 < c_500 <15*A10[1] and 0*A10[3]< b< 15*A10[3]: 
				return 0.0

	elif ndim ==3 and A10[-1]==-np.inf :
		#print ('theta_after abs', theta)
		Pe0, c_500, b = theta
		if 'logprior'in kwargs:
			if 0<Pe0< 15*A10[0] and c_500 <np.log(15*A10[1]) and  0<b< 15*A10[3]: 
				return 0.0
		if 'mirror_effect' in kwargs:
			if 0<Pe0< 15*A10[0] and -15*A10[1]<c_500 <15*A10[1] and  0<b< 15*A10[3]: 
				return 0.0
		else:
			if 0< Pe0< 30*A10[0] and 0 < c_500 <20*A10[1] and 0< b< 20*A10[3]: 
				return 0.0
		

	elif ndim ==4:
		if 'dclevel' in kwargs:
			Pe0, c_500, b, dc_level = theta
			if 0< Pe0< 15*A10[0] and 0 < c_500 <15*A10[1] and 0< b< 15*A10[3]: 
				return 0.0
		else:
			Pe0, c_500, a,b = theta
			if 0< Pe0< 15*A10[0] and 0 < c_500 <15*A10[1] and 0< a< 15*A10[2] and 0< b< 15*A10[3]: 
				return 0.0

	elif ndim ==5:
		if 'fitcenter' in kwargs:
			xc_lowlim = kwargs['fitcenter'][0]-10
			xc_uplim = kwargs['fitcenter'][0]+10
			yc_lowlim = kwargs['fitcenter'][1]-10
			yc_uplim = kwargs['fitcenter'][1]+10

			Pe0, c_500, b, xc, yc = theta
			if 0< Pe0< 30*A10[0] and 0 < c_500 <20*A10[1] and 0<b<20*A10[3] and xc_lowlim<xc<xc_uplim and yc_lowlim< yc< yc_uplim : 
				return 0.0
		else:
			Pe0, c_500, c,a, b = theta
			if 0< Pe0< 15*A10[0] and 0 < c_500 <15*A10[1] and 0<c<15*A10[4] and 0<a<15*A10[2] and 0< b< 15*A10[3] : 
				return 0.0
	return -np.inf

 
def lnlike(theta, data, yerr_i, ndim, A10, apex=False, planck=False, **kwargs):
		#kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500, resolution
 	
 # 	import time
	# start_like = time.clock()	

	from apex_planck_sim import sim2d

	if 'dclevel' in kwargs:
		ndim = ndim -1
		dc_level = theta[-1]
		theta= theta[0:-1]

	if ndim==1:
		Pe0 = theta
	elif ndim ==2:
		Pe0, c_500 = theta
	elif ndim ==3:
		# if 'logprior' in kwargs:
		# 	Pe0, c_500, b = np.exp(theta)
		# else:
		Pe0, c_500, b = theta
	elif ndim ==4:
		Pe0, c_500, b, dc_level= theta
	elif ndim ==5:
		Pe0, c_500,c,a, b = theta
	
	rmax = kwargs['rmax']
	width = kwargs['width']
	if 'width' in kwargs: del kwargs['width']
	if 'logprior' in kwargs: theta=[Pe0, np.exp(c_500), b]
	if 'mirror_effect' in kwargs: theta=[abs(Pe0), abs(c_500), abs(b)]
	#print (theta)

	if 'dclevel' in kwargs:
		theta = theta, dc_level

	if apex == True:
		weight = kwargs.get('weight', np.ones(shape=(rmax*2+1,rmax*2+1)))
		if 'resolution' in kwargs: del kwargs['resolution']
		if 'y_prior' in kwargs:
			yplanck = kwargs['yplanck']
			ysigma = kwargs['ysigma']
			model_con, Y_vals=(sim2d(theta, ndim, A10, apex=True, **kwargs ))[1:3]
			#print (Y_vals)
			yp = yprior(Y_vals, ndim, yplanck, ysigma)[1]
			#print (yp)
		else:
			model_con=(sim2d(theta, ndim, A10, apex=True, **kwargs ))[1]#*10
			yp = 0.0
		#print (np.all(weight==1))
		model_con = model_con*weight

	if planck == True:
		yp = 0.0
		weight = None
		if 'freq_obs' in kwargs: del kwargs['freq_obs']
		model_con=sim2d(theta, ndim, A10,  planck=True, **kwargs)[1]
		
	model_rad=rad.radial_data(model_con, annulus_width=width, rmax=rmax, weight=weight)
	mean_rad_model = np.array([model_rad.meannan])#meannan meanweight
	
	if 'centerout' in kwargs:
		mean_rad_model = mean_rad_model[:,1:]

	array = ( data - mean_rad_model)
	#print (array)
	chi2 =  (array).dot(yerr_i).dot(array.T)
	likelihood = -0.5*chi2 + yp
	#print "t_like=",  time.clock() - start_like
	#print (theta)
	#print ('like',likelihood)
	return likelihood[0][0]

def lnlike_combap(theta, data_a, data_p, yerr_ia, yerr_ip, ndim, A10, apex=False, planck=False, **kwargs):
		#kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500, resolution
	
 
	from apex_planck_sim import sim2d

	if ndim==1:
		Pe0 = theta
	elif ndim ==2:
		Pe0, c_500 = theta
	elif ndim ==3:
		Pe0, c_500, b = theta
	elif ndim ==5:
		Pe0, c_500,c,a, b = theta
	
	#print ("Theta", theta)

	########################## APEX Chi2 #################################
	try:

		if  kwargs['kwargs_apex']['like'] or apex:

			width_a = kwargs['kwargs_apex']['width']
			rmax_a = kwargs['kwargs_apex']['rmax']
			#print ("M500_a",kwargs['kwargs_apex']['m500'] )
			#if 'width' in kwargs['kwargs_apex']: del kwargs['kwargs_apex']['width']
			#if 'resolution' in kwargs['kwargs_apex']: del kwargs['kwargs_apex']['resolution']

			if kwargs.get('m500fit'):
				kwargs['kwargs_apex']['m500fit'] = True


			if 'y_prior' in kwargs['kwargs_apex']:
					model_cona, Y_vals=(sim2d(theta, ndim, A10, apex=True, **kwargs['kwargs_apex'] ))[1:3]
					#print(Y_vals)
			else:
				model_cona=(sim2d(theta, ndim, A10, apex=True, **kwargs['kwargs_apex'] ))[1]#*10

			model_rada=rad.radial_data(model_cona, annulus_width=width_a, rmax=rmax_a)
			mean_rad_modela = np.array([model_rada.meannan])#meannan meanweight
			arraya = ( data_a - mean_rad_modela)
			#arraya = arraya[0,:,:]
			chi2a =  (arraya).dot(yerr_ia).dot(arraya.T)
		else:
			chi2a = 0
	except:
		chi2a = 0

	# print ("lnlikea",-0.5*chi2a)
	# print ("chi2a",chi2a)
	
	#################################################################################
	
	########################## PLANCK Chi2 ###############################
	try:
		if  kwargs['kwargs_planck']['like'] or planck:
			width_p = kwargs['kwargs_planck']['width']
			rmax_p = kwargs['kwargs_planck']['rmax']
			# print ("M500_p",kwargs['kwargs_planck']['m500'] )
			#if 'freq_obs' in kwargs['kwargs_planck']: del kwargs['kwargs_planck']['freq_obs']
			#if 'width' in kwargs['kwargs_planck']: del kwargs['kwargs_planck']['width']
			if kwargs.get('m500fit'):
				kwargs['kwargs_planck']['m500fit'] = True
			model_conp=sim2d(theta, ndim, A10,  planck=True, **kwargs['kwargs_planck'])[1]
			model_radp=rad.radial_data(model_conp, annulus_width=width_p, rmax=rmax_p)
			mean_rad_modelp = np.array([model_radp.meannan])#meannan meanweight
			arrayp = ( data_p - mean_rad_modelp)
			#arrayp = arrayp[0,:,:]
			chi2p =  (arrayp).dot(yerr_ip).dot(arrayp.T)

			
		else:
			chi2p = 0 
	except:
		chi2p = 0
	# print ("lnlikep", -0.5*chi2p)
	# print ("chi2p", chi2p)
	#################################################################################

	likelihood = -0.5*(chi2a+chi2p )
	#print "t_like=",  time.clock() - start_like
	#print (theta)
	#print (likelihood)
	try: 
		if 'y_prior' in kwargs['kwargs_apex']:
			#print (Y_vals)
			return likelihood[0][0], Y_vals
		else:
			return likelihood[0][0]
	except:
		if 'y_prior' in kwargs['kwargs_planck']:
			#print (Y_vals)
			return likelihood[0][0], Y_vals
		else:
			return likelihood[0][0]


def lnprob_apex(theta, data, yerr_i, ndim, A10,   **kwargs):
	#kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500
	
	# if 'mirror_effect' in kwargs:
	# 	theta = np.asarray(theta)
	# 	theta[1] = abs(theta[1])
	lp = lnprior(theta, ndim, A10, **kwargs)
	if not np.isfinite(lp):
		return -np.inf
	if "exprior" in kwargs:
		ep = exponprior(theta, ndim, A10, **kwargs)[1]
		gp = 0
		if not np.isfinite(ep):
			return -np.inf
	else:
		gp = gaussprior(theta, ndim, A10, **kwargs)[1]
		ep =0 
		#print (gp)
		if not np.isfinite(gp):
			return -np.inf
	return lp + gp + ep +lnlike(theta, data, yerr_i, ndim, A10, apex=True, **kwargs)

def lnprob_planck(theta, data, yerr_i, ndim, A10,  **kwargs):
	#kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, m500, r500, resolution
	#print ('theta_before abs',theta)
	# if 'mirror_effect' in kwargs:
	# 	theta = np.asarray(theta)
	# 	#theta[1] = abs(theta[1])
	# 	#theta[0] = -theta[0]

	if 'sample_olda'in kwargs:
		num = np.random.randint(len(kwargs['sample_olda']))
		A10[1], A10[3] = kwargs['sample_olda'][num,1:3]
		del kwargs['sample_olda']
	lp = lnprior(theta, ndim, A10, **kwargs)
	if not np.isfinite(lp):
		return -np.inf
	if "exprior" in kwargs:
		ep = exponprior(theta, ndim, A10, **kwargs)[1]
		gp = 0
		if not np.isfinite(ep):
			return -np.inf
	else:
		ep = 0 
		gp = gaussprior(theta, ndim, A10, **kwargs)[1]
		#print ('m500 prior result', gp)
		if not np.isfinite(gp):
			return -np.inf
	return lp + ep+ gp + lnlike(theta, data, yerr_i, ndim, A10, planck=True, **kwargs )

def lnprob_combap(theta, data_a, data_p, yerr_ia, yerr_ip, ndim, A10,   **kwargs):
	if 'mirror_effect' in kwargs:
		theta = np.asarray(theta)
		theta[1] = abs(theta[1])

	if 'm500random' in kwargs:
		import random
		M500_random = kwargs['m500err'] ### confirm that this is read properly!!!
		M500 = random.choice(M500_random)
		if M500 <= 0:
			M500 = random.choice(M500_random)
		try:
			kwargs['kwargs_apex']['m500'] = M500
		except:
			pass
		try:
			kwargs['kwargs_planck']['m500'] = M500
		except:
			pass
		#print ("M500 Pick", M500)

	if "exprior" in kwargs:
		ep = exponprior(theta, ndim, A10)[1]
		#print (ep)
		gp = 0.0
		if not np.isfinite(ep):
			try:
				if 'y_prior' in kwargs['kwargs_apex']:
					return -np.inf, -np.inf
				else:
					return -np.inf
			except:
				if 'y_prior' in kwargs['kwargs_planck']:
					return -np.inf, -np.inf
				else:
					return -np.inf
			
	else:
		gp = gaussprior(theta, ndim, A10, **kwargs)[1]
		ep =0.0 
		if not np.isfinite(gp):
			try:
				if 'y_prior' in kwargs['kwargs_apex']:
					return -np.inf, -np.inf
				else:
					return -np.inf
			except:
				if 'y_prior' in kwargs['kwargs_planck']:
					return -np.inf, -np.inf
				else:
					return -np.inf

	lp = lnprior(theta, ndim, A10, **kwargs)
	
	if not np.isfinite(lp):
		try: 
			if 'y_prior' in kwargs['kwargs_apex']:
				return -np.inf, -np.inf
			else:
				return -np.inf
		except:
			if 'y_prior' in kwargs['kwargs_planck']:
				return -np.inf, -np.inf
			else:
				return -np.inf
	try:
		if 'y_prior' in kwargs['kwargs_apex']:
			like, Y_vals = lnlike_combap(theta, data_a, data_p, yerr_ia, yerr_ip, ndim, A10, **kwargs)
			#print ( M500, Y_vals)
			return lp +gp + ep + like, Y_vals
		else:
			return lp  + gp +ep + lnlike_combap(theta, data_a, data_p, yerr_ia, yerr_ip, ndim, A10, **kwargs)
	except:
		if 'y_prior' in kwargs['kwargs_planck']:
			like, Y_vals = lnlike_combap(theta, data_a, data_p, yerr_ia, yerr_ip, ndim, A10, **kwargs)
			#print ( M500, Y_vals)
			return lp +gp + ep + like, Y_vals
		else:
			return lp  + gp +ep + lnlike_combap(theta, data_a, data_p, yerr_ia, yerr_ip, ndim, A10, **kwargs)

def combl_multi (theta, data_a, data_p, cov_a, cov_p, **kwargs):
	#kwargs{ndim, A10} - #kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500, resolution

	import numpy as np
	
	
	ndim = kwargs['ndim']
	if 'ndim' in kwargs: del kwargs['ndim']
	A10 = kwargs['A10']
	if 'A10' in kwargs: del kwargs['A10']
	
	prob=0
	probt = []
	chi2 = 0

	try:
		nclusters = len(data_a)
	except:
		nclusters = len(data_p)

	for j in range (nclusters):
		#print (j)
		try:
			if kwargs['like_a'] == True:
				kwargs['kwargs_apex'][j]['like']=True
				if 'fitcenter' in kwargs:
					kwargs_i['fitcenter'] = kwargs['fitcenter']
					#del(kwargs['kwargs_apex'][j]['xcent'])
					#del(kwargs['kwargs_apex'][j]['ycent'])
					kwargs['kwargs_apex'][j]['fitcenter'] = kwargs['fitcenter']
			else:
				kwargs['kwargs_apex'][j]['like']=False
			kwargs_i ={'kwargs_apex':kwargs['kwargs_apex'][j]}
		except:
			kwargs_i ={}

		try:
			if kwargs['like_p'] == True:
				kwargs['kwargs_planck'][j]['like']=True
				if 'fitcenter' in kwargs:
					kwargs_i['fitcenter'] = kwargs['fitcenter']
					#del(kwargs['kwargs_planck'][j]['xcent'])
					#del(kwargs['kwargs_planck'][j]['ycent'])
					kwargs['kwargs_planck'][j]['fitcenter'] = kwargs['fitcenter']
					#print (kwargs['kwargs_planck'][j])
			else:
				kwargs['kwargs_planck'][j]['like']=False
			kwargs_i['kwargs_planck'] = kwargs['kwargs_planck'][j]
		except:
			kwargs['kwargs_planck'][j] =None




		#kwargs_i ={'kwargs_apex':kwargs['kwargs_apex'][j],'kwargs_planck': kwargs['kwargs_planck'][j]}

		
		try:
			name = kwargs['kwargs_planck'][j]['name']
		except:
			name = kwargs['kwargs_apex'][j]['name']

		if 'exprior' in kwargs:kwargs_i['exprior'] = True	

		if 'm500fit' in kwargs:
			kwargs_i['m500fit'] = True
			try:
				kwargs_i['m500'] = kwargs['kwargs_apex'][j]['m500']
			except:
				kwargs_i['m500'] = kwargs['kwargs_planck'][j]['m500']


		if 'm500prior' in kwargs:
			kwargs_i['m500prior'] = True
			try:
				A10[-1] = kwargs['kwargs_apex'][j]['m500err']
			except:
				A10[-1] = kwargs['kwargs_planck'][j]['m500err']

		if data_a == None:
			probj = lnprob_combap(theta, data_a, data_p[j], cov_a, cov_p[j], ndim, A10, **kwargs_i)
		 	#prob += lnprob_combap(theta, data_a, data_p[j], cov_a, cov_p[j], ndim, A10, **kwargs_i)
		elif data_p ==None:
			probj = lnprob_combap(theta, data_a[j], data_p, cov_a[j], cov_p, ndim, A10, **kwargs_i)
		 	#prob += lnprob_combap(theta, data_a[j], data_p, cov_a[j], cov_p, ndim, A10, **kwargs_i)
		else:
			probj = lnprob_combap(theta, data_a[j], data_p[j], cov_a[j], cov_p[j], ndim, A10, **kwargs_i)
		 	#prob += lnprob_combap(theta, data_a[j], data_p[j], cov_a[j], cov_p[j], ndim, A10, **kwargs_i)

		#f = open ('comb17A1835_likelihoods.dat', "a+")
		#results_k = j, probj
		#np.savetxt(f,np.atleast_2d(results_k), fmt='%0.0f %.3f')
		#f.close()
		prob +=probj

		#print (kwargs['kwargs_apex'][j]['trans_fun'][270,270])
		# print ('individuals', probj)
		# print ('prob',prob)
				
		#thetaj = np.hstack((theta[j],theta[nclusters:]))
		#prob += lnprob_combap(thetaj, data_a[j], data_p[j], cov_a[j], cov_p[j], ndim, A10, **kwargs_i)
		#print (prob)
	#prob = -0.5*chi2
	return prob

def stack_l(theta, data_a, data_p, cov_a, cov_p, **kwargs):
	#kwargs{ndim, A10} - #kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500, resolution

	import numpy as np
	
	
	ndim = kwargs['ndim']
	if 'ndim' in kwargs: del kwargs['ndim']
	A10 = kwargs['A10']
	if 'A10' in kwargs: del kwargs['A10']
	
	prob=0
	probt = []
	chi2 = 0

	try:
		nclusters = len(data_a)
	except:
		nclusters = len(data_p)

	for j in range (nclusters):
		#print (j)
		try:
			if kwargs['like_a'] == True:
				kwargs['kwargs_apex'][j]['like']=True
			else:
				kwargs['kwargs_apex'][j]['like']=False
			kwargs_i ={'kwargs_apex':kwargs['kwargs_apex'][j]}
		except:
			kwargs_i ={}

		try:
			if kwargs['like_p'] == True:
				kwargs['kwargs_planck'][j]['like']=True
			else:
				kwargs['kwargs_planck'][j]['like']=False
			kwargs_i['kwargs_planck'] = kwargs['kwargs_planck'][j]
		except:
			kwargs['kwargs_planck'][j] =None




		#kwargs_i ={'kwargs_apex':kwargs['kwargs_apex'][j],'kwargs_planck': kwargs['kwargs_planck'][j]}

		
		try:
			name = kwargs['kwargs_planck'][j]['name']
		except:
			name = kwargs['kwargs_apex'][j]['name']

		if 'exprior' in kwargs:kwargs_i['exprior'] = True	

		if 'm500fit' in kwargs:
			kwargs_i['m500fit'] = True
			try:
				kwargs_i['m500'] = kwargs['kwargs_apex'][j]['m500']
			except:
				kwargs_i['m500'] = kwargs['kwargs_planck'][j]['m500']

		if 'm500prior' in kwargs:
			kwargs_i['m500prior'] = True
			try:
				A10[-1] = kwargs['kwargs_apex'][j]['m500err']
			except:
				A10[-1] = kwargs['kwargs_planck'][j]['m500err']

		if data_a == None:
			probj = lnprob_combap(theta, data_a, data_p[j], cov_a, cov_p[j], ndim, A10, **kwargs_i)
		 	#prob += lnprob_combap(theta, data_a, data_p[j], cov_a, cov_p[j], ndim, A10, **kwargs_i)
		elif data_p ==None:
			probj = lnprob_combap(theta, data_a[j], data_p, cov_a[j], cov_p, ndim, A10, **kwargs_i)
		 	#prob += lnprob_combap(theta, data_a[j], data_p, cov_a[j], cov_p, ndim, A10, **kwargs_i)
		else:
			probj = lnprob_combap(theta, data_a[j], data_p[j], cov_a[j], cov_p[j], ndim, A10, **kwargs_i)
		 	#prob += lnprob_combap(theta, data_a[j], data_p[j], cov_a[j], cov_p[j], ndim, A10, **kwargs_i)

		#f = open ('comb17A1835_likelihoods.dat', "a+")
		#results_k = j, probj
		#np.savetxt(f,np.atleast_2d(results_k), fmt='%0.0f %.3f')
		#f.close()
		prob +=probj

		#print (kwargs['kwargs_apex'][j]['trans_fun'][270,270])
		# print ('individuals', probj)
		# print ('prob',prob)
				
		#thetaj = np.hstack((theta[j],theta[nclusters:]))
		#prob += lnprob_combap(thetaj, data_a[j], data_p[j], cov_a[j], cov_p[j], ndim, A10, **kwargs_i)
		#print (prob)
	#prob = -0.5*chi2
	return prob

def run_mcmc(datafile, theta, data, yerr_i, ndim, A10, apex = False, planck = False, **kwargs):
	#kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500, resolution
	import emcee
	import os
	import random
	import sys
	import time
	import acor

	nwalkers = kwargs['nwalkers']
	ite = kwargs['ite']
	threads = kwargs['threads']
	burnin = int(ite/3)
	per_factor = 100

	dclevel= kwargs.get('dclevel', False)


	if os.path.isfile(datafile):
		f= open(datafile, "r")
		file = list(f)
		f.close()
		diff = (nwalkers*ite - len(file))  
		ite_o = ite 
		if 0 < diff <= nwalkers:
			ite = round((nwalkers/diff))
			pos = np.zeros(shape=(nwalkers, ndim))
			file_pos = file[-nwalkers::]
			for line, k  in zip(file_pos, range(nwalkers)):
				pos[k,:] = line.split()[0:-1]
				
			if burnin*nwalkers < len(file): 
				burnin = 0
			else:
				burnin = int(ite/3.)
				print ("New Burnin", burnin)
			print ("Restarting MCMC - iterations left = ", ite)
		
		elif diff > nwalkers:
			missing_walkers = nwalkers - (diff//ite)
			ite = round(diff/nwalkers)
			pos = np.zeros(shape=(nwalkers, ndim))
			file_pos = file[-nwalkers::]
			for line, k  in zip(file_pos, range(int(nwalkers-missing_walkers ))):
				pos[k,:] = line.split()[0:-1]
			for m in range (int(missing_walkers)):
				pos[-1-m,:] = pos[m,:]+ 1e-1*np.random.randn(ndim)
			print ("Restarting MCMC - iterations left = ", ite)
			
		elif diff <= 0:
			print ("MCMC Run has been completeted")
			sys.exit(0)

	else :
		
		#define initial position of the walkers
		if ndim ==1:
			p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1)]
		elif ndim ==2:
			if 'betac500' in kwargs:
				print ('Sampling for c_500 and beta only')
				p0 = [random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]
			else:
				p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(A10[1]-0.5,A10[1]+0.5)]
		elif ndim ==3:
			if 'logprior' in kwargs:
				print ('Sampling in Log -corrected! -only on c500')
				p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1), np.log(random.uniform(A10[1]-0.5,A10[1]+0.5)), random.uniform(A10[3]-0.5,A10[3]+0.5)]#
			elif 'mirror_effect' in kwargs:
				print ('Sampling negative space c500 - Mirror Effect')
				p0 = [random.uniform(0.5,10), random.uniform(-0.5,+0.5), random.uniform(0.5,10)]
				#p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(-0.5,+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]
			else:
				p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]#
		elif ndim ==4:
			if 'dclevel' in kwargs:
				print ('Sampling for DC-Level')
				p0 = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5), random.uniform(-0.05,+0.05)]#, random.randint(1,2)]
			else:
				p0 = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]
		elif ndim ==5:
			p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[4]-0.5,A10[4]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]#, 
		
		if dclevel ==True and ndim != 4:
			print ('DC-Level included')
			dcran = random.uniform(-0.05,+0.05)
			p0.append(dcran)
			ndim = ndim +1

		pos = [ p0+ 1e0*np.random.randn(ndim) for i in range(nwalkers)]

		if 'mirror_effect' in kwargs:
			pos = np.asarray(pos)
			pos[len(pos)/2:,1]= (pos[len(pos)/2:,1])*-1

		#print (pos, ndim)
		
		#pos = np.load('GNFW_chains/poc500_w200_i100_apexnoise_TF.npy')[:,49,:]

		f = open(datafile, "w")
		f.close()
	
	print("Running MCMC...")

	if apex == True:
		print ("APEX")
	if planck == True:
		print ("PLANCK")
	if 'logprior' in kwargs:
		print ('Sampling in Log')
	if 'mirror_effect' in kwargs:
		print ('Mirror Effect')



	if 'threads' in kwargs: del kwargs['threads']
	if 'nwalkers' in kwargs: del kwargs['nwalkers']
	if 'ite' in kwargs: del kwargs['ite']
	if 'maf' in kwargs: del kwargs['maf']
	if 'tau' in kwargs: del kwargs['tau']

	
	

	if apex == True:
		#threads = 1
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_apex, args=(data, yerr_i, ndim, A10), kwargs=(kwargs), threads=threads) 
			# Clear and run the production chain.
	if planck ==True:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_planck, args=(data, yerr_i, ndim, A10),kwargs=(kwargs), threads=threads) 
			# Clear and run the production chain.

	if dclevel == True:
		ndim = ndim -1

	for i, result in enumerate(sampler.sample(pos, iterations=ite, rstate0=np.random.get_state(), storechain=True)):
		if i%per_factor==0:
			position = result[0]
			likelihood = result[1]
			f = open (datafile, "a+")
			for k in range(position.shape[0]):
				results_k = np.hstack((result[0][k], result[1][k]))
				np.savetxt(f,np.atleast_2d(results_k), fmt='%.6f')
			f.close()
		#if (i+1)%per_factor==0:
			print ("{0:5.1%}".format(float(i)/ite))
			time.sleep(0.01)

	print ("Done.")		

	if dclevel == True:
		ndim = ndim+1
	samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
	if 'logprior' in kwargs:
		samples = np.exp(samples[:,1])
	chains = sampler.chain[:, :, :]
	chains_re = chains.reshape((-1, ndim))
	lnprob_sampler=sampler.lnprobability[:,burnin:].reshape((-1, 1))
	

	maf = (np.mean(sampler.acceptance_fraction))
	try:
		tau, mean, sigmat = acor.acor(samples[:,0])
		#tau = emcee.autocorr.integrated_time(samples, axis=0)[0]
	except:
		tau = 100
	print("Mean acceptance fraction: {0:.3f}"
				.format(maf))
	print ("Autocorrelation time:", tau)

	if threads >1:
		sampler.pool.terminate()
	sampler.reset()

	return samples, lnprob_sampler, maf, tau


def run_mymcmc(datafile, theta, data, yerr_i, ndim, A10, apex = False, planck = False, **kwargs):
	#kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500, resolution

	import  numpy as np
	import os
	import random
	import time
	import sys


	nwalkers = kwargs['nwalkers']
	ite = kwargs['ite']
	threads = kwargs['threads']
	#numer of iterations
	num = int(ite*nwalkers)
	burnin= int(num/5.)#n/5
	iterations = num - burnin
	per_factor = 100
	

	if os.path.isfile(datafile):
		file = np.loadtxt(datafile)
		#diff = (num - len(file))  
		#ite_o = ite 
		if len(file) < burnin and len(file)>0 :
			burnin = burnin - len(file)
			pos = file[-1,1:-1]
			A_file = file[:,1:-1]
			print ("MCMC re-starting. Number of burnin steps left %0.f" %float(burnin))
		elif len(file) >= num:
			print ("MCMC Run has been completeted")
			sys.exit(0)
		elif burnin <= len(file) < num:
			iterations = num - len(file)
			burnin = 0
			pos = file[-1,1:-1]
			A_file = file[:,1:-1] #
			like_file = file[:, -1]
			print ("MCMC re-starting after burinin. Number of iterations left %0.f" %float(iterations))
		else:
			os.remove(datafile)
			print ("Restarting MCMC - since beginning")
			if ndim ==1:
				pos = [random.uniform(A10[0]-0.1,A10[0]+0.1)]
			elif ndim ==2:
				pos = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(A10[1]-0.5,A10[1]+0.5)]
			elif ndim ==3:
				if 'mirror_effect' in kwargs:
					print ('Sampling negative space c500 - Mirror Effect')
					pos = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(-A10[1]-0.5,-A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]
				else:
					pos = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]#
			elif ndim ==4:
				if 'dclevel' in kwargs:
					print ('Sampling for DC-Level')
					pos = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5), random.uniform(-0.05,+0.05)]#, random.randint(1,2)]
				else:
					pos = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]

			elif ndim ==5:
				pos = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[4]-0.5,A10[4]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]#, 
			f = open(datafile, "w")
			f.close()

	else :
		
		#define initial position of the walkers
		if ndim ==1:
			pos = [random.uniform(A10[0]-0.1,A10[0]+0)]
		elif ndim ==2:
			pos = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5)]
		elif ndim ==3:
			if 'mirror_effect' in kwargs:
				print ('Sampling negative space c500 - Mirror Effect')
				pos = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(-A10[1]-0.5,-A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]
			else:
				pos = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]#
		elif ndim ==4:
			if 'dclevel' in kwargs:
				print ('Sampling for DC-Level')
				pos = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5), random.uniform(-0.05,+0.05)]#, random.randint(1,2)]
			else:
				pos = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]
		elif ndim ==5:
			pos = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[4]-0.5,A10[4]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]#, 
		
		#pos = [ p0+ 1e-1*np.random.randn(ndim) for i in range(nwalkers)]
		#pos = np.load('GNFW_chains/poc500_w200_i100_apexnoise_TF.npy')[:,49,:]

		f = open(datafile, "w")
		f.close()
	

	print("Running MCMC...")

	if apex == True:
		print ("APEX")
	if planck == True:
		print ("PLANCK")



	if 'threads' in kwargs: del kwargs['threads']
	if 'nwalkers' in kwargs: del kwargs['nwalkers']
	if 'ite' in kwargs: del kwargs['ite']
	if 'maf' in kwargs: del kwargs['maf']
	if 'tau' in kwargs: del kwargs['tau']

	if burnin != 0:
		#####MCMC -MH####

		# initial guess 
		guess = pos#theta
		stepsizes = []
		# define stepsize of MCMC.
		# if ndim ==1:
		# 	stepsizes.append(abs(guess[0]*0.1)) #!!!!!
		# else:
		for i in range(len(guess)):
			stepsizes.append(abs(guess[i]*0.1))
		stepsizes = np.array(stepsizes)
		
		
		# Prepare storing MCMC chain as array of arrays.
		#A = [guess]
		A = np.zeros(shape=(burnin,ndim))
		like_MCMC= np.zeros(shape=(burnin))

		print ('initial postion', guess)
		print ('sept size', stepsizes)
		accepted  = 0.0

		print ("Running Burn-In Phase")

		# Metropolis-Hastings to calculate the step size from the burnin phase iterations.
		for n in range(burnin):
			old_params  =guess #A[len(A)-1]  # old parameter value as array
			#print ('old_param', old_params)
			if apex == True:
				old_loglik = lnprob_apex(old_params, data, yerr_i, ndim, A10, **kwargs)
			if planck == True:
				old_loglik = lnprob_planck(old_params, data, yerr_i, ndim, A10, **kwargs)
			# Suggest new candidate from Gaussian proposal distribution.
			new_params = np.zeros([len(old_params)])
			for i in range(len(old_params)):
			# Use stepsize provided for every dimension.
				new_params[i] = random.gauss(old_params[i], stepsizes[i])
			#ax1.plot(x[n], new_params[1], 'bo')
			#print ('new_params', new_params)
			if apex == True:
				new_loglik = lnprob_apex(new_params, data, yerr_i, ndim, A10, **kwargs)
			if planck == True:
				new_loglik = lnprob_planck(new_params, data, yerr_i, ndim, A10, **kwargs)
			#print 'new likelihood', new_loglik
			# Accept new candidate in Monte-Carlo fashing.
			if (new_loglik > old_loglik):
				A[n,:] = (new_params)
				#ax1.plot(x[n], new_params[1], 'go')
				like_MCMC[n]=new_loglik
				guess = new_params
				accepted = accepted + 1.0  # monitor acceptance
			else:
				u = random.uniform(0.0,1.0)
				if (u < np.exp(new_loglik - old_loglik)):
					A[n,:] =(new_params)
					#ax1.plot(x[n], new_params[1], 'go')
					like_MCMC[n]=new_loglik
					guess = new_params
					accepted = accepted + 1.0  # monitor acceptance
				else:
					A[n,:]=(old_params)
					like_MCMC[n]=old_loglik
					guess = old_params
			if n%per_factor==0:
				f = open (datafile, "a")
				if ndim ==1:
					f.write("{0:4d} {1:f} {2:f} \n".format(n, guess[0], like_MCMC[n]))
				if ndim ==3:
					f.write("{0:4d} {1:f} {2:f} {3:f} {4:f} \n".format(n, guess[0],guess[1],guess[2] , like_MCMC[n]))
				if ndim ==4:
					f.write("{0:4d} {1:f} {2:f} {3:f} {4:f} {5:f} \n".format(n, guess[0],guess[1],guess[2] , guess[3], like_MCMC[n]))
				f.close()
			#if (n+1)%per_factor==0:
				#print ("{0:5.1%}".format(float(n)/burnin))
				#pylab.pause(0.01)
				#time.sleep(0.01)
			#print (A[n,:])
			#print (like_MCMC[n])


		#new step size and position from the burnin phase
		guess_burnin = A[-1,:]
		#guess_burnin = A[burnin-1,:]
		try:
			A = np.concatenate((A_file, A), axis=0)
		except NameError:
			A = A
		print 'initial position after burnin', guess_burnin
	else:
		burnin = num/5
		A = A_file[:burnin+1,:]
		guess_burnin = pos
		print 'initial position after burnin', guess_burnin
	
	#make the covariance proposal distribution
	if ndim == 1:
		cov = np.std(A)
	else:
		cov = np.cov(A.T)
	
	print (cov)
	#reset to 0 the arrays
	A = np.zeros(shape=(iterations,ndim))
	like_MCMC= np.zeros(shape=(iterations))
	accepted  = 0.0
	
	print ('Running MCMC ...')
	# Metropolis-Hastings with n iterations.
	
	for n in range(iterations):
		old_params  = guess_burnin #A[len(A)-1]  # old parameter value as array
		#print old_params
		#print ('old_params', old_params)
		if apex == True:
			old_loglik = lnprob_apex(old_params, data, yerr_i, ndim, A10, **kwargs)
		if planck == True:
			old_loglik = lnprob_planck(old_params, data, yerr_i, ndim, A10, **kwargs)
		# Suggest new candidate from Gaussian proposal distribution.
		#print (old_loglik)
		new_params = np.zeros([len(old_params)])
		#for i in range(len(old_params)):
			# Use stepsize provided for every dimension.
			# new_params[i] = random.gauss(old_params[i], stepsizes[i])
			#new_params[i] = random.gauss(old_params[i], cov_proposal[n,i])
			#new_params[i] = old_params[i]+cov_proposal[n,i]
		if ndim ==1:
			new_params = np.random.normal(old_params,cov)
		else:
			new_params= np.random.multivariate_normal (old_params,cov)
		#print ('new params', new_params)
		#ax1.plot(x[n+burnin],new_params[1], 'ro')
		if apex == True:
			new_loglik = lnprob_apex(new_params, data, yerr_i, ndim, A10, **kwargs)
		if planck == True:
			new_loglik = lnprob_planck(new_params, data, yerr_i, ndim, A10, **kwargs)
		# Accept new candidate in Monte-Carlo fashing.
		if (new_loglik > old_loglik):
			A[n,:] = (new_params)
			#ax1.plot(x[n+burnin],new_params[1], 'go')
			like_MCMC[n]=new_loglik
			guess_burnin = new_params
			accepted = accepted + 1.0  # monitor acceptance
		else:
			u = random.uniform(0.0,1.0)
			if (u < np.exp(new_loglik - old_loglik)):
				A[n,:] =(new_params)
				#ax1.plot(x[n+burnin],new_params[1], 'go')
				like_MCMC[n]=new_loglik
				guess_burnin = new_params
				accepted = accepted + 1.0  # monitor acceptance
			else:
				A[n,:]=(old_params)
				guess_burnin= old_params
				like_MCMC[n]=old_loglik
		if n%per_factor ==0:
			f = open (datafile, "a")
			if ndim ==1:
				f.write("{0:4d} {1:f} {2:f} \n".format(n, guess_burnin[0], like_MCMC[n]))
			if ndim ==3:
				f.write("{0:4d} {1:f} {2:f} {3:f} {4:f} \n".format(n, guess_burnin[0],guess_burnin[1],guess_burnin[2] , like_MCMC[n]))
			if ndim ==4:
				f.write("{0:4d} {1:f} {2:f} {3:f} {4:f} {5:f} \n".format(n, guess_burnin[0],guess_burnin[1],guess_burnin[2] , guess_burnin[3],  like_MCMC[n]))
			f.close()
		#if (n+1)%per_factor==0:
			print ("{0:5.1%}".format(float(n)/iterations))
			#pylab.pause(0.01)
			time.sleep(0.01)
		#print A[n,:]
		#print like_MCMC[n]


	# 	#fig.savefig('MH_plots/rc_param_pos_covproposal.png',dpi=300)
	maf = accepted/iterations
	print "Acceptance rate = "+str(maf)
	

	tau = 0
	#maf = 0

	
	print("Done.")	

	try:
		A= np.concatenate((A_file[burnin:,:], A), axis=0)
		samples = A
		lnprob_sampler = np.concatenate((like_file[burnin:], like_MCMC), axis=0)
		print ('combining previous run')
	except NameError:
		samples = A
		lnprob_sampler = like_MCMC


	return samples, lnprob_sampler, maf, tau



def run_mcmc_combap_multi(datafile, theta, data_a, data_p, **kwargs):
	#kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500, resolution
	import emcee
	import os
	import random
	import sys
	import time
	import acor

	nwalkers = kwargs['nwalkers']
	ite = kwargs['ite']
	threads = kwargs['threads']
	ndim = kwargs['ndim']
	A10 =  kwargs['A10']
	burnin = int(ite/3)
	per_factor = 100
	#f = open('comb17A1835_likelihoods.dat', "w")
	#f.close()

	try:
		dataa = data_a.meannan_rad
		cov_i_a = data_a.cov_selec
	except:
		dataa = None
		cov_i_a = None

	try:
		datap = data_p.meannan_rad
		cov_i_p = data_p.cov_selec
	except:
		datap = None
		cov_i_p = None

	

	if os.path.isfile(datafile):
		ndim_s = ndim #(ndim-1)+len(data_a.z_selec)
		f= open(datafile, "r")
		file = list(f)
		f.close()
		diff = (nwalkers*ite - len(file))  
		ite_o = ite 
		if 0 < diff <= nwalkers:
			print ('a)')
			ite = round((nwalkers/diff))
			pos = np.zeros(shape=(nwalkers, ndim_s))
			file_pos = file[-nwalkers::]
			for line, k  in zip(file_pos, range(nwalkers)):
				pos[k,:] = line.split()[0:-1]
				
			if burnin*nwalkers < len(file): 
				burnin = 0
			else:
				burnin = int(ite/3.)
				print ("New Burnin", burnin)
			print ("Restarting MCMC - iterations left = ", ite)
		
		elif diff > nwalkers:
			missing_walkers = nwalkers - (diff//ite)
			ite = round(diff/nwalkers)
			pos = np.zeros(shape=(nwalkers, ndim_s))
			file_pos = file[-nwalkers::]
			for line, k  in zip(file_pos, range(int(nwalkers-missing_walkers ))):
				pos[k,:] = line.split()[0:-1]
			for m in range (int(missing_walkers)):
				pos[-1-m,:] = pos[m,:]+ 1e-1*np.random.randn(ndim)
			print ("Restarting MCMC - iterations left = ", ite)
			
		elif diff <= 0:
			print ("MCMC Run has been completeted")
			sys.exit(0)

		
	else :
		
		#define initial position of the walkers

		#param1 = np.random.uniform(A10[0]-0.1,A10[0]+0.1, len(data_a.name_selec)) 

		param1 = np.random.uniform(A10[0]-0.1,A10[0]+0.1)
		if ndim ==1:

			otherparams = []
		elif ndim ==2:
			otherparams= random.uniform(A10[1]-0.5,A10[1]+0.5)
		elif ndim ==3:
			otherparams = random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)
		elif ndim ==4: 
			if 'dclevel' in kwargs:
				print ('Sampling for DC-Level')
				otherparams = random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5), random.uniform(-0.05,+0.05)#, random.randint(1,2)]
			else:
				otherparams =  random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)

		elif ndim ==5:
			
			if 'fitcenter' in kwargs:
				print ('Fitting for the x and y center position of the clusters')
				xc, yc = kwargs.get('fitcenter')
				otherparams = random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5), random.randint(xc-5,xc+5), random.randint(yc-5,yc+5)
			else:
				otherparams = random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[4]-0.5,A10[4]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)
			
		p0 = np.hstack((param1, otherparams))
		ndim_s = len(p0) # the number of dimensions for creating the sampler. This will be the total number of clusters combined + the extra parameters that you want to measure (c500, beta)
		
		pos = [ p0+ 1e-1*np.random.randn(ndim_s) for i in range(nwalkers)]

		#pos = np.load('GNFW_chains/poc500_w200_i100_apexnoise_TF.npy')[:,49,:]

		print ("Number of dimensions for mcmc = %0.i"%len(p0))

		f = open(datafile, "w")
		f.close()
	
	if kwargs['like_a'] and kwargs['like_p']  == True:
		print("Running MCMC... APEX+PLANCK")
	elif kwargs['like_a'] == True and kwargs['like_p'] ==False:
		print("Running MCMC... APEX")
	elif kwargs['like_p'] ==True and kwargs['like_a'] == False:
		print("Running MCMC... PLANCK")

	



	if 'threads' in kwargs: del kwargs['threads']
	if 'nwalkers' in kwargs: del kwargs['nwalkers']
	if 'ite' in kwargs: del kwargs['ite']
	if 'maf' in kwargs: del kwargs['maf']
	if 'tau' in kwargs: del kwargs['tau']

	
	if 'stack' in kwargs:
		sampler = emcee.EnsembleSampler(nwalkers, ndim_s, stack_l, args=(dataa, datap, cov_i_a, cov_i_p), kwargs=(kwargs), threads=threads) 
			# Clear and run the production chain.
	else:
		sampler = emcee.EnsembleSampler(nwalkers, ndim_s, combl_multi, args=(dataa, datap, cov_i_a, cov_i_p), kwargs=(kwargs), threads=threads) 
			# Clear and run the production chain.			
	


	for i, result in enumerate(sampler.sample(pos, iterations=ite, rstate0=np.random.get_state(), storechain=True)):
		if i%per_factor ==0:
			position = result[0]
			likelihood = result[1]
			f = open (datafile, "a+")
			for k in range(position.shape[0]):
				results_k = np.hstack((result[0][k], result[1][k]))
				np.savetxt(f,np.atleast_2d(results_k), fmt='%.6f')
			f.close()

		#if (i+1)%per_factor==0:
			print ("{0:5.1%}".format(float(i)/ite))
			time.sleep(0.01)

	print ("Done.")		
	samples = sampler.chain[:, burnin:, :].reshape((-1, ndim_s))
	chains = sampler.chain[:, :, :]
	chains_re = chains.reshape((-1, ndim_s))
	lnprob_sampler=sampler.lnprobability[:,burnin:].reshape((-1, 1))
	

	maf = (np.mean(sampler.acceptance_fraction))
	try:
		tau, mean, sigmat = acor.acor(samples[:,0])
		#tau = emcee.autocorr.integrated_time(samples, axis=0)[0]
	except:
		tau = 100
	#tau = emcee.autocorr.integrated_time(samples, axis=0)[0]
	#tau = 100
	print("Mean acceptance fraction: {0:.3f}"
				.format(maf))
	print ("Autocorrelation time:", tau)

	if threads >1:
		sampler.pool.terminate()
	sampler.reset()

	return samples, lnprob_sampler, maf, tau


def run_mcmc_combap(datafile, theta, data_a, data_p, yerr_ia, yerr_ip, ndim, A10, **kwargs):
	#kwargs = nwalkers, ite, trans_fun, rmax, width, z, pixindeg, freq_obs, m500, r500, resolution
	import emcee
	import os
	import random
	import sys
	import time
	import acor

	nwalkers = kwargs['nwalkers']
	ite = kwargs['ite']
	threads = kwargs['threads']
	burnin = int(ite/3)
	per_factor = 100

	

	if os.path.isfile(datafile):
		f= open(datafile, "r")
		file = list(f)
		f.close()
		diff = (nwalkers*ite - len(file))  
		ite_o = ite 
		if 0 < diff <= nwalkers:
			ite = round((nwalkers/diff))
			pos = np.zeros(shape=(nwalkers, ndim))
			file_pos = file[-nwalkers::]
			for line, k  in zip(file_pos, range(nwalkers)):
				pos[k,:] = line.split()[0:-1]
				
			if burnin*nwalkers < len(file): 
				burnin = 0
			else:
				burnin = int(ite/3.)
				print ("New Burnin", burnin)
			print ("Restarting MCMC - iterations left = ", ite)
		
		elif diff > nwalkers:
			missing_walkers = nwalkers - (diff//ite)
			ite = round(diff/nwalkers)
			pos = np.zeros(shape=(nwalkers, ndim))
			file_pos = file[-nwalkers::]
			for line, k  in zip(file_pos, range(int(nwalkers-missing_walkers ))):
				pos[k,:] = line.split()[0:-1]
			for m in range (int(missing_walkers)):
				pos[-1-m,:] = pos[m,:]+ 1e-1*np.random.randn(ndim)
			print ("Restarting MCMC - iterations left = ", ite)
			
		elif diff <= 0:
			print ("MCMC Run has been completeted")
			sys.exit(0)

	else :
		
		#define initial position of the walkers
		if ndim ==1:
			p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1)]
		elif ndim ==2:
			p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(A10[1]-0.5,A10[1]+0.5)]
		elif ndim ==3:
			p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]#
		elif ndim ==4:
			if 'dclevel' in kwargs:
				print ('Sampling for DC-Level')
				p0 = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5), random.uniform(-0.05,+0.05)]#, random.randint(1,2)]
			else:
				p0 = [random.uniform(A10[0]-0.1,A10[0]+0), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]
		elif ndim ==5:
			p0 = [random.uniform(A10[0]-0.1,A10[0]+0.1), random.uniform(A10[1]-0.5,A10[1]+0.5), random.uniform(A10[4]-0.5,A10[4]+0.5), random.uniform(A10[2]-0.5,A10[2]+0.5), random.uniform(A10[3]-0.5,A10[3]+0.5)]#, 
		
		pos = [ p0+ 1e-1*np.random.randn(ndim) for i in range(nwalkers)]
		#pos = np.load('GNFW_chains/poc500_w200_i100_apexnoise_TF.npy')[:,49,:]

		f = open(datafile, "w")
		f.close()
	

	print("Running MCMC... APEX+PLANCK")

	



	if 'threads' in kwargs: del kwargs['threads']
	if 'nwalkers' in kwargs: del kwargs['nwalkers']
	if 'ite' in kwargs: del kwargs['ite']
	if 'maf' in kwargs: del kwargs['maf']
	if 'tau' in kwargs: del kwargs['tau']

	
	

	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_combap, args=(data_a, data_p, yerr_ia, yerr_ip, ndim, A10), kwargs=(kwargs), threads=threads) 
			# Clear and run the production chain.
	


	for i, result in enumerate(sampler.sample(pos, iterations=ite, rstate0=np.random.get_state(), storechain=True)):
		if i%per_factor==0:
			position = result[0]
			likelihood = result[1]
			if 'y_prior' in kwargs['kwargs_apex']:
				y = result[3]
			f = open (datafile, "a")
			for k in range(position.shape[0]):
				results_k = np.hstack((result[0][k], result[1][k]))
				np.savetxt(f,np.atleast_2d(results_k), fmt='%.6f')
			f.close()

		#if (i+1)%per_factor==0:
			print ("{0:5.1%}".format(float(i)/ite))
			time.sleep(0.01)

	print ("Done.")		

	samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
	chains = sampler.chain[:, :, :]
	chains_re = chains.reshape((-1, ndim))
	lnprob_sampler=sampler.lnprobability[:,burnin:].reshape((-1, 1))

	if len(sampler.blobs) >0:
		Y_vals = np.asarray(sampler.blobs)[burnin:,:].reshape(-1,1)
	else:
		Y_vals =  []

	maf = (np.mean(sampler.acceptance_fraction))
	try:
		tau, mean, sigmat = acor.acor(samples[:,0])
		#tau = emcee.autocorr.integrated_time(samples, axis=0)[0]
	except:
		tau = 100
	
	#tau = emcee.autocorr.integrated_time(samples, axis=0)[0]
	#tau = 100
	print("Mean acceptance fraction: {0:.3f}"
				.format(maf))
	print ("Autocorrelation time:", tau)

	if threads >1:
		sampler.pool.terminate()

	sampler.clear_blobs
	sampler.reset()

	return samples, lnprob_sampler, maf, tau, Y_vals

def run_grid(datafile, theta, data, yerr_i, ndim, A10, apex = False, planck = False, **kwargs):

	import multiprocessing as mu
	from joblib import Parallel, delayed

	threads = kwargs['threads']

	
	print ('Creating Grid')
	print ('Very High precission')

	# Pei = np.arange(8.400,8.405,0.001)#np.arange(4,15,0.1)
	# m500 = np.arange(10,17,0.1)
	# c500 = np.arange(1.077,1.377,0.001)#np.arange(0.5,2.5,0.5)
	# b = np.arange(2,10,0.5)
	
	Pei = np.logspace(np.log10(2),np.log10(50),30)#np.arange(4,15,0.1)
	c500 = np.logspace(np.log10(0.1),np.log10(12),30)#np.arange(0.5,2.5,0.5)
	b = np.logspace(np.log10(2),np.log10(11),25)

	# Pei = np.arange(6,10,1)
	# c500 = np.arange(0.5,2.5,1)
	# b = np.arange(4,6,1)

	if ndim == 1:
		grid = np.array(np.meshgrid(Pei)).T.reshape(-1,1)
	if ndim ==2:
		if 'betac500' in kwargs:
			print ('Making grid for c500 and beta')
			grid = np.array(np.meshgrid(c500,b)).T.reshape(-1,2)
		else:
			grid = np.array(np.meshgrid(Pei,c500)).T.reshape(-1,2)

	if ndim ==3:
		if 'm500fit' in kwargs:
			print ('Making grid for M500, c500 and beta')
			grid = np.array(np.meshgrid(m500,c500,b)).T.reshape(-1,3)
		else:
			grid = np.array(np.meshgrid(Pei,c500,b)).T.reshape(-1,3)

	print('Grid Done!', grid.shape[0], 'Models')
	print('Computing the Likelihood')

	if apex == True:
		prob=Parallel(verbose=1, n_jobs=threads, backend='threading')(delayed(lnprob_apex)( grid[j,:],data, yerr_i, ndim, A10, **kwargs) for j in np.arange(grid.shape[0])) 

	if planck == True:
		prob=Parallel(verbose=1, n_jobs=threads)(delayed(lnprob_planck)( grid[j,:],data, yerr_i, ndim, A10, **kwargs) for j in np.arange(grid.shape[0])) 

	
	maf = np.nan
	tau = np.nan
	Y_vals = []

	return grid, prob, maf, tau


def pval_post (file, model, cov_i, A10, num, apex=False, Planck=False, **kwargs):
	""" Calculates the posterior predictive p-value"""

	import numpy as np 

	# read chain and likelihood from file

	chain = np.load(file)['chain'][:,0:-1]
	likelihood = np.exp(np.load(file)['lnlikelihood']) #  for this aproach you must take the exp of the lnlikelihood

	# compute the posterior probability density function (L.B.Lucy 2018)

	lam = likelihood/np.sum(likelihood) #!!!!! CONFIMR THAT THS IS THE CORRECT WAY TO DO THIS

	alpha_random = np. random.choice(chain, num)

def gamma(x):
	from scipy.integrate import quad
	#x = dof/2
	return quad(lambda t:(t**(x-1) * np.exp(-t)), 0, np.inf)[0]

def pval(dof, chi):
	from scipy.integrate import quad
	f_bottom = 2**(dof/2) * gamma(dof/2)
	return quad(lambda x: (x**(dof/2 -1) * np.exp(-x/2))/f_bottom, chi, np.inf)[0]
