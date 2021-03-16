def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '%1i' % (x)



def tri(files, absolute = False, m500fit = False, Y_out = True, alphafit=False, meta_chain = False,multiple=False, **kwargs):
	"""
	**kwargs:
			xlim_extra:  x range for the derived parameter (Y) e.g. [(15,45)]
			truth_extra: true value of the derived parameter (Y)
			legend: list of legends e.g. ['legend1', 'legend2']
			legend_coord': coordinates for the legend location. Default is (-0.15, 0.75)

			"""
	import mycorner as corner
	import numpy as np 
	import matplotlib as mpl
	from matplotlib.ticker import MaxNLocator
	from matplotlib.ticker import ScalarFormatter
	import matplotlib.lines as mlines

	reload(corner)
	
	if meta_chain:
		files = meta_chain_maker(files, multiple=multiple, alphafit=alphafit)[0]


	if type(files) == list:
		try:
			chain = np.load(files[0])['chain']
		except:
			chain = files[0]
	else:
		try:
			chain = np.load(files)['chain']
		except:
			chain = files
		files = [files]

	#Definitions
	ndim = chain.shape[1] - 1 if Y_out else chain.shape[1]
	ndim = 3 if alphafit else ndim
	A10=[8.403 ,1.177, 1.0510, 5.4905,0.3081] #GNFW values
	oplot = True if len(files) >1 else False
	glass = color_maker(0,oplot=oplot)[0]
	handles = []
	labels = []
	colors = []

	
	#general plotting params
	mpl.rcParams['xtick.labelsize'] = 16
	mpl.rcParams['ytick.labelsize'] = 16
	mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans'
	mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

	
	#arguments for the corner function
	if 'labels' in kwargs:
		labels_plot = kwargs.get('labels')
	else:
		if ndim == 1:
			labels_plot = [r"$P_0$"]
		elif ndim ==2:
			labels_plot = [r"$P_0$", r"$c_{500}$"]
		elif ndim == 3:
			labels_plot = [r"$P_0$", r"$c_{500}$", r"$\beta$"]
		elif ndim == 4:
			labels_plot = [r"$P_0$", r"$c_{500}$", r"$\alpha$",r"$\beta$"]
		elif ndim == 5:
			labels_plot = [r"$P_0$", r"$c_{500}$", r"$\gamma$",r"$\alpha$",r"$\beta$"]
		else:
			labels_plot = ['']*ndim

		if m500fit:
			labels_plot[0]=r"$M_{500}$"

	if 'truths' in kwargs:
		truths_in = kwargs.get('truths')
		print(truths_in)
	else:
		if ndim == 3:
			truths_in = [A10[0], A10[1], A10[3]]
		elif ndim == 4:
			truths_in = [A10[0], A10[1], A10[2], A10[3]]
		else:
			truths_in = [np.nan]*ndim
		
		if m500fit:
			truths_in[0] = kwargs.get('M500', '')

	if 'range' in kwargs:
		range_in = kwargs.get('range')
	else:
		if ndim ==3:
			range_in = [(0,40),(0,8),(0,15)] 
		elif ndim ==4:
			range_in = [(0,40),(0,8),(0,6),(0,15)] 
		else:
			range_in = None
	bins_num = kwargs.get('bins',45)#Number of bins to use for the hist AND contours. round(len(chain)**(1/3.))
	levs = [0.683, 0.954]
	st = False if oplot else True #show titles
	title_fmt = ".2f"
	q68 = False if oplot else True #plot 68% interval
	col_t = 'black' if oplot else 'red'
	smooth_val = 1
	label_kw = {"fontsize": 18, "weight":'bold'}
	title_kw = {"fontsize": 14, "weight":'semibold', "verticalalignment":'bottom'}
	hist_kw = {"log":False}
	cont_kw ={ "colors":[glass, glass]} if oplot else { "colors":["black", "black"]} 

	for num, f in enumerate(files):

		#Open the file that contains the chain information
		try:
			chain = np.load(f)['chain']
		except:
			chain = f

		
		#If mirror effect was used for the fit, then set absolute to true.
		if absolute == True:
			chain = abs(chain)

		#If Y is included on the chain then plot it on a top right corner or the plot

		if Y_out:
			Y_vals = chain[:,-1]
			chain = chain[:,0:-1]

		#If alpha is one of the parameters of the chain but wnats to be taking out
		if alphafit and meta_chain==False:
			if chain.shape[1] >= 4:
				chain = np.vstack((chain[:,0], chain[:,1], chain[:,3]))
				chain= chain.T

		if 'colors' in kwargs:
			from matplotlib.colors import colorConverter
			alpha = 0.5
			c = kwargs.get('colors')[num]
			c = list(colorConverter.to_rgba(c))
			cl = list(colorConverter.to_rgba(c))
			cl[-1] = alpha
		else:
			c, cl = color_maker(num,oplot=oplot)[1:]
		
		col_hist = c
		contf_kw = { "colors":[glass, cl,c], "cmap":None}
		cont_kw ={ "colors":[cl, c]} if oplot else { "colors":["black", "black"]}

		if num == 0:
			#plotting
			fig, axes =corner.corner(chain, labels=labels_plot,truths=truths_in, bins=bins_num,quantiles_68=q68,  label_kwargs=label_kw, show_titles=st,title_fmt=title_fmt, title_kwargs=title_kw,levels=levs, verbose=False, plot_datapoints=False, fill_contours=True, plot_density=True, color=col_hist,contour_kwargs=cont_kw, contourf_kwargs=contf_kw, truth_color=col_t,  hist_kwargs=hist_kw, range=range_in)
		else:
			corner.corner(chain, labels=labels_plot,truths=truths_in, bins=bins_num,  label_kwargs=label_kw, levels=levs, verbose=False, plot_datapoints=False, fill_contours=True, plot_density=True, color=col_hist,contour_kwargs=cont_kw, contourf_kwargs=contf_kw, truth_color=col_t,  hist_kwargs=hist_kw, range=range_in, fig=fig)
	
		if Y_out:
			ax=axes[0,-1]
			range = kwargs.get('xlim_extra',[(Y_vals.min(), Y_vals.max())])
			if len(range)>1:
				range = range[num]
			else:
				range = range[0]
			n,nbins = np.histogram(Y_vals, bins=bins_num, range=np.sort(range))
			n = n/float(n.max())
			hist_kw.pop('histtype', '')
			hist_kw.pop('log', '')
			hist_kw['color']= c
			ax.step(nbins[0:-1],n, where='post', **hist_kw)
			if oplot == False:
				qvalues = corner.mostcompact_68(Y_vals)
				for q in qvalues:
					ax.axvline(q, ls="dotted", color='black') #color

				q_16, q_50, q_84 = qvalues
				q_m, q_p = q_50-q_16, q_84-q_50
				title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
				label = r"$Y_{sph}$"
				fmt = "{{0:{0}}}".format(title_fmt).format
				title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
				title = "{0} = {1}".format(label, title)
				ax.set_title(title,  **title_kw)

		#legend format
		colors.append(c)

	#Final details of the Y histogram
	if Y_out: 
		ax.set_frame_on(True)
		label = kwargs.get('label_extra',r"$Y_{sph}$")
		truth_Y = kwargs.get('truth_extra', np.nan)
		max_n_ticks = 5
		use_math_text = False
		ax.axvline(truth_Y, ls="solid", color=col_t)
		ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
		ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
		ax.set_yticklabels([])
		[l.set_rotation(45) for l in ax.get_xticklabels()]
		ax.set_xlabel(label, **label_kw)
		ax.xaxis.set_label_coords(0.5, -0.25)
		ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text))
		ax.get_xaxis().set_tick_params(direction='in', bottom='on', top='on')
		ax.get_yaxis().set_tick_params(direction='in', left='on', right='on')
		#ax.set_xlim(kwargs.get('xlim_extra',(0,100)))
	
	#legend format
	if ndim==1:
		ax=axes
	else:
		ax = axes[0,-1]
	label =(kwargs.get("legend", " "*len(colors)))
	for i, c in enumerate(colors):
		l = label[i]
		if len(l)>1:
			labels.append(l)
			patch = mlines.Line2D([], [],color=c, 
                          markersize=5) #if 'legends' in kwargs else ''
			handles.append(patch)
	eclb = 'white' #edge color legend box
	if ndim ==1:
		legend_coord = (1.70, 0.75)
	elif ndim ==2:
		legend_coord = (0.90, 0.75)
	else:
		legend_coord = (-0.15, 0.75)
	ax.legend(handles, labels, numpoints=1, loc='upper right', markerscale=10, bbox_to_anchor=(kwargs.get('legend_coord', legend_coord)), fontsize=16, fancybox=True,edgecolor=eclb, handletextpad=0.7,borderaxespad=-0.2) # bbox_to_anchor=(1,-0.5) - for box on the empyt space on the right , #(kwargs.get('legend_coord', (-0.15, 0.75) if ndim != 2 else (0.90, 0.75)))

	#Final details of the plot
	fig.subplots_adjust(top=0.85)
	fig.suptitle(kwargs.get('title', ''), fontsize=18)

	return fig,axes


def plot_chain(file, ndim, **kwargs):

	import numpy as np 
	import pylab
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	pylab.ion()

	if type(file) == str:
		chain = np.load(file)['chain']
	else:
		chain = file

	A10=[8.403 ,1.177, 1.0510, 5.4905,0.3081] #GNFW values
	mpl.rcParams['xtick.labelsize'] = 14
	mpl.rcParams['ytick.labelsize'] = 14
	mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans'
	mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
	
	
	fig, ax = plt.subplots(ndim,1, sharex=True)


	if ndim ==1:
		ax.plot(chain[::100,0], c='gray')
		ax.axhline(A10[0], c='black')
		ax.set_ylabel(r'$P_0$', fontsize=16)
	

	if ndim ==3:
		ax[0].plot(chain[::100,0], c='gray')
		ax[1].plot(chain[::100,1], c='gray')
		ax[2].plot(chain[::100,2], c='gray')

		ax[0].axhline(A10[0], c='black')
		ax[1].axhline(A10[1], c='black')
		ax[2].axhline(A10[3], c='black')

		ax[0].set_ylabel(r'$P_0$', fontsize=16)
		ax[1].set_ylabel(r'$c_{500}$', fontsize=16)
		ax[2].set_ylabel(r'$\beta$', fontsize=16)

	if ndim ==4:
		ax[0].plot(chain[::100,0], c='gray')
		ax[1].plot(chain[::100,1], c='gray')
		ax[2].plot(chain[::100,2], c='gray')
		ax[3].plot(chain[::100,3], c='gray')

		ax[0].axhline(A10[0], c='black')
		ax[1].axhline(A10[1], c='black')
		ax[2].axhline(A10[2], c='black')
		ax[3].axhline(A10[3], c='black')

		ax[0].set_ylabel(r'$P_0$', fontsize=16)
		ax[1].set_ylabel(r'$c_{500}$', fontsize=16)
		ax[2].set_ylabel(r'$\alpha$', fontsize=16)
		ax[3].set_ylabel(r'$\beta$', fontsize=16)

	

	fig.suptitle(kwargs.get('title', ''), fontsize=18)
	fig.tight_layout()
	fig.subplots_adjust(top=0.85)
		

def gnfw_plot(file, ndim, cluster=None, oplot =False, oplot_planck=False, m500fit = False, only_shape=False, **kwargs):

	"""This function plots the pressure profiles for a given cluster assuming gnfw in the list cluster_info_dict['tot_myclass']. It shows  the highest likelihood pressure profile obtained from a fit and the  A10 profile.
	Args:
		file:  npz file that include the likelihood and the parameters (eg. chain and lnlikelihood ).
		ndim: number of parameter dimensions (excluding Y).
		cluster: the number of the list position of the cluster.
		oplot: over plot - show more than one profile.
		oplot_planck: NOT WORKING WELL THE 68% SHADED REGION!!. over plots a pressure profile for the Planck data and if desired an A10_mod in kwargs can be given for the assumption of the parameters not fit e.g. give Planck parameter values. Does not work when only_shape is True.
		m500fit: Boolean. If the values for the first column of the chain correspond to m500 instead of P0.Default False
		only_shape: Boolean. Does not assume any mass or z for the pressure profile and only plots the shape of it. Does not work when oplot_planck is True. Default is False
		kwargs: m500, m500fit, legend, A10_mod, title, colors
	Returns:
		A10 plot in red, overplot each of the best fit profiles ( highest likelihood) from the files list"""

	import numpy as np
	import GNFW_functions as fun 
	#from cluster_info import list_clus_wlm as lc
	import cluster_info_dict as cid
	lc = cid.dic_lc['tot_myclass']
	import pylab
	import matplotlib.pyplot as plt
	from matplotlib.ticker import ScalarFormatter
	import matplotlib 
	

	pylab.ion()

	reload(fun)

	
	#if 'm500' in kwargs:del kwargs['m500']

	try:
		chain = np.load(file)['chain']
		lnlike = np.load(file)['lnlikelihood']
	except:
		chain = file
		lnlike = kwargs.get('lnlikelihood')

	chain = chain[np.where(lnlike != -np.inf)[0], :]
	lnlike = lnlike[np.where(lnlike != -np.inf)[0]]
	
	A10=[8.403 ,1.177, 1.0510, 5.4905,0.3081] #GNFW values
	if only_shape == False:
		name = lc[cluster][0]
		m500 = kwargs.get('m500',lc[cluster][2])
		z = kwargs.get('z', lc[cluster][1])
		M500, r500_m = fun.mr500(z,m500=m500, r500=None) [0:2]
		pixindeg = np.nan#0.025 #0.002777778 - apex # 0.025 -planck ####!!!! Not importnatn for this plot
		kwargs.pop('m500','')
		kwargs.pop('z','')



	if ndim ==1:
		theta_0 = A10[0]
	elif ndim ==2:
		if betac500 ==True:
			theta_0 = A10[1], A10[3]
		else:
			theta_0 = A10[0], A10[1]
	elif ndim ==3:
		theta_0 = A10[0], A10[1], A10[3]
	elif ndim ==4:
		 theta_0 =A10[0], A10[1], A10[2],A10[3]

	if m500fit:
		kwargs['m500fit'] = True
		theta_0[0] = m500
	
		
	max_idx = np.argmax(lnlike)#np.where(lnlike.max()==lnlike)[0] 
	theta_max = chain[max_idx, 0:ndim]
	mc_num = int(len(chain)*0.01)#1000#00

	prof_fin= []
	#x =np.logspace(0.01,2.5*r500_m,200, endpoint=True)
	
	#rg =np.linspace(0.01*r500_m/pixel_to_meters,1.5*r500_m/pixel_to_meters,200, endpoint=True)*pixel_to_meters
	#rg =np.linspace(0.01*r500_m,1.5*r500_m,200, endpoint=True)
	if only_shape:
		r =np.linspace(0,5,200, endpoint=True)
	else:
		r =np.logspace(20,23.4,200, endpoint=True)

	#covf = np.cov(chain[:,0:3].T)
	#Pei_mean, c500_mean, beta_mean, Y_mean = chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean(),chain[:,3].mean()
	#Pei_mean, c500_mean, beta_mean = chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean()
	##theta_A10= np.random.multivariate_normal((chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean(),chain[:,3].mean()),covf,(mc_num))
	#theta_A10= np.random.multivariate_normal((chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean()),covf,(mc_num))
	theta_A10 = []

	for i in range (mc_num):
		index = np.random.randint(0,chain.shape[0])
		theta_A10.append(chain[index,0:ndim])
	theta_A10 = np.array(theta_A10)

	fig, ax = plt.subplots(figsize=(10,8))
	ax.set_yscale('log')
	ax.set_xscale('log')

	for axis in [ax.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	
	matplotlib.rcParams['mathtext.fontset'] = 'custom'
	matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
	matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans'
	matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

	ax.tick_params(labelsize = 16)

	if only_shape:
		ax.set_ylabel(r'$P$', fontsize = 16)
		ax.set_xlabel(r'$Radius$', fontsize = 16)
	else:
		ax.set_ylabel(r'$P/P_{500}$', fontsize = 16)
		ax.set_xlabel(r'$Radius\ (R/R_{500})$', fontsize = 16)

	if 'colors' in kwargs:
			from matplotlib.colors import colorConverter
			alpha = 0.5
			c = kwargs.get('colors')[0]
			c = list(colorConverter.to_rgba(c))
			cl = list(colorConverter.to_rgba(c))
			cl[-1] = alpha
	else:
			c, cl = color_maker(0,oplot=oplot)[1:]

	if only_shape:
		gnfw_prof = fun.gnfw_profile(r, ndim, A10, theta_0)
		#gnfw_prof = fun.gnfw_profile(r, ndim, A10, [6.41, 1.81,1.33,4.13 ])
	else:
		gnfw_prof = fun.p_profile(theta_0, ndim, A10, z, pixindeg, m500,r,**kwargs)[1]
		#gnfw_prof = fun.p_profile([6.41, 1.81,1.33,4.13 ], ndim, A10, z, pixindeg, m500,r,**kwargs)[1]
	if only_shape:
		r500_m = 1
	ax.plot(r/r500_m,gnfw_prof, c='red', label='A10 Parameters', zorder=3)
	#ax.plot(r/r500_m,gnfw_prof, c='red', label='Planck Parameters', zorder=3)

	if only_shape:
		maxl_prof = fun.gnfw_profile(r, ndim, A10, theta_max)
		label_max = ''
	else:
		maxl_prof = fun.p_profile(theta_max, ndim, A10, z, pixindeg, m500,r,**kwargs)[1]
		label_max = r'Maximum $L$'

	ax.plot(r/r500_m,maxl_prof, c=c, label=label_max, zorder=2)


	profile_max = gnfw_prof#*0
	profile_min = gnfw_prof#	*0

	for i in range(theta_A10.shape[0]):

		# if kwargs.get('m500fit'):
		# 	m500  = theta_A10[i,0]
		# 	M500, r500_m = fun.mr500(z,m500=m500, r500=None) [0:2]
		if only_shape:
			profile_mc =fun.gnfw_profile(r, ndim, A10, theta_A10[i,:])
		else:
			profile_mc =fun.p_profile(theta_A10[i,:], ndim, A10, z, pixindeg, m500,r, **kwargs)[1]

		#ax.plot(r/r500_m,profile_mc, c='wheat', alpha=0.3, ls='dashed')
		# if np.all(profile_max < profile_mc):
		# 	profile_max = profile_mc
		# if np.all(profile_min > profile_mc):
		# 	profile_min = profile_mc

		prof_fin.append(profile_mc)
		
	
	prof_fin = np.asarray(prof_fin)
	#print (prof_fin[:,0])
	range68 = np.zeros(shape=(3, prof_fin.shape[1]))

	for i in range (prof_fin.shape[1]):
		low, mid, up = np.percentile(prof_fin[:,i], [16, 50, 84])
		range68[:,i] = low, mid, up
		#print (range68[:,i])
	
	#print (range68[0,:], range68[-1,:])

	ax.fill_between(r/r500_m,range68[0,:], range68[2,:],  color=c, edgecolor='black', alpha = 0.45,label = kwargs.get('legend',[('68 %, Ndim = ' +str(ndim))]*ndim)[0], zorder=2)
	# if oplot == True:
	# 	ax.fill_between(r/r500_m,range68[0,:], range68[2,:],  color='steelblue', edgecolor='black', alpha = 0.3,label = '68 %, Ndim = 3', zorder=2)
	# else:
	# 	ax.fill_between(r/r500_m,range68[0,:], range68[2,:],  color='steelblue', edgecolor='black', alpha = 0.3,label = '68 %', zorder=2)
		#ax.fill_between(r/r500_m, profile_max, profile_min, color='grey', alpha = 0.5)

	error = profile_max - gnfw_prof, gnfw_prof - profile_min

	if oplot ==True:
		prof_fin = []
		ndim = ndim #+1
		pylab.pause(0.1)
		file2 = kwargs['file2']
		try:
			chain = np.load(file2)['chain']
			lnlike = np.load(file2)['lnlikelihood']
		except:
			chain = file2
			lnlike = kwargs.get('lnlikelihood2')

		ndim = kwargs.get('ndim2', ndim)
		mc_num = int(len(chain)*0.01)
		# covf = np.cov(chain[:,0:3].T)
		# Pei_mean, c500_mean, beta_mean = chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean()
		# theta_A10= np.random.multivariate_normal((Pei_mean,c500_mean, beta_mean),covf,(mc_num))


		if 'colors' in kwargs:
			from matplotlib.colors import colorConverter
			alpha = 0.5
			c = kwargs.get('colors')[1]
			c = list(colorConverter.to_rgba(c))
			cl = list(colorConverter.to_rgba(c))
			cl[-1] = alpha
		else:
			c, cl = color_maker(1,oplot=oplot)[1:]
		
		theta_A10 = []
		for i in range (mc_num):
			index = np.random.randint(0,chain.shape[0])
			theta_A10.append(chain[index,0:ndim])
		
		theta_A10 = np.array(theta_A10)

		profile_max = gnfw_prof
		profile_min = gnfw_prof

		max_idx = np.argmax(lnlike)#np.where(lnlike.max()==lnlike)[0] 
		theta_max = chain[max_idx, 0:ndim]

		if only_shape:
			maxl_prof = fun.gnfw_profile(r, ndim, A10, theta_max)
		else:
			maxl_prof = fun.p_profile(theta_max, ndim, A10, z, pixindeg, m500,r,**kwargs)[1]

		ax.plot(r/r500_m,maxl_prof, c=c, label='', zorder=2)

		for i in range(theta_A10.shape[0]):
			if only_shape:
				profile_mc =fun.gnfw_profile(r, ndim, A10, theta_A10[i,:])
			else:
				profile_mc =fun.p_profile(theta_A10[i,:], ndim, A10, z, pixindeg, m500,r,**kwargs)[1]

			prof_fin.append(profile_mc)
			#ax.plot(r/r500_m, profile_mc, c='grey', alpha = 0.3)
		prof_fin = np.asarray(prof_fin)
		#print (prof_fin[:,0])
		range68 = np.zeros(shape=(3, prof_fin.shape[1]))

		for i in range (prof_fin.shape[1]):
			low, mid, up = np.percentile(prof_fin[:,i], [16, 50, 84])
			range68[:,i] = low, mid, up
		
		ax.fill_between(r/r500_m,range68[0,:], range68[2,:],  color=c, edgecolor='black', alpha = 0.4,label = kwargs.get('legend',[('68 %, Ndim = ' +str(ndim))]*ndim)[1], zorder=2)		

		error = profile_max - gnfw_prof, gnfw_prof - profile_min

		try:
			prof_fin = []
			ndim = ndim #+1
			pylab.pause(0.1)
			file3 = kwargs['file3']
			try:
				chain = np.load(file3)['chain']
				lnlike = np.load(file3)['lnlikelihood']
			except:
				chain = file3
				lnlike = kwargs.get('lnlikelihood3')
			ndim = kwargs.get('ndim2', ndim)
			mc_num = int(len(chain)*0.01)
			# covf = np.cov(chain[:,0:3].T)
			# Pei_mean, c500_mean, beta_mean = chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean()
			# theta_A10= np.random.multivariate_normal((Pei_mean,c500_mean, beta_mean),covf,(mc_num))


			if 'colors' in kwargs:
				from matplotlib.colors import colorConverter
				alpha = 0.5
				c = kwargs.get('colors')[2]
				c = list(colorConverter.to_rgba(c))
				cl = list(colorConverter.to_rgba(c))
				cl[-1] = alpha
			else:
				c, cl = color_maker(2,oplot=oplot)[1:]
			
			theta_A10 = []
			for i in range (mc_num):
				index = np.random.randint(0,chain.shape[0])
				theta_A10.append(chain[index,0:ndim])
			
			theta_A10 = np.array(theta_A10)

			profile_max = gnfw_prof
			profile_min = gnfw_prof


			max_idx = np.argmax(lnlike)#np.where(lnlike.max()==lnlike)[0] 
			theta_max = chain[max_idx, 0:ndim]

			if only_shape:
				maxl_prof = fun.gnfw_profile(r, ndim, A10, theta_max)
			else:
				maxl_prof = fun.p_profile(theta_max, ndim, A10, z, pixindeg, m500,r,**kwargs)[1]

			ax.plot(r/r500_m,maxl_prof, c=c, label='', zorder=2)

			for i in range(theta_A10.shape[0]):
				if only_shape:
					profile_mc =fun.gnfw_profile(r, ndim, A10, theta_A10[i,:])
				else:
					profile_mc =fun.p_profile(theta_A10[i,:], ndim, A10, z, pixindeg, m500,r,**kwargs)[1]

				prof_fin.append(profile_mc)
				#ax.plot(r/r500_m, profile_mc, c='grey', alpha = 0.3)
			prof_fin = np.asarray(prof_fin)
			#print (prof_fin[:,0])
			range68 = np.zeros(shape=(3, prof_fin.shape[1]))

			for i in range (prof_fin.shape[1]):
				low, mid, up = np.percentile(prof_fin[:,i], [16, 50, 84])
				range68[:,i] = low, mid, up
			

			ax.fill_between(r/r500_m,range68[0,:], range68[2,:],  color=c, edgecolor='black', alpha = 0.3,label = kwargs.get('legend',[('68 %, Ndim = ' +str(ndim))]*ndim)[2], zorder=2)		

			error = profile_max - gnfw_prof, gnfw_prof - profile_min
		except:
			file2 = []

	if oplot_planck ==True:
		pylab.pause(0.1)
		file2 = kwargs['file2']
		A10_mod = kwargs.get('A10mod', A10)
		chain = np.load(file2)['chain']
		covf = np.cov(chain.T)
		Pei_mean,  Y_mean = chain[:,0].mean(),chain[:,1].mean()
		theta_A10= np.random.multivariate_normal(np.mean(chain, axis=0),covf,(mc_num))
		
		profile_max = gnfw_prof
		profile_min = gnfw_prof
		prof_fin = []
		for i in range(theta_A10.shape[0]):

			profile_mc =fun.p_profile(theta_A10[i,0], 1, A10, z, pixindeg, m500,r,**kwargs)[1]

			if np.all(profile_max < profile_mc):
				profile_max = profile_mc
			if np.all(profile_min > profile_mc):
				profile_min = profile_mc

			prof_fin.append(profile_mc)
			#ax.plot(r/r500_m, profile_mc, c='grey', alpha = 0.3)

		ax.fill_between(r/r500_m, profile_max, profile_min, color='green', alpha = 0.3, label='Planck')

		error = profile_max - gnfw_prof, gnfw_prof - profile_min


	#fig.suptitle(str(name)+' - Sim 99', fontsize = 16)
	try:
		name=name
	except:
		name = ''
	fig.suptitle(kwargs.get('title',str(name)) , fontsize = 18)

	ax.legend(loc='best', fontsize=14)

	#return prof_fin, error


def gnfw_plot_p0(file, ndim, cluster, oplot =False, oplot_planck=False, **kwargs):
	
	import numpy as np
	import GNFW_functions as fun 
	import cluster_info_dict as cid
	lc = cid.dic_lc['total']
	import pylab
	import matplotlib.pyplot as plt
	from matplotlib.ticker import ScalarFormatter
	import matplotlib 
	

	pylab.ion()

	reload(fun)
	
	name = lc[cluster][0]
	m500 = lc[cluster][2]
	z = lc[cluster][1]
	M500, r500_m = fun.mr500(z,m500=m500, r500=None) [0:2]
	pixindeg = 0.002777778
	chain = np.load(file)['chain']
	A10=[8.403 ,1.177, 1.0510, 5.4905,0.3081] #GNFW values
	theta_0 = A10[0]#, A10[1], A10[3]
	mc_num = 1000

	prof_fin= []
	#x =np.logspace(0.01,2.5*r500_m,200, endpoint=True)
	
	#rg =np.linspace(0.01*r500_m/pixel_to_meters,1.5*r500_m/pixel_to_meters,200, endpoint=True)*pixel_to_meters
	#rg =np.linspace(0.01*r500_m,1.5*r500_m,200, endpoint=True)

	r =np.logspace(20,23.4,200, endpoint=True)

	covf = np.cov(chain[:,0].T)
	sigma = np.std(chain[:,0])
	#Pei_mean, c500_mean, beta_mean, Y_mean = chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean(),chain[:,3].mean()
	Pei_mean = chain[:,0].mean()
	#theta_A10= np.random.multivariate_normal((chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean(),chain[:,3].mean()),covf,(mc_num))
	#theta_A10= np.random.multivariate_normal((Pei_mean),covf,(mc_num))
	theta_A10= np.random.normal(Pei_mean, sigma, mc_num)

	fig, ax = plt.subplots()
	ax.set_yscale('log')
	ax.set_xscale('log')

	for axis in [ax.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	
	matplotlib.rcParams['mathtext.fontset'] = 'custom'
	matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
	matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans'
	matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

	ax.tick_params(labelsize = 12)

	ax.set_ylabel(r'$P/P_{500}$', fontsize = 14)
	ax.set_xlabel(r'$Radius (R500)$', fontsize = 14)

	gnfw_prof = fun.p_profile(theta_0, ndim, A10, z, pixindeg, m500,r)[1]
	#ax.plot(r/r500_m,gnfw_prof, c='red', label='A10 Parameters')
	profile_max = gnfw_prof
	profile_min = gnfw_prof	

	for i in range(theta_A10.shape[0]):

		profile_mc =fun.p_profile(theta_A10[i], ndim, A10, z, pixindeg, m500,r)[1]

		if np.all(profile_max < profile_mc):
			profile_max = profile_mc
		if np.all(profile_min > profile_mc):
			profile_min = profile_mc

		prof_fin.append(profile_mc)
		#ax.plot(r/r500_m, profile_mc, c='grey', alpha = 0.3)

	if oplot ==True:
		#ax.fill_between(r/r500_m, profile_max, profile_min,  hatch='X', edgecolor='blue', alpha = 0.3)
		ax.fill_between(r/r500_m, profile_max, profile_min,  color='blue', edgecolor='blue', alpha = 0.3)

	else:
		ax.fill_between(r/r500_m, profile_max, profile_min, color='grey', alpha = 0.5, label = 'APEX-SZ')

	error = profile_max - gnfw_prof, gnfw_prof - profile_min

	if oplot ==True:
		pylab.pause(0.1)
		file2 = kwargs['file2']
		chain = np.load(file2)['chain']
		covf = np.cov(chain.T)
		Pei_mean, c500_mean, beta_mean, Y_mean = chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean(),chain[:,3].mean()
		theta_A10= np.random.multivariate_normal((chain[:,0].mean(),chain[:,1].mean(), chain[:,2].mean(),chain[:,3].mean()),covf,(mc_num))
		
		profile_max = gnfw_prof
		profile_min = gnfw_prof
		for i in range(theta_A10.shape[0]):

			profile_mc =fun.p_profile(theta_A10[i,0:3], ndim, A10, z, pixindeg, m500,r)[1]

			if np.all(profile_max < profile_mc):
				profile_max = profile_mc
			if np.all(profile_min > profile_mc):
				profile_min = profile_mc

			prof_fin.append(profile_mc)
			#ax.plot(r/r500_m, profile_mc, c='grey', alpha = 0.3)

		ax.fill_between(r/r500_m, profile_max, profile_min, color='green', alpha = 0.3)

		error = profile_max - gnfw_prof, gnfw_prof - profile_min

	if oplot_planck ==True:
		pylab.pause(0.1)
		file2 = kwargs['file2']
		A10_mod = kwargs['A10mod']
		chain = np.load(file2)['chain']
		covf = np.cov(chain.T)
		Pei_mean,  Y_mean = chain[:,0].mean(),chain[:,1].mean()
		theta_A10= np.random.multivariate_normal((chain[:,0].mean(),chain[:,1].mean()),covf,(mc_num))
		
		profile_max = gnfw_prof
		profile_min = gnfw_prof
		for i in range(theta_A10.shape[0]):

			profile_mc =fun.p_profile(theta_A10[i,0], 1, A10, z, pixindeg, m500,r)[1]

			if np.all(profile_max < profile_mc):
				profile_max = profile_mc
			if np.all(profile_min > profile_mc):
				profile_min = profile_mc

			prof_fin.append(profile_mc)
			#ax.plot(r/r500_m, profile_mc, c='grey', alpha = 0.3)

		ax.fill_between(r/r500_m, profile_max, profile_min, color='green', alpha = 0.3, label='Planck')

		error = profile_max - gnfw_prof, gnfw_prof - profile_min


	#fig.suptitle(str(name)+' - Sim 99', fontsize = 16)
	fig.suptitle(str(name), fontsize = 16)

	ax.legend(loc='best')

	#return prof_fin, error

def gnfw_plot_multi(files, ndim, cluster, betac500=False, extreme=False, **kwargs):
	"""This function plots the pressure profiles for a given cluster in the list cluster_info_dict['total']. It shows  the highest likelihood pressure profile obtained from a fit and the  A10 profile.
	Args:
		files: list of npz files that include the likelihood and the paramters (eg. chain and lnlikelihood ).
		ndim: number of parameter dimensions (excluding Y).
		cluster: the number of the list position of the cluter.
		betac500: default False
		extreme: default False - chooses the line with the max and min P0, c500 and beta and plot them in color with label included
		kwargs: m500, m500fit
	Returns:
		A10 plot in red, overplot each of the best fit profiles ( highest likelihood) from the files list"""
	import numpy as np
	import GNFW_functions as fun 
	import cluster_info_dict as cid
	lc = cid.dic_lc['total']
	import pylab
	import matplotlib.pyplot as plt
	from matplotlib.ticker import ScalarFormatter
	import matplotlib 
	

	pylab.ion()

	reload(fun)

	if 'm500' in kwargs:del kwargs['m500']

	if betac500 == True:
		kwargs['betac500'] = True
	
	name = lc[cluster][0]
	m500 = lc[cluster][2]
	z = lc[cluster][1]
	M500, r500_m = fun.mr500(z,m500=m500, r500=None) [0:2]
	pixindeg = 0.025 #0.002777778 - apex # 0.025 -planck
	#chain = np.load(file)['chain']
	A10=[8.403 ,1.177, 1.0510, 5.4905,0.3081] #GNFW values
	if ndim ==1:
		theta_0 = A10[0]
	elif ndim ==2:
		if betac500 ==True:
			theta_0 = A10[1], A10[3]
		else:
			theta_0 = A10[0], A10[1]
	elif ndim ==3:
		theta_0 = A10[0], A10[1], A10[3]
	elif ndim ==4:
		 theta_0 =A10[0], A10[1], A10[2],A10[3]
	

	prof_fin= []
	r =np.logspace(20,23.4,200, endpoint=True)

	
	theta_A10 = []
	for i in range (len(files)):
		chain = np.load(files[i])['chain']
		lnlike = np.load(files[i])['lnlikelihood']
		index = np.where(lnlike == lnlike.max())[0][0]
		theta_A10.append(chain[index,0:ndim])
	theta_A10 = np.array(theta_A10)

	fig, ax = plt.subplots(figsize=(10.5,7))
	ax.set_yscale('log')
	ax.set_xscale('log')

	for axis in [ax.xaxis]:
		axis.set_major_formatter(ScalarFormatter())

	
	matplotlib.rcParams['mathtext.fontset'] = 'custom'
	matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
	matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans'
	matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

	ax.tick_params(labelsize = 12)

	ax.set_ylabel(r'$P/P_{500}$', fontsize = 14)
	ax.set_xlabel(r'$Radius\ (R/R_{500})$', fontsize = 14)

	gnfw_prof = fun.p_profile(theta_0, ndim, A10, z, pixindeg, m500,r, **kwargs)[1]
	ax.plot(r/r500_m,gnfw_prof, c='red', label='A10 Parameters', zorder=3)
	

	for i in range(theta_A10.shape[0]):

		if kwargs.get('m500fit'):
			m500  = theta_A10[i,0]
			M500, r500_m = fun.mr500(z,m500=m500, r500=None) [0:2]

		profile_hl =fun.p_profile(theta_A10[i,:], ndim, A10, z, pixindeg, m500,r, **kwargs)[1]

		ax.plot(r/r500_m, profile_hl, color='grey', alpha = 0.5)

	
	ax.plot(r/r500_m, profile_hl, color='grey', alpha = 0.5, label='Highest Likelihood')
	
	if extreme == True:
		idxl = []
		for l in range (theta_A10.shape[1]):
			idxl.append(np.argmax(theta_A10[:,l]))
			idxl.append(np.argmin(theta_A10[:,l]))
			
		for idx in idxl:
			profile_hl =fun.p_profile(theta_A10[idx,:], ndim, A10, z, pixindeg, m500,r, **kwargs)[1]
			text = r"""$P_0 = {0[0]:0.2f}, c_{{500}} = {0[1]:0.2f}, \beta = {0[2]:0.2f}$""".format(theta_A10[idx,:])
			ax.plot(r/r500_m, profile_hl, alpha = 0.8, ls = 'dashed', label=text)



	print (theta_A10)
	fig.suptitle(str(name) , fontsize = 16)

	ax.legend(loc='best')

	#return prof_fin, error

def ite_results(p0_sims,p0_err, c500_sims, c500_err, beta_sims, beta_err):

	import matplotlib.pyplot as plt
	import numpy as np
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import pylab

	pylab.ion()
	A10=[8.403 ,1.177, 1.0510, 5.4905,0.3081] #GNFW values
	cmap = 'copper'#'Greens'
	col = np.arange(len(p0_sims))+1
	size = 200 
	cbar_col = 'brown'
	cbar_alpha = 0.3

	fig, ax = plt.subplots(1,3, figsize=(18,10))
	for i in range(len(ax)):
		ax[i].tick_params(labelsize=12)

	if len(beta_err) ==2:
		beta_err_neg = beta_err[0]
		beta_err_pos = beta_err[1]
	else:
		beta_err_pos = beta_err
		beta_err_neg = beta_err 

	mark = ['x','<', '>', 's']

	ax[0].errorbar(p0_sims, c500_sims, xerr=p0_err, yerr=c500_err, ls='none', c=cbar_col, alpha = cbar_alpha)
	# for xp, yp, m, co  in zip(p0_sims, c500_sims, mark, col):
	# 	ax[0].scatter([xp], [yp], marker =m)
	ax[0].scatter(p0_sims, c500_sims, c=col, cmap =cmap, marker="o", s=size)
	ax[0].scatter(A10[0],A10[1], c='red', marker='^', s=size)
	ax[0].set_ylabel(r'$c_{500}$', fontsize='xx-large', fontweight='heavy')
	ax[0].set_xlabel(r'$P_0$', fontsize='xx-large', fontweight='heavy')

	ax[1].errorbar(p0_sims, beta_sims, xerr=p0_err, yerr=[beta_err_neg, beta_err_pos],ls='none', c=cbar_col, alpha = cbar_alpha)
	ax[1].scatter(p0_sims, beta_sims, c=col, cmap =cmap, marker="o", s=size)
	ax[1].scatter(A10[0],A10[3], c='red', marker='^', s=size)
	ax[1].set_ylabel(r'$\beta$', fontsize='xx-large', fontweight='heavy')
	ax[1].set_xlabel(r'$P_0$', fontsize='xx-large', fontweight='heavy')

	ax[2].errorbar(c500_sims, beta_sims, xerr=c500_err, yerr=[beta_err_neg, beta_err_pos],ls='none', c=cbar_col, alpha = cbar_alpha)
	cbar = ax[2].scatter(c500_sims, beta_sims, c=col, cmap =cmap, marker="o", s=size)
	ax[2].scatter(A10[1],A10[3], c='red', marker='^', s=size)
	ax[2].set_ylabel(r'$\beta$', fontsize='xx-large', fontweight='heavy')
	ax[2].set_xlabel(r'$c_{500}$', fontsize='xx-large', fontweight='heavy')

	divider = make_axes_locatable(ax[2])
	cbar_ax =  divider.append_axes("right", size="7%", pad = 0.2)
	cb = fig.colorbar(cbar, cax=cbar_ax, format='%.2f', ticks=col)
	cb.set_label('Iteration Number', fontsize=16, labelpad=10)
	#fig.tight_layout(h_pad=3)


	fig.suptitle(r'A1835 - Sim S/N $\approx$ 800 - Intermediate Results Iterative Code with Posterior Sampling', fontsize=16)

	#plt.savefig('Apex_figures/scatterplot_simshighsn_iterative_allresults_possampling', bbox_inches='tight')#, dpi = 300)#

def ite_res_lin(p0_sims,p0_err, c500_sims, c500_err, beta_sims, beta_err):

	import matplotlib.pyplot as plt
	import numpy as np
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import pylab

	pylab.ion()

	A10=[8.403 ,1.177, 1.0510, 5.4905,0.3081] #GNFW values

	col = np.arange(len(p0_sims))+1
	size = 150 
	cbar_col = 'green'
	cbar_alpha = 0.6
	ma = 'o'

	fig, ax = plt.subplots(1,3, figsize=(18,10))
	for i in range(len(ax)):
		ax[i].tick_params(labelsize=12)
		ax[i].set_xticks(col)

	if len(beta_err) ==2:
		beta_err_neg = beta_err[0]
		beta_err_pos = beta_err[1]
	else:
		beta_err_pos = beta_err
		beta_err_neg = beta_err 

	if len(c500_err) ==2:
		c500_err_neg = c500_err[0]
		c500_err_pos = c500_err[1]
	else:
		c500_err_pos = c500_err
		c500_err_neg = c500_err 

	if len(p0_err) ==2:
		p0_err_neg = p0_err[0]
		p0_err_pos = p0_err[1]
	else:
		p0_err_pos = p0_err
		p0_err_neg = p0_err 


	ax[0].errorbar(col,p0_sims, yerr=[p0_err_neg, p0_err_pos],ls='none', c=cbar_col, alpha = cbar_alpha,)
	ax[0].scatter(col,p0_sims, c=cbar_col, marker = ma, s=size, alpha = cbar_alpha)
	ax[0].axhline(A10[0], c='red')
	ax[0].set_ylabel(r'$P_0$', fontsize='xx-large', fontweight='heavy')
	ax[0].set_xlabel(r'Iteration Steps', fontsize='xx-large', fontweight='heavy')
	# ax[0].axhline(7.37, c='blue', alpha=cbar_alpha)
	# ax[0].axhline(7.37+0.047, c='blue', ls ='dashed', alpha=cbar_alpha)
	# ax[0].axhline(7.37-0.047, c='blue', ls ='dashed', alpha=cbar_alpha)

	ax[1].errorbar(col,c500_sims, yerr=[c500_err_neg, c500_err_pos],ls='none', c=cbar_col, alpha = cbar_alpha)
	ax[1].scatter(col,c500_sims, c=cbar_col, marker = ma, s=size, alpha = cbar_alpha)
	ax[1].axhline(A10[1], c='red')
	ax[1].set_ylabel(r'$c_{500}$', fontsize='xx-large', fontweight='heavy')
	ax[1].set_xlabel(r'Iteration Steps', fontsize='xx-large', fontweight='heavy')
	# ax[1].axhline(0.868, c='blue', alpha=cbar_alpha)
	# ax[1].axhline(0.868+0.012, c='blue', ls ='dashed', alpha=cbar_alpha)
	# ax[1].axhline(0.868-0.012, c='blue', ls ='dashed', alpha=cbar_alpha)

	ax[2].errorbar(col,beta_sims, yerr=[beta_err_neg, beta_err_pos],ls='none', c=cbar_col, alpha = cbar_alpha)
	ax[2].scatter(col,beta_sims, c=cbar_col, marker = ma, s=size, alpha = cbar_alpha)
	ax[2].axhline(A10[3], c='red', label='Input Value')
	ax[2].set_ylabel(r'$\beta$', fontsize='xx-large', fontweight='heavy')
	ax[2].set_xlabel(r'Iteration Steps', fontsize='xx-large', fontweight='heavy')
	# ax[2].axhline(7.014, c='blue', label='APEX Fit', alpha=cbar_alpha)
	# ax[2].axhline(7.014+0.077, c='blue', ls ='dashed', alpha=cbar_alpha)
	# ax[2].axhline(7.014-0.075, c='blue', ls ='dashed', alpha=cbar_alpha)

	fig.legend(loc='best')
	fig.tight_layout(h_pad=3)
	fig.subplots_adjust(top=0.85)
	fig.suptitle(r'A2744- Sim S/N $\approx$ 900 - Intermediate Results Iterative Code with Posterior Sampling', fontsize=16)

def bias_test(files, A10, ndim, peak = False, m500fit=False, betac500=False,  **kwargs):
	import matplotlib.pyplot as plt
	import numpy as np
	import pylab
	import scipy.stats
	if peak == False:
		from GNFW_functions import mean_unc_cal as unc_cal
		color = 'sandybrown'
	else:
		import GNFW_functions as fun
		reload(fun)
		unc_cal = fun.peak_unc_cal#_eqp
		#from GNFW_functions import peak_unc_cal_eqp as unc_cal
		color = 'seagreen'

	
	pylab.ion()

	
	alphav=0.8

	results =[]
	Pei = []
	c500 = []
	b = []

	Y_true = 29.423#12.18#29.423

	if ndim ==1:
		fig, ax = plt.subplots(1,1, figsize = (20,5))
		ax =[ax]
	elif ndim ==3:
		fig, ax = plt.subplots(1,3, figsize = (20,5))
	else:
		fig, ax = plt.subplots(1,2, figsize = (20,5))
	
	for i, index in enumerate(files):
		try:
			chain = np.load(index)['chain']
		except:
			chain = np.loadtxt(index)[100000:,1:-1]
		#chain = np.load(index)['chain']
		#chain = chain.reshape((-1,3))
		if m500fit ==True:
			kwargs['m500fit']=True
		else:
			kwargs = dict()

		print (i, index[99:109])
		results.append(unc_cal(chain, A10, ndim, Y_true,**kwargs)[1])
		ax[0].scatter(results[i][0,0],results[i][1,0], c=color, alpha = alphav)
		ax[0].errorbar(results[i][0,0],results[i][1,0], xerr=[[results[i][0,2]], [results[i][0,1]]], yerr= [[results[i][1,2]], [results[i][1,1]]], ls='none', color=color, alpha = alphav)
		if ndim > 1:

			ax[1].scatter(results[i][1,0],results[i][2,0], c=color, alpha = alphav)
			ax[1].errorbar(results[i][1,0],results[i][2,0], xerr=[[results[i][1,2]], [results[i][1,1]]], yerr= [[results[i][2,2]], [results[i][2,1]]], ls='none', color=color, alpha = alphav)
		if ndim ==3:
			ax[2].scatter(results[i][0,0],results[i][3,0], c=color, alpha = alphav)
			ax[2].errorbar(results[i][0,0],results[i][3,0], xerr=[[results[i][0,2]], [results[i][0,1]]], yerr= [[results[i][3,2]], [results[i][3,1]]], ls='none', color=color, alpha = alphav)

	color = 'navy'
	ecolor= 'white'
	size = 80

	if m500fit ==True:
		ax[0].set_xlabel(r'$M_{500}$', fontsize=16)
	elif betac500 == True:
		ax[0].set_xlabel(r'$c_{500}$', fontsize=16)
	else:
		ax[0].set_xlabel(r'$P_0$', fontsize=16)

	if ndim ==1:
		if m500fit ==True:
			ax[0].scatter(kwargs.get('m500'),Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(kwargs.get('m500'), c='grey', ls='dotted', alpha=0.5)
		else:
			ax[0].scatter(A10[0],Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(A10[0], c='grey', ls='dotted', alpha=0.5)

		ax[0].axhline(Y_true, c='grey', ls='dotted', alpha=0.5)

		ax[0].set_ylabel(r'$Y_{sph}$', fontsize=16)
	else:
		if m500fit ==True:
			ax[0].scatter(kwargs.get('m500'),A10[1], c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(kwargs.get('m500'), c='grey', ls='dotted', alpha=0.5)
			ax[0].axhline(A10[1], c='grey', ls='dotted', alpha=0.5)
			ax[0].set_ylabel(r'$c_{500}$', fontsize=16)
		elif betac500 == True:
			ax[0].scatter(A10[1],A10[3], c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(A10[1], c='grey', ls='dotted', alpha=0.5)
			ax[0].axhline(A10[3], c='grey', ls='dotted', alpha=0.5)
			ax[0].set_ylabel(r'$\beta$', fontsize=16)
		else:
			ax[0].scatter(A10[0],A10[1], c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(A10[0], c='grey', ls='dotted', alpha=0.5)
			ax[0].axhline(A10[1], c='grey', ls='dotted', alpha=0.5)

			ax[0].set_ylabel(r'$c_{500}$', fontsize=16)

	if ndim >1:
		if betac500 ==True:
			ax[1].scatter(A10[3],Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)

			ax[1].axvline(A10[3], c='grey', ls='dotted', alpha=0.5)
			ax[1].axhline(Y_true, c='grey', ls='dotted', alpha=0.5)

			ax[1].set_xlabel(r'$\beta$', fontsize=16)
			ax[1].set_ylabel(r'$Y_{sph}$', fontsize=16)
		else:

			if betac500 ==True:
				ax[1].scatter(A10[1],Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)

				ax[1].axvline(A10[1], c='grey', ls='dotted', alpha=0.5)
				ax[1].axhline(Y_true, c='grey', ls='dotted', alpha=0.5)

				ax[1].set_xlabel(r'$c_{500}$', fontsize=16)
				ax[1].set_ylabel(r'$Y_{sph}$', fontsize=16)
			else:
				ax[1].scatter(A10[1],A10[3], c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)

				ax[1].axvline(A10[1], c='grey', ls='dotted', alpha=0.5)
				ax[1].axhline(A10[3], c='grey', ls='dotted', alpha=0.5)

				ax[1].set_xlabel(r'$c_{500}$', fontsize=16)
				ax[1].set_ylabel(r'$\beta$', fontsize=16)

	if ndim ==3:
		if m500fit ==True:
			ax[2].scatter(kwargs.get('m500'),Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[2].axvline(kwargs.get('m500'), c='grey', ls='dotted', alpha=0.5)
		else:
			ax[2].scatter(A10[0],Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[2].axvline(A10[0], c='grey', ls='dotted', alpha=0.5)
	
	
		ax[2].axhline(Y_true, c='grey', ls='dotted', alpha=0.5)

		if m500fit == True:
			ax[2].set_xlabel(r'$M_{500}$', fontsize=16)
		else:
			ax[2].set_xlabel(r'$P_0$', fontsize=16)
		
		ax[2].set_ylabel(r'$Y_{sph}$', fontsize=16)

	#fig.suptitle(kwargs.get('title', ''), fontsize=18)
	#plt.legend(bbox_to_anchor=(0.5, 0.012), loc='lower right', borderaxespad=0., bbox_transform=plt.gcf().transFigure, ncol=3)
	for i in range(len(ax)):
		#ax[i].set_aspect(aspect='equal')#, adjustable='datalim')
		ax[i].set_adjustable('box')
	fig.tight_layout()
	fig.subplots_adjust(top=0.85,wspace = 0.3, bottom=0.2)
	
	
	
	
	
	results = np.asarray(results)
	rr = results.reshape(-1,3)
	step = len(rr)/len(files)
	fig, ax = plt.subplots(1, step, figsize =(20,5), sharex= True )
	init = 0
	nbins = len(files)**(1/3.)
	nbins = 2*int(round(nbins%int(nbins)) + nbins)

	labels = [r'$P_0$',r'$c_{500}$', r'$\beta$', r'$Y_{sph}$' ]
	params = [A10[0], A10[1], A10[3], Y_true]

	
	
	
	#################

	if m500fit:
		labels[0] = r'$M_{500}$'

	for n in range(step):
		sigmal = []
		peakl = []
		if n == step -1:
			n = -1 
		for i in range (len(results)):
			sigma = np.average(results[i][n,1:])
			peak = results[i][n,0]
			sigmal.append(sigma)
			if betac500 ==True:
				d=n+1
				if n==-1:
					d=n
				peakl.append(peak-params[d])
			else:
				d=n
				# if n ==step:
				# 	peakl.append(peak-params[-1])
				# else:
				peakl.append(peak-params[d])
		ratio = np.array(peakl)/np.array(sigmal)
		num1s = (len(np.where(abs(ratio)<=1)[0])*100)/len(ratio)
		print (labels[d] + ':'+str( num1s)+ '% within 1 sigma')
		counts, bins, _=ax[n].hist(ratio, density=True, color='khaki', alpha =0.8, bins =nbins, label = str(num1s)+r'$\%$'+' in 1'+r'$\sigma$' )
		ax[n].axvline(0.0, ls='dotted', color = 'hotpink')
		ax[n].set_xlabel('(peak - input)/'+r'$\sigma \ [ $'+labels[d]+' ]', fontsize = 16)
		ax[n].legend(loc='best')
		#ax[n].set_xlim(-1,1)
		#ax[n].set_aspect(aspect=0.7)#, adjustable='datalim')
		#ax[n].set_adjustable('box')

		#########
		
		x = np.linspace(bins[0]-2, bins[-1]+2, 1000)
		ax[n].plot(x,scipy.stats.norm.pdf(x, 0, 1), c='hotpink', alpha=0.5)
		

	##########
	lb = str('Total Number \n of Points = ' + str(len(files)))
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)

	# place a text box in upper left in axes coords
	fig.text(0.910, 0.5, lb, transform=ax[n].transAxes, fontsize=14,
        verticalalignment='center', bbox=props)


	#fig.tight_layout()
	fig.subplots_adjust(top=0.85, bottom=0.2) #, wspace = 0.3
	print('Total Number of data points:%0.0f' %len(ratio))

	results = rr
	return results

def bias_test_contours(files, A10, ndim, peak = False):

	import matplotlib.pyplot as plt
	import numpy as np
	import pylab
	import corner
	if peak == False:
		from GNFW_functions import mean_unc_cal as unc_cal
		color = 'sandybrown'
	else:
		import GNFW_functions as fun
		reload(fun)
		unc_cal = fun.peak_unc_cal#_eqp
		#from GNFW_functions import peak_unc_cal_eqp as unc_cal
		color = 'seagreen'
	
	pylab.ion()

	
	alphav=0.5
	alphavc =0.5
	ct= 0,0,0,0
	cont_colors = ['black', 'black']#["#6495ed","#0000cd"]
	contf_colors = [ct, "#6495ed","#0000cd"]
	cont_levels = [0.68]#, 0.95]
	#cont_levels = [0.98]
	zorders = 2

	results =[]
	Pei = []
	c500 = []
	b = []

	Y_true = 29.423#12.18#29.423

	if ndim ==1:
		fig, ax = plt.subplots(1,1)
		ax =[ax]
	elif ndim ==3:
		fig, ax = plt.subplots(3,1)
	else:
		fig, ax = plt.subplots(2,1)

	for axes in ax:
		#axes.set_aspect('equal')
		axes.xaxis.label.set_size(16)
		axes.yaxis.label.set_size(16)

	for i, index in enumerate(files):
		
		try:
			chain = np.load(index)['chain']
		except:
			chain = np.loadtxt(index)[30000:,1:-1]
		#chain = chain.reshape((-1,3))
		print (i, index[0:10])
		results.append(unc_cal(chain, A10, ndim, Y_true)[1])

		corner.hist2d(chain[:,0],chain[:,1],plot_datapoints=False, fill_contours=True, plot_density=False,levels = cont_levels,contour_kwargs={  "colors":cont_colors, 'alpha':alphavc},contourf_kwargs={ "colors":contf_colors}, smooth=True, ax=ax[0])
		#ax[0].scatter(results[i][0,0],results[i][1,0], c=color, alpha = alphav, zorder=zorders)
		ax[0].set_xlim(chain[:,0].min()-1,chain[:,0].max()+1)
		ax[0].set_ylim(chain[:,1].min()-1,chain[:,1].max()+1)
		

			
		if ndim > 1:
			corner.hist2d(chain[:,1],chain[:,2],plot_datapoints=False, fill_contours=True, plot_density=False,levels = cont_levels,contour_kwargs={  "colors":cont_colors, 'alpha':alphavc},contourf_kwargs={ "colors":contf_colors},smooth=True, ax=ax[1])
			#ax[1].scatter(results[i][1,0],results[i][2,0], c=color, alpha = alphav, zorder=zorders)
			ax[1].set_xlim(chain[:,1].min()-1,chain[:,1].max()+1)
			ax[1].set_ylim(chain[:,2].min()-1,chain[:,2].max()+1)
			
		if ndim ==3:
			corner.hist2d(chain[:,0],chain[:,3],plot_datapoints=False, fill_contours=True, plot_density=False,levels = cont_levels,contour_kwargs={  "colors":cont_colors, 'alpha':alphavc},contourf_kwargs={ "colors":contf_colors},smooth=True, ax=ax[2])
			#ax[2].scatter(results[i][0,0],results[i][3,0], c=color, alpha = alphav, zorder=zorders)
			ax[2].set_xlim(chain[:,0].min()-1,chain[:,0].max()+1)
			#ax[2].set_ylim(chain[:,3].min()-1,chain[:,3].max()+1)
			ax[2].set_ylim(-20,200)

	color = 'hotpink'
	ecolor= 'white'
	size = 80

	ax[0].set_xlabel(r'$P_0$')
	if ndim ==1:
		ax[0].scatter(A10[0],Y_true, c=color, marker='v', zorder=10, s=size, edgecolor=ecolor)

		ax[0].set_ylabel(r'$Y_{sph}$')
	else:
		ax[0].scatter(A10[0],A10[1], c=color, marker='v', zorder=10, s=size, edgecolor=ecolor)

		ax[0].set_ylabel(r'$c_{500}$')

	if ndim >1:
		ax[1].scatter(A10[1],A10[3], c=color, marker='v', zorder=10, s=size, edgecolor=ecolor)

		ax[1].set_xlabel(r'$c_{500}$')
		ax[1].set_ylabel(r'$\beta$')

	if ndim ==3:
		ax[2].scatter(A10[0],Y_true, c=color, marker='v', zorder=10, s=size, edgecolor=ecolor)

		ax[2].set_xlabel(r'$P_0$')
		ax[2].set_ylabel(r'$Y_{sph}$')

	#fig.suptitle(kwargs.get('title', ''), fontsize=18)
	#plt.legend(bbox_to_anchor=(0.5, 0.012), loc='lower right', borderaxespad=0., bbox_transform=plt.gcf().transFigure, ncol=3)
	fig.tight_layout()
	fig.subplots_adjust(top=0.85)



# def correct_errors:
# 	#to calculate errorrs from side or peak??? based on equal probability
# 	original = np.unique(chainex[:,1])
# 	like_0 = -np.inf
# 	for i, val in enumerate(original):
#     index = np.where(val==chainex[:,1])[0]
#     for ii in index:
#         like_old = lnlikeex[ii]
#         if like_old > like_0:
#             lnmax[i] = like_old
#             like_0=like_old
#    .....:       like_0 =-np.inf
	
	

	


def chainprofile_68 (file, ndim, A10, **kwargs):
	#if  want to consider m500fit then include in on kwargs
	import numpy as np
	import pylab
	import GNFW_functions as fun
	import apex_planck_sim as apsim
	import matplotlib.pyplot as plt
	import radial_data as rad

	reload(fun)
	reload(rad)

	if type(file) == str:
		chain = np.load(file)['chain']
	else:
		chain = file

	
	rmax = kwargs['rmax'] #pixels
	width = kwargs['width'] #pixels
	num = 100#0#chain.shape[0]
	mean_append = np.zeros(shape=(num,int(round(rmax/width))))

	fig, ax = plt.subplots(1,1, figsize = (7.5,5.5))

	#if m500fit ==True:
	# 	kwargs['m500fit']=True
	# 	#del kwargs['m500']

	for i in range (num):
		ii = np.random.random_integers(chain.shape[0])
	 	zz_con = apsim.sim2d(theta=chain[ii,0:-1], ndim=ndim, A10=A10, apex=True, **kwargs)[1]
	 	rmax_0 =int(zz_con.shape[0]/2)
	 	zz_con = zz_con[rmax_0-rmax:rmax_0+rmax+1,rmax_0-rmax:rmax_0+rmax+1]
	 	#plt.imshow(zz_con)
	 	r_rad, meannan_rad = rad.radialdata_applied(zz_con,annulus_width=width, rmax=rmax)
	 	mean_append[i,:] = meannan_rad[0,:]
	 	#print (meannan_rad[0,:])
	 	#ax.plot(r_rad[0,:], meannan_rad[0,:], ls='dotted', c='blue', alpha = 0.3,marker=None)
	 	pylab.pause(0.1)

	range68 = np.zeros(shape=(2,mean_append.shape[1]))

	for j in range (mean_append.shape[1]):
		low, mid, up = np.percentile(mean_append[:,j], [16, 50, 84])
		range68[0,j]= low
		range68[1,j]=up
	
	ax.fill_between(r_rad[0,:], range68[0,:],range68[1,:],  color='grey', edgecolor='black', alpha = 0.3,label = '68 from posterior distribution')

	ax.legend(loc='best', numpoints=1)

	x = plt.xticks()
	xticks = x[0]*kwargs.get('pixindeg')*60
	ii=0
	for d in xticks:
		xticks[ii] = round(d)
		ii = ii+1

	plt.xticks(x[0][1:],xticks[1:])

	ax.set_xlabel('Radial Coordinate [arcmin]', fontsize = 16)
	ax.set_ylabel(r'$ \Delta T\ $' + '[mK CMB]', fontsize = 16)
	
def plot_sims_avg_res(path, num, noise_factor=1):

	import fits_files as ff
	import GNFW_functions as fun
	import covariance as cov 
	import matplotlib.pyplot as plt 
	import cluster_info_dict as cid
	import apex_planck_sim as apsim
	import radial_data as rad
	import numpy as np

	reload(rad)
	reload(cov)
	reload(ff)
	reload(fun)
	reload(cid)
	reload(apsim)

	lc = cid.dic_lc['total']
	A10=[8.403 ,1.177, 1.0510, 5.4905,0.3081]
	ndim = 3
	theta = A10[0], A10[1], A10[3]
	cluster_param = lc[0] # 0 for A1835
	name =cluster_param[0]
	z = cluster_param[1]
	M500 = cluster_param[2]
	r500 = None
	ra = cluster_param[5]
	dec = cluster_param[6]
	cxy = apsim.radectodeg(ra, dec) #center coordinates in degrees

	header, mask = ff.open_fits(path, data_a=True)[1:]
	freq_obs=150*10**9 #Hz 152?
	width = 6.0#pixel #0.2#6#0.2#*pixel_to_meters
	pixindeg =  header['CDELT2'] #0.002777778
	pixel_to_meters, pixeltoarcsec = fun.pix_con (z, pixindeg)

	path_apex_TF= '/vol/aibn160/data1/amikler/Documents/APEX_SZ/2017REDUX.0.20171108.ALL/coadded.maps/'+str(name)
	trans_fun = ff.open_fits(path_apex_TF, TF_a=True)[0]
	sim = ff.open_fits(path, sim_a= True)[0]

	rmax_0a = int(sim.shape[0]/2)#map_size[l]#126#30#12#30#126#int(rmax)#90

	kwargs = {'z':z,'pixindeg':pixindeg,'freq_obs':freq_obs, 'rmax':rmax_0a, 'm500':M500, 'r500':r500, 'trans_fun':trans_fun, 'path':path, 'noise_factor': noise_factor, 'name':name}

	zz_con= apsim.sim2d(theta=theta, ndim=ndim,  A10=A10, apex = True, **kwargs)[1]

	rmax = 1*apsim.rmax_cal(highsn_model=zz_con, data=sim, width=width,  working_mask=mask, **kwargs)

	mean_rad_all = []

	for i in range (num):
		siml = ff.open_fits(path, other= '/sim.noise/'+str(i)+'.fits')[0]
		xcent, ycent = siml.shape[0]/2, siml.shape[1]/2
		r_rad_siml, meannan_rad_siml = rad.radialdata_applied(siml, annulus_width=width, rmax=rmax)
		mean_rad_all.append(meannan_rad_siml[0,:])

	

	
	
	
	#Radial profile of model convolved
	r_rad_mc, meannan_rad_mc = rad.radialdata_applied(zz_con, annulus_width=width, rmax=rmax)

	cov, cov_i, error, corr = cov.cov(name=name,rmax=rmax,xcent=xcent, ycent=ycent, path=path, width=width, apex=True )#[0:3]

	low, mid, up = np.percentile(mean_rad_all, [16, 50, 84], axis=0)

	residual = mid - meannan_rad_mc


	fig,ax = plt.subplots(1,1, figsize = (15,8))

	ax.plot(r_rad_siml[0,:], mid, marker = 'D', color='black', label='Data', alpha = 0.7)
	ax.plot(r_rad_siml[0,:], meannan_rad_mc[0,:], marker= 'x', color='red', label = 'Convolved Model', alpha = 0.7)
	ax.errorbar(r_rad_siml[0,:], residual[0,:], yerr=error, marker='s', color='purple', markerfacecolor='none', label = 'Residual (Data - Model)')
	ax.axhline(0.0, color='gray', alpha= 0.5, ls='dashed')

	

	ax.fill_between(r_rad_siml[0,:], low,up,  color='grey', edgecolor='black', alpha = 0.3,label = '68% of simulations')
	ax.legend(loc='lower right', numpoints=1)

	x = plt.xticks()
	xticks = x[0]*pixindeg*60
	ii=0
	for d in xticks:
		xticks[ii] = round(d)
		ii = ii+1

	plt.xticks(x[0][1:-1],xticks[1:-1])

	ax.set_xlabel('Radial Coordinate [arcmin]', fontsize = 16)
	ax.set_ylabel(r'$ \Delta T\ $' + '[mK CMB]', fontsize = 16)

	return (up-low) 


def bias_test_grid(files, A10, ndim, m500fit=False, betac500=False,  **kwargs):
	import matplotlib.pyplot as plt
	import numpy as np
	import pylab
	import GNFW_functions as fun
	reload(fun)
	unc_cal = fun.grid_unc_cal#_eqp
	#from GNFW_functions import peak_unc_cal_eqp as unc_cal
	color = 'hotpink'

	
	pylab.ion()

	
	alphav=0.8

	results =[]
	Pei = []
	c500 = []
	b = []

	Y_true = 29.423#12.18#29.423

	if ndim ==1:
		fig, ax = plt.subplots(1,1)
		ax =[ax]
	elif ndim ==3:
		fig, ax = plt.subplots(3,1)
	else:
		fig, ax = plt.subplots(2,1)
	
	for i, index in enumerate(files):
		chain = np.load(index)['chain']
		lnlike = np.load(index)['lnlikelihood']
		if m500fit ==True:
			kwargs['m500fit']=True
		else:
			kwargs = dict()

		print (i, index[99:109])
		results.append(unc_cal(chain, lnlike,A10, ndim, Y_true,**kwargs)[1])
		ax[0].scatter(results[i][0,0],results[i][1,0], c=color, alpha = alphav)
		ax[0].errorbar(results[i][0,0],results[i][1,0], xerr=[[results[i][0,2]], [results[i][0,1]]], yerr= [[results[i][1,2]], [results[i][1,1]]], ls='none', color=color, alpha = alphav)
		if ndim > 1:

			ax[1].scatter(results[i][1,0],results[i][2,0], c=color, alpha = alphav)
			ax[1].errorbar(results[i][1,0],results[i][2,0], xerr=[[results[i][1,2]], [results[i][1,1]]], yerr= [[results[i][2,2]], [results[i][2,1]]], ls='none', color=color, alpha = alphav)
		if ndim ==3:
			ax[2].scatter(results[i][0,0],results[i][3,0], c=color, alpha = alphav)
			ax[2].errorbar(results[i][0,0],results[i][3,0], xerr=[[results[i][0,2]], [results[i][0,1]]],  ls='none', color=color, alpha = alphav) #yerr= [[results[i][3,2]], [results[i][3,1]]], -- this will include the errors in Y for 3 dimensions -- BUT the errors are wrongly calculated

	color = 'navy'
	ecolor= 'white'
	size = 80

	if m500fit ==True:
		ax[0].set_xlabel(r'$M_{500}$', fontsize=16)
	elif betac500 == True:
		ax[0].set_xlabel(r'$c_{500}$', fontsize=16)
	else:
		ax[0].set_xlabel(r'$P_0$', fontsize=16)

	if ndim ==1:
		if m500fit ==True:
			ax[0].scatter(kwargs.get('m500'),Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(kwargs.get('m500'), c='grey', ls='dotted', alpha=0.5)
		else:
			ax[0].scatter(A10[0],Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(A10[0], c='grey', ls='dotted', alpha=0.5)

		ax[0].axhline(Y_true, c='grey', ls='dotted', alpha=0.5)

		ax[0].set_ylabel(r'$Y_{sph}$', fontsize=16)
	else:
		if m500fit ==True:
			ax[0].scatter(kwargs.get('m500'),A10[1], c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(kwargs.get('m500'), c='grey', ls='dotted', alpha=0.5)
			ax[0].axhline(A10[1], c='grey', ls='dotted', alpha=0.5)
			ax[0].set_ylabel(r'$c_{500}$', fontsize=16)
		elif betac500 == True:
			ax[0].scatter(A10[1],A10[3], c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(A10[1], c='grey', ls='dotted', alpha=0.5)
			ax[0].axhline(A10[3], c='grey', ls='dotted', alpha=0.5)
			ax[0].set_ylabel(r'$\beta$', fontsize=16)
		else:
			ax[0].scatter(A10[0],A10[1], c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[0].axvline(A10[0], c='grey', ls='dotted', alpha=0.5)
			ax[0].axhline(A10[1], c='grey', ls='dotted', alpha=0.5)

			ax[0].set_ylabel(r'$c_{500}$', fontsize=16)

	if ndim >1:
		if betac500 ==True:
			ax[1].scatter(A10[3],Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)

			ax[1].axvline(A10[3], c='grey', ls='dotted', alpha=0.5)
			ax[1].axhline(Y_true, c='grey', ls='dotted', alpha=0.5)

			ax[1].set_xlabel(r'$\beta$', fontsize=16)
			ax[1].set_ylabel(r'$Y_{sph}$', fontsize=16)
		else:

			ax[1].scatter(A10[1],A10[3], c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)

			ax[1].axvline(A10[1], c='grey', ls='dotted', alpha=0.5)
			ax[1].axhline(A10[3], c='grey', ls='dotted', alpha=0.5)

			ax[1].set_xlabel(r'$c_{500}$', fontsize=16)
			ax[1].set_ylabel(r'$\beta$', fontsize=16)

	if ndim ==3:
		if m500fit ==True:
			ax[2].scatter(kwargs.get('m500'),Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[2].axvline(kwargs.get('m500'), c='grey', ls='dotted', alpha=0.5)
		else:
			ax[2].scatter(A10[0],Y_true, c=color, marker='v', zorder=3, s=size, edgecolor=ecolor)
			ax[2].axvline(A10[0], c='grey', ls='dotted', alpha=0.5)
	
	
		ax[2].axhline(Y_true, c='grey', ls='dotted', alpha=0.5)

		if m500fit == True:
			ax[2].set_xlabel(r'$M_{500}$', fontsize=16)
		else:
			ax[2].set_xlabel(r'$P_0$', fontsize=16)
		ax[2].set_ylabel(r'$Y_{sph}$', fontsize=16)

	#fig.suptitle(kwargs.get('title', ''), fontsize=18)
	#plt.legend(bbox_to_anchor=(0.5, 0.012), loc='lower right', borderaxespad=0., bbox_transform=plt.gcf().transFigure, ncol=3)
	# for i in range(len(ax)):
	# 	ax[i].set_aspect(aspect=0.7)#, adjustable='datalim')
	# 	#ax[i].set_adjustable('box')
	fig.tight_layout()
	fig.subplots_adjust(top=0.85)



	# if ndim ==1:
	# 	fig, ax = plt.subplots(2,1)
	# 	ax =[ax]
	# elif ndim ==3:
	# 	fig, ax = plt.subplots(4,1)
	# else:
	# 	fig, ax = plt.subplots(3,1)
	results = np.asarray(results)
	rr = results.reshape(-1,3)
	step = len(rr)/len(files)
	fig, ax = plt.subplots(step,1)
	init = 0

	labels = [r'$P_0$',r'$c_{500}$', r'$\beta$', r'$Y_{sph}$' ]

	if m500fit:
		labels[0] = r'$M_{500}$'

	for i in range(step):
		ar = rr[i::step, :]
		counts, bins, _ = ax[i].hist(ar[:,0], normed = False, bins=20)
		if betac500:
			ax[i].set_xlabel(labels[i+1], fontsize = 16)
		else:
			ax[i].set_xlabel(labels[i], fontsize = 16)
		print (counts.max(), bins[np.where(counts.max()==counts)])

	ax[i].set_xlabel(labels[-1], fontsize = 16)
	fig.tight_layout()
	fig.subplots_adjust(top=0.85)

	

	return results


def plot_grid_tot(files, A10, ndim, m500fit=False, betac500=False,  **kwargs):
	import matplotlib.pyplot as plt
	import numpy as np
	import pylab
	# import GNFW_functions as fun
	# reload(fun)
	# unc_cal = fun.grid_unc_cal#_eqp
	# #from GNFW_functions import peak_unc_cal_eqp as unc_cal
	# color = 'hotpink'

	
	pylab.ion()

	
	
	

	
	Y_true = 29.423#12.18#29.423

	lnlike_tot = 0 

	if ndim ==1:
		import sys
		print ('Only one dimension - lookat results with a histogram')
		sys.exit(0)
	else:
		
		fig, ax = plt.subplots(1, ndim, figsize=(15,4))
		try:
		 len(ax)
		except:
			ax = [ax]
	
	for i, f in enumerate(files):
		lnlike_tot += np.load(f)['lnlikelihood']
	
	grid = np.load(f)['chain']

	##################

	colormax = 'seagreen'
	ecolormax = 'white'
	sizemax = 80
	zm = 5 #zorder for max L value
	#######
	alphav=0.8
	gs = 30 #gridsize value for hexbin - represents number of bins in the x axis.
	ecolor = 'black' #color for the line around the bins
	lw = 0.05 #line width for the hexbins
	cmapc='magma'

	#####
	vmaxl = lnlike_tot.max()
	minl = lnlike_tot[np.isfinite(lnlike_tot)].min()
	power = int(np.log10(abs(minl))) -2
	vminl  = vmaxl +(minl/(10**power))
	#####
	print (grid[:,-1].max())
	print('Maximun Likelihood Parameters:',grid[np.argmax(lnlike_tot),:] )


	gs = np.unique(grid[:,0]).shape[0]/2#, np.unique(grid[:,1]).shape[0]/2
	ax[0].scatter(grid[np.argmax(lnlike_tot),0], grid[np.argmax(lnlike_tot),1], marker = '^', c=colormax, edgecolor=ecolormax, s=sizemax, zorder = zm, alpha=1)
	cbi = ax[0].hexbin(grid[:,0], grid[:,1], C=(lnlike_tot), reduce_C_function=np.max, gridsize=gs,  vmin =vminl,vmax = vmaxl, extent=(grid[:,0].min() -4, grid[:,0].max() +4, grid[:,1].min() -1, grid[:,1].max() ), edgecolor=ecolor, linewidths=lw, cmap=cmapc, alpha=alphav, mincnt=1)

	if ndim > 2:

		gs = np.unique(grid[:,1]).shape[0]/2#, np.unique(grid[:,2]).shape[0]
		ax[1].scatter(grid[np.argmax(lnlike_tot),1], grid[np.argmax(lnlike_tot),2], marker = '^', c=colormax, edgecolor=ecolormax,s=sizemax, zorder = zm)

		cbi = ax[1].hexbin(grid[:,1], grid[:,2], C=(lnlike_tot), reduce_C_function=np.max, gridsize=gs,  vmin =vminl,vmax = vmaxl, extent=(grid[:,1].min() -1, grid[:,1].max()-1 , grid[:,2].min() -1, grid[:,2].max() +1 ), edgecolor=ecolor, linewidths=lw, cmap=cmapc, alpha=alphav, mincnt=1)
		

	if ndim ==3:

		gs = np.unique(grid[:,0]).shape[0]/2#, np.unique(grid[:,3]).shape[0]
		ax[2].scatter(grid[np.argmax(lnlike_tot),0], grid[np.argmax(lnlike_tot),2], marker = '^', c=colormax, edgecolor=ecolormax,s=sizemax, zorder = zm)
		cbi = ax[2].hexbin(grid[:,0], grid[:,2], C=(lnlike_tot), reduce_C_function=np.max, gridsize=gs,  vmin =vminl,vmax = vmaxl, extent=(grid[:,0].min() -5, grid[:,0].max() +5, grid[:,2].min() -1, grid[:,2].max() +1 ), edgecolor=ecolor, linewidths=lw, cmap=cmapc, alpha=alphav, mincnt=1)

	#fig.subplots_adjust( left= 0.17, right= 0.83, top = 0.85, hspace = 0.4)

	fig.subplots_adjust( left= 0.05, right= 1.05, top = 0.80, bottom=0.2, wspace = 0.33, hspace = 0.4)
	cbar = fig.colorbar(cbi,ax=ax.ravel().tolist(), label=r'$ln(L)$')
	cbar.ax.get_yaxis().labelpad = 5
	cbar.ax.set_ylabel(r'$ln(L)$',fontsize=16)

	

	############################################################

	color = 'navy'
	ecolor= 'white'
	markert = 'o'
	size = 80

	
	if ndim ==1:
		if m500fit ==True:
			ax[0].scatter(kwargs.get('m500'),Y_true, c=color, marker=markert, zorder=3, s=size, edgecolor=ecolor)
			ax[0].set_xlabel(r'$M_{500}$', fontsize=16)

		else:
			ax[0].scatter(A10[0],Y_true, c=color, marker=markert, zorder=3, s=size, edgecolor=ecolor)
			ax[0].set_xlabel(r'$P_0$', fontsize=16)

		
	else:
		if m500fit ==True:
			ax[0].scatter(kwargs.get('m500'),A10[1], c=color, marker=markert, zorder=3, s=size, edgecolor=ecolor)
			ax[0].set_xlabel(r'$M_{500}$', fontsize=16)
			ax[0].set_ylabel(r'$c_{500}$', fontsize=16)
		elif betac500 == True:
			ax[0].scatter(A10[1],A10[3], c=color, marker=markert, zorder=3, s=size, edgecolor=ecolor)
			ax[0].set_xlabel(r'$c_{500}$', fontsize=16)
			ax[0].set_ylabel(r'$\beta$', fontsize=16)
		else:
			ax[0].scatter(A10[0],A10[1], c=color, marker=markert, zorder=3, s=size, edgecolor=ecolor)
			ax[0].set_xlabel(r'$P_0$', fontsize=16)
			ax[0].set_ylabel(r'$c_{500}$', fontsize=16)

	if ndim >1:
		if betac500 ==True:
			ax[1].scatter(A10[1],A10[3], c=color, marker=markert, zorder=3, s=size, edgecolor=ecolor)
			ax[1].set_xlabel(r'$\beta$', fontsize=16)
			#ax[1].set_ylabel(r'$Y_{sph}$', fontsize=16)
		else:

			ax[1].scatter(A10[1],A10[3], c=color, marker=markert, zorder=3, s=size, edgecolor=ecolor)
			ax[1].set_xlabel(r'$c_{500}$', fontsize=16)
			ax[1].set_ylabel(r'$\beta$', fontsize=16)

	if ndim ==3:
		if m500fit ==True:
			ax[2].scatter(kwargs.get('m500'),Y_true, c=color, marker=markert, zorder=3, s=size, edgecolor=ecolor)
		else:
			ax[2].scatter(A10[0],A10[3], c=color, marker=markert, zorder=3, s=size, edgecolor=ecolor)
	
		if m500fit == True:
			ax[2].set_xlabel(r'$M_{500}$', fontsize=16)
		else:
			ax[2].set_xlabel(r'$P_0$', fontsize=16)
		#ax[2].set_ylabel(r'$Y_{sph}$', fontsize=16)

	#ax[-1].set_ylabel(r'$Y_{sph}$', fontsize=16)
	ax[-1].set_ylabel(r'$\beta$', fontsize=16)
	#ax[-1].set_ylim(0,100)
	#ax[-1].set_ylim(grid[np.argmax(lnlike_tot),-1]-grid[np.argmax(lnlike_tot),-1]/0.5,grid[np.argmax(lnlike_tot),-1]+grid[np.argmax(lnlike_tot),-1]/0.5 )
	

	#fig.suptitle(kwargs.get('title', ''), fontsize=18)
	#plt.legend(bbox_to_anchor=(0.5, 0.012), loc='lower right', borderaxespad=0., bbox_transform=plt.gcf().transFigure, ncol=3)
	for i in range(len(ax)):
		ax[i].set_aspect(aspect='auto')#, adjustable='datalim')
		#ax[i].set_adjustable('box')
	
def color_maker(n, oplot=False, alpha=0.5):
	
	from matplotlib.colors import colorConverter
	import numpy as np

	glass = (1,1,1,0)
	list_colors = ['C2', 'C4', 'C1', 'C0', 'C5', 'C6', 'C7', 'C8', 'C9', 'C3']

	if n >9:
		nr = np.random.random(), np.random.random(), np.random.random()
		c = list(colorConverter.to_rgba((nr))) 
		cl = list(colorConverter.to_rgba((nr)))
		cl[-1] = alpha
	else:
		c = list(colorConverter.to_rgba(list_colors[n])) if oplot else list(colorConverter.to_rgba("C0")) 
		cl = list(colorConverter.to_rgba(list_colors[n])) if oplot else list(colorConverter.to_rgba("C0")) 
		cl[-1] = alpha
	return glass, c, cl


def meta_chain_maker(files, multiple=False, alphafit = False,  **kwargs):

	
	import numpy as np
	if multiple:
		ff = []
		ll = []
		for i in range(len(files)):
			f = files[i]
			try:
				ref = np.load(f[0])['chain']
			except:
				ref = f[0][:,0:-1]

			chain_tot = np.zeros(shape=(ref.shape[0]*len(f),ref.shape[1]))
			lnlike_tot = np.zeros(shape=(chain_tot.shape[0]))
			start = 0

			for file in f:
				try:
					chain = np.load(file)['chain']
					lnlikelihood = np.load(file)['lnlikelihood']
				except:
					chain = file[:,0:-1]
					lnlikelihood = np.asarray([file[:,-1]])
				chain_tot[start:start+chain.shape[0],:] = chain
				try:
					lnlike_tot[start:start+chain.shape[0]] = lnlikelihood[:,0]
				except:
					lnlike_tot[start:start+chain.shape[0]] = lnlikelihood[:]
				start = chain.shape[0]+start

			if alphafit and chain_tot.shape[1]==5:
				chain_tot = np.vstack((chain_tot[:,0], chain_tot[:,1], chain_tot[:,3], chain_tot[:,-1]))
				chain_tot = chain_tot.T
			ff.append(chain_tot)
			ll.append(lnlike_tot)
	else:
		try:
			ref = np.load(files[0])['chain']
		except:
			ref = files[0][:,0:-1]

		chain_tot = np.zeros(shape=(ref.shape[0]*len(files),ref.shape[1]))
		lnlike_tot = np.zeros(shape=(chain_tot.shape[0]))
		start = 0


		for file in files:
			try:
				chain = np.load(file)['chain']
				lnlikelihood = np.load(file)['lnlikelihood']
			except:
				chain = file[:,0:-1]
				lnlikelihood = np.asarray([file[:,-1]])
			chain_tot[start:start+chain.shape[0],:] = chain
			try:
				lnlike_tot[start:start+chain.shape[0]] = lnlikelihood[:,0]
			except:
				lnlike_tot[start:start+chain.shape[0]] = lnlikelihood[:]
			start = chain.shape[0]+start
	
		if alphafit:
			chain_tot = np.vstack((chain_tot[:,0], chain_tot[:,1], chain_tot[:,3], chain_tot[:,-1]))
			chain_tot = chain_tot.T

		ff = chain_tot
		ll = lnlike_tot

	return ff, ll

##################################################################

def comb_chains_contours(files, A10, ndim, m500fit =False, alphafit = False, oplot=False,  **kwargs):

	import corner
	import numpy as np
	import matplotlib.pyplot as plt
	import pylab as py

	py.ion()

	ref = np.load(files[0])['chain']

	chain_tot = np.zeros(shape=(ref.shape[0]*len(files),ref.shape[1]))
	lnlike_tot = np.zeros(shape=(chain_tot.shape[0]))
	start = 0

	for file in files:
		chain = np.load(file)['chain']
		lnlikelihood = np.load(file)['lnlikelihood']
		chain_tot[start:start+chain.shape[0],:] = chain
		lnlike_tot[start:start+chain.shape[0]] = lnlikelihood[:,0]
		start = chain.shape[0]+start
	
	if alphafit:
		chain_tot = np.vstack((chain_tot[:,0], chain_tot[:,1], chain_tot[:,3], chain_tot[:,-1]))
		chain_tot = chain_tot.T
		ndim = 3
	
	idxmax = np.argmax(lnlike_tot)

	if 'file2' in kwargs:
		files = kwargs['file2']
		ref = np.load(files[0])['chain']
		chain_tot2 = np.zeros(shape=(ref.shape[0]*len(files),ref.shape[1]))
		lnlike_tot2 = np.zeros(shape=(chain_tot2.shape[0]))
		start = 0
		for file in files:
			chain = np.load(file)['chain']
			lnlikelihood = np.load(file)['lnlikelihood']
			chain_tot2[start:start+chain.shape[0],:] = chain
			lnlike_tot2[start:start+chain.shape[0]] = lnlikelihood[:,0]
			start = chain.shape[0]+start
		kwargs['file2'] = chain_tot2

	if 'file3' in kwargs:
		files = kwargs['file3']
		ref = np.load(files[0])['chain']
		chain_tot3 = np.zeros(shape=(ref.shape[0]*len(files),ref.shape[1]))
		lnlike_tot3 = np.zeros(shape=(chain_tot3.shape[0]))
		start = 0
		for file in files:
			chain = np.load(file)['chain']
			lnlikelihood = np.load(file)['lnlikelihood']
			chain_tot3[start:start+chain.shape[0],:] = chain
			lnlike_tot3[start:start+chain.shape[0]] = lnlikelihood[:,0]
			start = chain.shape[0]+start
		kwargs['file3'] = chain_tot3

	#print (chain_tot2.shape, chain_tot.shape)
	fig, ax = tri(chain_tot, oplot=oplot, m500fit = m500fit, **kwargs)

	# alphav=0.5
	# alphavc =0.7
	# ct= 1,1,1,0
	# cont_colors = ['black', 'black']#["#6495ed","#0000cd"]
	# #contf_colors = [ct, "#6495ed","#0000cd"]
	# contf_colors = [ct, "steelblue","peachpuff"]
	# cont_levels = [0.68, 0.95]
	# #cont_levels = [0.98]
	# zorders = 2
	# markert = '^'
	# colormarker = 'hotpink'
	# ecolor = 'white'
	# size = 80
	# markermax = 'H'
	# colormax = 'seagreen'
	# bins_number = int(len(chain_tot)**(1/3.))

	# Y_true = kwargs.get('Y_true',29.423)#12.18#29.423

	# if ndim ==1:
	# 	fig, ax = plt.subplots(1,1, figsize = (15,4))
	# 	ax =[ax]
	# elif ndim ==3:
	# 	fig, ax = plt.subplots(1,3, figsize = (15,4) )
	# else:
	# 	fig, ax = plt.subplots(1,2, figsize = (15,4))

	# for axes in ax:
	# 	#axes.set_aspect('equal')
	# 	axes.xaxis.label.set_size(16)
		

	# corner.hist2d(chain_tot[:,0],chain_tot[:,1],plot_datapoints=False, fill_contours=True, plot_density=False,levels = cont_levels,contour_kwargs={  "colors":cont_colors, 'alpha':alphavc},contourf_kwargs={ "colors":contf_colors}, smooth=True, bins= bins_number, ax=ax[0])
	# ax[0].scatter(chain_tot[idxmax,0], chain_tot[idxmax,1], marker = markermax, c=colormax, s = size, edgecolor=ecolor  )
	# if m500fit == True:
	# 	ax[0].scatter(kwargs.get('m500'), A10[1], marker = markert, c=colormarker, s = size, edgecolor=ecolor  )
	# 	ax[0].set_xlim(0,kwargs.get('m500')+20)
	# else:
	# 	ax[0].scatter(A10[0], A10[1], marker = markert, c=colormarker, s = size, edgecolor=ecolor  )
	# 	ax[0].set_xlim(0,40)

	
	# ax[0].set_ylim(0,8)

	# if ndim >1:
	# 	corner.hist2d(chain_tot[:,1],chain_tot[:,2],plot_datapoints=False, fill_contours=True, plot_density=False,levels = cont_levels,contour_kwargs={  "colors":cont_colors, 'alpha':alphavc},contourf_kwargs={ "colors":contf_colors}, smooth=True,bins= bins_number, ax=ax[1])
	# 	ax[1].scatter(chain_tot[idxmax,1], chain_tot[idxmax,2], marker = markermax, c=colormax, s = size, edgecolor=ecolor  )
	# 	ax[1].scatter(A10[1], A10[3], marker = markert, c=colormarker, s = size, edgecolor=ecolor  )
	# 	ax[1].set_xlim(0,8)
	# 	ax[1].set_ylim(0,30)
	# if ndim ==3:
	# 	corner.hist2d(chain_tot[:,0],chain_tot[:,2],plot_datapoints=False, fill_contours=True, plot_density=False,levels = cont_levels,contour_kwargs={  "colors":cont_colors, 'alpha':alphavc},contourf_kwargs={ "colors":contf_colors}, smooth=True, bins= bins_number, ax=ax[2])
	# 	ax[2].scatter(chain_tot[idxmax,0], chain_tot[idxmax,2], marker = markermax, c=colormax, s = size, edgecolor=ecolor  )
	# 	if m500fit == True:
	# 		ax[2].scatter(kwargs.get('m500'), Y_true, marker = markert, c=colormarker, s = size, edgecolor=ecolor  )
	# 		ax[2].set_xlim(0,kwargs.get('m500')+20)

	# 	else:
	# 		ax[2].scatter(A10[0], A10[3], marker = markert, c=colormarker, s = size, edgecolor=ecolor  )
	# 		ax[2].set_xlim(0,40)
	# 	ax[2].set_ylim(0,30)

	# if m500fit:
	# 	ax[0].set_xlabel(r'$M_{500}$', fontsize=16)
	# else:
	# 	ax[0].set_xlabel(r'$P_0$', fontsize=16)
	# ax[0].set_ylabel(r'$c_{500}$', fontsize=16)

	# ax[1].set_xlabel(r'$c_{500}$', fontsize=16)
	# ax[1].set_ylabel(r'$\beta$', fontsize=16)
	# if ndim ==3:
	# 	if m500fit:
	# 		ax[2].set_xlabel(r'$M_{500}$', fontsize=16)
	# 	else:
	# 		ax[2].set_xlabel(r'$P_0$', fontsize=16)
	# 	ax[2].set_ylabel(r'$\beta$', fontsize=16)

	# for i in range(len(ax)):
	# 	#ax[i].set_aspect(aspect='auto')#, adjustable='datalim')
	# 	ax[i].set_adjustable('box')

	# fig.tight_layout()
	# fig.subplots_adjust(top=0.85, wspace = 0.25)
	return chain_tot, lnlike_tot





