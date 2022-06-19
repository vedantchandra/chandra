# Shared functions for outer halo science

from astropy.coordinates import SkyCoord
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import io
import urllib
try:
	import aplpy
except:
	print('no aplpy, no finder charts..')
from astropy.io import fits
import astropy.units as u
from gala import coordinates as gc, potential as gp, dynamics as gd
from tqdm import tqdm
import copy
from scipy import stats
from astropy import coordinates as acoord


lmc = [280.4652, -32.8884]
smc = [302.8084, -44.3277]

 ############# ############# ############# ############# #############
 # COMPUTE HALO quantitities 


def make_coords(table):

	coord = SkyCoord(ra = table['ra'], dec = table['dec'], frame = 'icrs',
			 pm_ra_cosdec = table['pmra'], pm_dec = table['pmdec'],
			 radial_velocity = table['rv'],
			 distance = table['distance'])

	coord_rc = gc.reflex_correct(coord)

	gal = coord.galactic
	gal_rc = coord_rc.galactic

	table['X'] = coord.galactocentric.x.value
	table['Y'] = coord.galactocentric.y.value
	table['Z'] = coord.galactocentric.z.value
	table['Rgal'] = np.sqrt(table['X']**2 + table['Y']**2 + table['Z']**2)
	table['lwrap'] = gal.l.value
	table['b'] = gal.b.value

	table['l'] = table['lwrap'].copy()
	table['l'][table['l'] < 0] = table['l'][table['l'] < 0] + 360

	table['sgr_l'] = coord.transform_to(gc.SagittariusLaw10).Lambda.value
	table['sgr_b'] = coord.transform_to(gc.SagittariusLaw10).Beta.value

	table['pmra_rc'] = coord_rc.pm_ra_cosdec.value
	table['pmdec_rc'] = coord_rc.pm_dec.value
	table['pm_rc'] = np.sqrt(table['pmra_rc']**2 + table['pmdec_rc']**2)

	table['rv_rc'] = coord_rc.radial_velocity.value

	table['pml'] = gal.pm_l_cosb.value
	table['pmb'] = gal.pm_b.value

	table['pml_rc'] = gal_rc.pm_l_cosb.value
	table['pmb_rc'] = gal_rc.pm_b.value

	table['vtl'] = 4.7 * table['pml'] * table['distance']
	table['vtb'] = 4.7 * table['pmb'] * table['distance']

	table['vtl_rc'] = 4.7 * table['pml_rc'] * table['distance']
	table['vtb_rc'] = 4.7 * table['pmb_rc'] * table['distance']

	table['d_lmc'] = np.sqrt(((table['l'] - lmc[0]) * np.cos(np.radians(table['b'])))**2 + (table['b'] - lmc[1])**2)
	table['d_smc'] = np.sqrt(((table['l'] - smc[0]) * np.cos(np.radians(table['b'])))**2 + (table['b'] - smc[1])**2)

	return table


def make_kinematics(tab, nmc = 10, orbit = False, pot = gp.MilkyWayPotential(),
	orbit_kw = dict(dt= 1 * u.Myr, n_steps=2500)):
	
	newtab = copy.deepcopy(tab)
	
	kin_mc = {};
	
	kin_mc['Etot'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Ek'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Epot'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Lx'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Ly'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Lz'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Vtan'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Vr_gal'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Vtheta_gal'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Vphi_gal'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['R_gal'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['ecc'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['rperi'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['rapo'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['zmax'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Jr'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Jphi'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Jz'] = np.zeros((len(tab), nmc)) * np.nan
	kin_mc['Jtot'] = np.zeros((len(tab), nmc)) * np.nan
		
	print('sampling astrometry and distance, computing phase-space...')
	
	for idx,star in enumerate(tqdm(tab)):
		
		V_ra         = (star['ra_error'])**2.0
		V_dec        = (star['dec_error'])**2.0
		V_pmra       = star['pmra_error']**2.0
		V_pmdec      = star['pmdec_error']**2.0
		V_ra_dec     = (star['ra_error'] ) * (star['dec_error']) * star['ra_dec_corr']
		V_ra_pmra    = (star['ra_error'] ) * star['pmra_error']  * star['ra_pmra_corr']
		V_dec_pmra   = (star['dec_error']) * star['pmra_error']  * star['dec_pmra_corr']
		V_ra_pmdec   = (star['ra_error'] ) * star['pmdec_error'] * star['ra_pmdec_corr']
		V_dec_pmdec  = (star['dec_error']) * star['pmdec_error'] * star['dec_pmdec_corr']
		V_pmra_pmdec = star['pmra_error'] * star['pmdec_error'] * star['pmra_pmdec_corr']

		mu = [star['ra'],star['dec'],star['pmra'],star['pmdec']]

		cov = ([
		[V_ra,      V_ra_dec,   V_ra_pmra,    V_ra_pmdec  ],
		[V_ra_dec,  V_dec,      V_dec_pmra,   V_dec_pmdec ],
		[V_ra_pmra, V_dec_pmra, V_pmra,       V_pmra_pmdec],
		[V_ra_pmdec,V_dec_pmdec,V_pmra_pmdec, V_pmdec     ],
		])

		astsamples = np.random.multivariate_normal(mu, cov, nmc)
					
		distmean = star['distance']
		diststd = star['e_distance']
		a, b = (0.0 - distmean) / diststd, (500.0 - distmean) / diststd # min max distance
		distances = stats.truncnorm.rvs(a,b,loc=distmean,scale=diststd,size=nmc)
		
		ras = astsamples[:, 0] * u.deg
		decs = astsamples[:, 1] * u.deg
		pmras = astsamples[:, 2] * u.mas / u.yr
		pmdecs = astsamples[:, 3] * u.mas / u.yr
		distances = distances * u.kpc
		rvs = (star['rv'] + np.random.normal(size = nmc) * star['e_rv']) * u.km / u.s
		
		coord_i = SkyCoord(ra = ras, 
						dec = decs, 
						 pm_ra_cosdec = pmras, 
						   pm_dec = pmdecs,
						distance = distances,
						radial_velocity = rvs,
						frame = 'icrs')

		coord_gal_i = coord_i.transform_to(acoord.Galactocentric())

		w0 = gd.PhaseSpacePosition(coord_gal_i.cartesian)

		x = np.array([w0.pos.x.value, w0.pos.y.value, w0.pos.z.value]) * w0.pos.x.unit
		v = np.array([w0.vel.d_x.value, w0.vel.d_y.value, w0.vel.d_z.value]) * w0.vel.d_x.unit
		L = np.cross(x.value, v.value, axis=0) * w0.pos.x.unit * w0.vel.d_x.unit
		
		R_gal      = np.sqrt((x[0]**2.0)+(x[1]**2.0)+(x[2]**2.0)).value

		Ltot = np.linalg.norm(L.value, axis=0) * L.unit
		Lx = L[0] / 1e3
		Ly = L[1] / 1e3
		Lz = L[2] / 1e3

		Ek   = w0.kinetic_energy().to(u.km**2*u.s**-2) / 1e5
		Epot = w0.potential_energy(pot).to(u.km**2*u.s**-2) / 1e5
		Etot = w0.energy(pot).to(u.km**2*u.s**-2) / 1e5

		Vtan = 4.74 * np.sqrt(coord_i.pm_ra_cosdec.value**2 + coord_i.pm_dec.value**2) * coord_i.distance.value

		radec = SkyCoord(ra = coord_i.ra, dec =  coord_i.dec, frame = 'icrs')

		v_gsr = gc.vhel_to_vgsr(radec, coord_i.radial_velocity, 
										  vsun = acoord.Galactocentric().galcen_v_sun.to_cartesian())

		ws = gd.PhaseSpacePosition(coord_gal_i.sphericalcoslat)

		Vtheta_gal = (ws.distance * ws.pm_lat).to(u.km/u.s, u.dimensionless_angles()).value
		Vphi_gal   = (ws.distance * ws.pm_lon_coslat).to(u.km/u.s, u.dimensionless_angles()).value

		kin_mc['Etot'][idx, :] = Etot.value
		kin_mc['Ek'][idx, :] = Ek.value
		kin_mc['Epot'][idx, :] = Epot.value
		kin_mc['Lx'][idx, :] = Lx.value
		kin_mc['Ly'][idx, :] = Ly.value
		kin_mc['Lz'][idx, :] = Lz.value

		kin_mc['Vtan'][idx, :] = Vtan
		kin_mc['Vr_gal'][idx, :] = v_gsr.value

		kin_mc['Vtheta_gal'][idx, :] = Vtheta_gal
		kin_mc['Vphi_gal'][idx, :] = Vphi_gal
		
		kin_mc['R_gal'][idx, :] = R_gal
		
		if orbit:
		
			orbit = gp.Hamiltonian(pot).integrate_orbit(w0, **orbit_kw)
			kin_mc['ecc'][idx, :] = orbit.eccentricity().value
			kin_mc['rperi'][idx, :] = orbit.pericenter().value
			kin_mc['rapo'][idx, :] = orbit.apocenter().value
			kin_mc['zmax'][idx, :] = orbit.zmax().value
            
			aaf = gd.find_actions_staeckel(pot, orbit)
			kin_mc['Jr'][idx, :] = aaf['actions'][:, 0].value * 1e-3
			kin_mc['Jphi'][idx, :] = aaf['actions'][:, 1].value * 1e-3
			kin_mc['Jz'][idx, :] = aaf['actions'][:, 2].value * 1e-3
			kin_mc['Jtot'][idx, :] = np.sqrt(kin_mc['Jr'][idx, :]**2
									+ kin_mc['Jphi'][idx, :]**2 
									+ kin_mc['Jz'][idx, :]**2)

	print('computing parameter quantiles...')

	for param,values in kin_mc.items():
		
		newtab[param] = np.nanmean(values, axis = 1)
		newtab['e_' + param] = (np.nanquantile(values, 0.84, axis = 1) - np.nanquantile(values, 0.16, axis = 1)) / 2
		
		newtab['le_' + param] = (np.nanquantile(values, 0.5, axis = 1) - np.nanquantile(values, 0.16, axis = 1))
		newtab['ue_' + param] = (np.nanquantile(values, 0.84, axis = 1) - np.nanquantile(values, 0.5, axis = 1))
	
	return newtab



 ############# ############# ############# ############# #############
 # GAL XYZ PLOTS


def plot_galstat(z, name, vmin, vmax, bins, statistic = np.median):
	fig = plt.figure(figsize = (16, 10))
	plt.subplot(231)
	binstat(subtab['X'], subtab['Y'], subtab[z], statistic = statistic,
		   bins = bins, lim = 50, vmin = vmin, vmax = vmax, xlab = 'X (kpc)', ylab = 'Y (kpc)',
		   mrk = [0,1])
	plt.title('%i < |Z/kpc| < %i' % (zmin, zmax))


	plt.subplot(232)
	binstat(subtab['X'], subtab['Z'], subtab[z], statistic = statistic,
		   bins = bins, lim = 50, vmin = vmin, vmax = vmax, xlab = 'X (kpc)', ylab = 'Z (kpc)',
		   mrk = [0,2])
	
	plt.title('%s' % (name))

	plt.subplot(233)
	binstat(subtab['Y'], subtab['Z'], subtab[z], statistic = statistic,
		   bins = bins, lim = 50, vmin = vmin, vmax = vmax, xlab = 'Y (kpc)', ylab = 'Z (kpc)',
		   mrk = [1,2])
	
	plt.title('[%.1f, %.1f]' % (vmin, vmax))
	
	plt.tight_layout()
	
	return fig


 ############# ############# ############# ############# #############
 # HEALPY MAPS
  ############# ############# ############# ############# #############

def cat2hpx(lon, lat, nside, radec=True):
	"""
	Convert a catalogue to a HEALPix map of number counts per resolution
	element.

	Parameters
	----------
	lon, lat : (ndarray, ndarray)
		Coordinates of the sources in degree. If radec=True, assume input is in the icrs
		coordinate system. Otherwise assume input is glon, glat

	nside : int
		HEALPix nside of the target map

	radec : bool
		Switch between R.A./Dec and glon/glat as input coordinate system.

	Return
	------
	hpx_map : ndarray
		HEALPix map of the catalogue number counts in Galactic coordinates

	"""

	npix = hp.nside2npix(nside)

	if radec:
		eq = SkyCoord(lon, lat, 'icrs', unit='deg')
		l, b = eq.galactic.l.value, eq.galactic.b.value
	else:
		l, b = lon, lat

	# conver to theta, phi
	theta = np.radians(90. - b)
	phi = np.radians(l)

	# convert to HEALPix indices
	indices = hp.ang2pix(nside, theta, phi)

	idx, counts = np.unique(indices, return_counts=True)

	# fill the fullsky map
	hpx_map = np.zeros(npix, dtype=int)
	hpx_map[idx] = counts

	return hpx_map

def map_stars(tab, nside = 32, log = False, title = '', hp_kw = {}, fig = None,
			 label = ''):
			 
	hpx_map = cat2hpx(tab['l'], tab['b'], nside=nside, radec=False)

	if log:
		qty = np.log10(hpx_map+1)
	else:
		qty = hpx_map
	
	starmap = hp.mollview(qty, **hp_kw, fig = fig, unit = label);

	plt.title(title)
	
	return hpx_map


 ############# ############# ############# #############
 ## PLOT SELECTION
  ############# ############# ############# ############# 

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class SelectFromCollection:
	"""Select indices from a matplotlib collection using `LassoSelector`.

	Selected indices are saved in the `ind` attribute. This tool fades out the
	points that are not part of the selection (i.e., reduces their alpha
	values). If your collection has alpha < 1, this tool will permanently
	alter the alpha values.

	Note that this tool selects collection objects based on their *origins*
	(i.e., `offsets`).

	Parameters
	----------
	ax : :class:`~matplotlib.axes.Axes`
		Axes to interact with.

	collection : :class:`matplotlib.collections.Collection` subclass
		Collection you want to select from.

	alpha_other : 0 <= float <= 1
		To highlight a selection, this tool sets all selected points to an
		alpha value of 1 and non-selected points to `alpha_other`.
	"""

	def __init__(self, ax, collection, alpha_other=0.3):
		self.canvas = ax.figure.canvas
		self.collection = collection
		self.alpha_other = alpha_other

		self.xys = collection.get_offsets()
		self.Npts = len(self.xys)

		# Ensure that we have separate colors for each object
		self.fc = collection.get_facecolors()
		if len(self.fc) == 0:
			raise ValueError('Collection must have a facecolor')
		elif len(self.fc) == 1:
			self.fc = np.tile(self.fc, (self.Npts, 1))

		self.lasso = LassoSelector(ax, onselect=self.onselect)
		self.ind = []

	def onselect(self, verts):
		path = Path(verts)
		self.ind = np.nonzero(path.contains_points(self.xys))[0]
		self.fc[:, -1] = self.alpha_other
		self.fc[self.ind, -1] = 1
		self.collection.set_facecolors(self.fc)
		self.canvas.draw_idle()

	def disconnect(self):
		self.lasso.disconnect_events()
		self.fc[:, -1] = 1
		self.collection.set_facecolors(self.fc)
		self.canvas.draw_idle()

def plot_selector(x, y, hist = False, scatter_kw = dict(s = 1, alpha = 0.5), hist_kw = dict(bins = 50)):
	fig = plt.figure(figsize = (10,7))
	ax = plt.gca()
	pts = plt.scatter(x, y, **scatter_kw)
	
	if hist:
		plt.hist2d(x, y, **hist_kw)
	
	selector = SelectFromCollection(ax, pts)

	def accept(event):
		if event.key == "enter":
			print("Selected points:")
			print(selector.xys[selector.ind])
			selector.disconnect()
			ax.set_title("")
			fig.canvas.draw()

	fig.canvas.mpl_connect("key_press_event", accept)
	ax.set_title("Press enter to accept selected points.")
	return selector

 ############# ############# #############
### PLOT DSS IMAGE #############
 ############# ############# #############

imserver = 'poss2ukstu_red'

def get_dss(ra, dec, size = 60.0): # size in arcmin
	url = "http://archive.stsci.edu/cgi-bin/dss_search?v=%s&r=%f&d=%f&e=J2000&h=%.1f&w=%.1f&f=fits&c=none" %\
					(imserver, ra, dec, size, size)
	print(url)
	fitsData = io.BytesIO()
	data = urllib.request.urlopen(url).read()
	fitsData.write(data)
	fitsData.seek(0)
	return fits.open(fitsData)

def init_plot(hdu, title, ra, dec):
	servname = {}
	servname['none']=''
	servname['poss2ukstu_red'] = "POSS2/UKSTU Red"
	servname['poss2ukstu_blue'] = "POSS2/UKSTU Blue"
	servname['poss2ukstu_ir'] = "POSS2/UKSTU IR"
	servname['poss1_blue'] = "POSS1 Blue"
	servname['poss1_red'] = "POSS1 Red"
	
	imserver = 'poss2ukstu_red'

	#out = sys.stdout
	#sys.stdout = open("/dev/null", 'w')
	plot = aplpy.FITSFigure(hdu)
	plot.show_grayscale()
	plot.set_theme('publication')
	return plot

def plot_dss(sel_ra, sel_dec, size = 30, radius = 6, plot_circles = False, circle_ra = None, circle_dec = None):
	hdu = get_dss(sel_ra, sel_dec, size = size)
	plot = init_plot(hdu, '', sel_ra, sel_dec)
	rad = (radius * u.arcsec).to (u.degree).value
	if plot_circles:
		plot.show_circles(circle_ra, circle_dec, radius = rad, color = 'C3')

	plt.show()


#### CORRELATION FUNCTIONS

from astropy.coordinates import SkyCoord

try:
	from sklearn.neighbors import KDTree
except:
	print('no sklearn, no corrfuncs')

def shuffle_along_axis(a, axis):
	idx = np.random.rand(*a.shape).argsort(axis=axis)
	return np.take_along_axis(a,idx,axis=axis)

def shuffle_data(data):
	data_shuff = shuffle_along_axis(data, 0)
	data_shuff[:, :2] = data[:, :2] # retain ra/dec
	return data_shuff

def data_to_xyz(data):
	
	coord_gal = SkyCoord(ra = data[:, 0] * u.degree, dec = data[:, 1] * u.degree, 
				 distance = data[:, 2] * u.kpc, frame = 'icrs').galactocentric
	
	x = coord_gal.x.value
	y = coord_gal.y.value
	z = coord_gal.z.value
	
	return np.vstack((x,y,z)).T

shuffle_axis = 2 # 1: shuffle all except RA 2: shuffle (ra/dec), (distance, pmra, pmdec) tuples

def shuffle_data(data):
	
	idxs = np.arange(len(data))
	randidx_dist = np.random.choice(idxs, size = len(data), replace = False)
	
	data_shuff = np.zeros_like(data)
	data_shuff[:, :2] = data[:, :2]
	data_shuff[:, 2] = data[:, 2][randidx_dist]
	if data.shape[1] == 5:
		randidx_vtra = np.random.choice(idxs, size = len(data), replace = False)
		randidx_vtdec = np.random.choice(idxs, size = len(data), replace = False)
		data_shuff[:, 3] = data[:, 3][randidx_vtra]
		data_shuff[:, 4] = data[:, 4][randidx_vtdec]
	return data_shuff

def data_to_phase(data, w): # 5-d data to phase space in kpc
	
	coord_gal = SkyCoord(ra = data[:, 0] * u.degree, dec = data[:, 1] * u.degree, 
				 distance = data[:, 2] * u.kpc, frame = 'icrs').galactocentric
	
	x = coord_gal.x.value
	y = coord_gal.y.value
	z = coord_gal.z.value
	
	if data.shape[1] == 3:
		
		return np.vstack((x,y,z)).T
	
	else:    
#         vt_ra = pm_vt * data[:, 3]  * data[:, 2] # vt in km/s
#         vt_dec = pm_vt * data[:, 4]  * data[:, 2] # vt in km/s
				
		vt_ra = data[:, 3]
		vt_dec = data[:, 4]

		vt_ra_norm = w * vt_ra
		vt_dec_norm = w * vt_dec
		
		return np.vstack((x,y,z, vt_ra_norm, vt_dec_norm)).T


def two_point_phase_i(data, bins, mock = None, w = None, subsamp = None, method = 'landy-szalay', factor = 1,
					plot_corr = False, verbose = False): 
	
	### if kind == 'spatial', data has 3 features: ra,dec,distance
	### if kind == 'phase', data has 5 features: ra,dec,distance,pmra,pmdec
	### nboot defines number of boostrap samples
	
	if subsamp is None:
		if verbose:
			print('using full dataset for computation')
		n_sample = len(data)
	else:
		if subsamp <= len(data):
			if verbose:
				print('using subsample of %i points for computation' % subsamp)
			n_sample = subsamp
		else:
			if verbose:
				print('data smaller then subsamp, using all %i datapoints' % len(data))
			n_sample = len(data)
	
	idxs = np.random.choice(np.arange(len(data)), size = n_sample, replace = True)
	data = data[idxs]
	
	if data.shape[1] == 3:
		if verbose:
			print('performing spatial correlation')
	
	elif data.shape[1] == 5:
		if verbose:
			print('performing phase-space correlation')
		if w is None:
			print('please provide a w scaling for phase correlations!')
			raise
	
	n_samples, n_features = data.shape

	# NOW DEFINE PHASE AND PHASE_R with the correct dimensions (convert to kpc)

	phase = data_to_phase(data, w)

	if mock is None:
		phase_R = shuffle_data(phase)
	else:
		randidx = np.random.choice(np.arange(len(data)), size = len(data), replace = False)
		data_R = mock[randidx]
		print(len(data_R))
		phase_R = data_to_phase(data_R, w)

	KDT_D = KDTree(phase)
	KDT_R = KDTree(phase_R)

	counts_DD = KDT_D.two_point_correlation(phase, bins)
	counts_RR = KDT_R.two_point_correlation(phase_R, bins)

	DD = np.diff(counts_DD)
	RR = np.diff(counts_RR)

	RR_zero = (RR == 0)
	RR[RR_zero] = 1

	if method == 'standard':
			corr = factor ** 2 * DD / RR - 1

	elif method == 'landy-szalay':

		counts_DR = KDT_R.two_point_correlation(phase, bins)

		DR = np.diff(counts_DR)
		
		if plot_corr:
			plt.figure(figsize = (8, 8))

			plt.plot(DD, label = 'DD')
			plt.plot(RR, label = 'RR', ls = '--')
			plt.plot(DR, label = 'DR')

			plt.legend()

			plt.show()

		corr = (factor ** 2 * DD - 2 * factor * DR + RR) / RR

	corr[RR_zero] = np.nan
	
	return corr

def two_point_phase(data, bins, mock = None, w = None, subsamp = None, method = 'landy-szalay', factor = 1,
					plot_corr = False, nboot = 1, verbose = False):
	
	
	bootstraps = np.zeros((nboot, len(bins) - 1))
	
	for nn in tqdm(range(nboot)):
		corr_i = two_point_phase_i(data = data, bins = bins, mock = mock, w = w, subsamp = subsamp, method = method,
							factor = factor, plot_corr = plot_corr, verbose = verbose)
		bootstraps[nn] = corr_i
		
	corr = np.mean(bootstraps, axis = 0)
	e_corr = np.std(bootstraps, axis = 0, ddof = 1)
	
	return corr, e_corr, bootstraps