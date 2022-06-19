# Functions to aid in the discovery and analysis of dwarf galaxies. 

from . import read_mist_models
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy import stats
import scipy
import glob
from astropy.coordinates import SkyCoord
import pandas as pd

def load_mist(mistfeh, mistage):

	mistpath = '/Users/vedantchandra/0_research/data/mist/'

	mist = Table(read_mist_models.ISOCMD('%s/iso_gaia_%sgyr_feh%s.cmd' % (mistpath, 
																		  mistage, 
																		  mistfeh)).isocmds[0])
	pmist = Table(read_mist_models.ISOCMD('%s/iso_ps_%sgyr_feh%s.cmd' % (mistpath, 
																		 mistage, 
																		 mistfeh)).isocmds[0])

	mist['i_z'] = pmist['PS_i'] - pmist['PS_z']
	mist['z'] = pmist['PS_z']
	mist['g'] = pmist['PS_g']
	mist['g_r'] = pmist['PS_g'] - pmist['PS_r']
	mist['r'] = pmist['PS_r']
	mist['r_i'] = pmist['PS_r'] - pmist['PS_i']
	mist['i'] = pmist['PS_i']
	mist['G'] = mist['Gaia_G_EDR3']
	mist['bp_rp'] = mist['Gaia_BP_EDR3'] - mist['Gaia_RP_EDR3']
	mist['g_rp'] = mist['Gaia_G_EDR3'] - mist['Gaia_RP_EDR3']

	mist['msrgb'] = (mist['EEP'] > 300) & (mist['EEP'] < 600)
	mist['bhb'] = ((mist['EEP'] > 600) & (mist['EEP'] < 610)) | ((mist['EEP'] > 620) & (mist['EEP'] < 650))

	return mist

def make_isomask(mist, dm, 
				 mist_bhb = None,
				 color = 'g_r', 
				 mag = 'g', 
				 color_width = 0.02, 
				 mag_width = 0.2, 
				 bhb_width = 0.3, 
				 colormag_fac = 1e-2, 
				 lim_mag = 22, 
				 sat_mag = 0,
				deltacolor = 0,
				deltamag = 0):

	if mist_bhb is None:
		mist_bhb = mist[mist['bhb']]
	else:
		mist_bhb = mist_bhb[mist_bhb['bhb']]

	mist_msrgb = mist[mist['msrgb']]

	left_color_rgb = mist_msrgb[color] - color_width - colormag_fac * (mist_msrgb[mag]) + deltacolor
	right_color_rgb = mist_msrgb[color] + color_width + colormag_fac * (mist_msrgb[mag]) + deltacolor
	lower_mag_rgb = mist_msrgb[mag] - mag_width + dm + deltamag
	upper_mag_rgb = mist_msrgb[mag] + mag_width + dm + deltamag

	left_color_bhb = mist_bhb[color] - color_width - colormag_fac * (mist_bhb[mag]) + deltacolor
	right_color_bhb = mist_bhb[color] + color_width + colormag_fac * (mist_bhb[mag]) + deltacolor
	lower_mag_bhb = mist_bhb[mag] - bhb_width + dm + deltamag
	upper_mag_bhb = mist_bhb[mag] + bhb_width + dm + deltamag

	color_loop = np.concatenate((left_color_rgb[:230], right_color_bhb[10:], 
								 left_color_bhb[::-1], right_color_rgb[::-1]))
	mag_loop = np.concatenate((lower_mag_rgb[:230], upper_mag_bhb[10:], 
							   lower_mag_bhb[::-1], upper_mag_rgb[::-1]))
	
	in_limits = (mag_loop < lim_mag) & (mag_loop > sat_mag)
	
	color_loop = color_loop[in_limits]
	mag_loop = mag_loop[in_limits]

	iso_mask = plt.Polygon(np.vstack((color_loop, mag_loop)).T)
	
	return iso_mask, color_loop, mag_loop


def mg_bhb(gr):
	# returns PS BHB ridegline from Deason et al. 2011

	return 0.434 - 0.169*gr + 2.319*gr**2 + 20.449*gr**3 + 94.517*gr**4



def kop_filter(delra, delde, r = 15, R = 100, bins = 100, 
			   lim = None, boundary = None, vmin = None, vmax = None,
			  factor = 1, cmap = 'viridis'):

	r = (r * u.arcmin).to(u.deg)
	R = (R * u.arcmin).to(u.deg)
	
	if lim is None:
		lim = np.max([np.abs(delra.min()), delra.max(), np.abs(delde.min()), delde.max()])

	cts,x,y = np.histogram2d(delra, delde, 
							 bins = bins,
							range = ([[-lim, lim],[-lim,lim]]))
	
	cts = cts.T.copy() #np.rot90(cts).copy()
	side = (x.max() - x.min()) * u.degree
	pix_size = side / bins

	sig1 = (r / pix_size).value
	sig2 = (R / pix_size).value
	
	inner_kernel = Gaussian2DKernel(x_stddev = sig1, y_stddev = sig1,
								   factor = factor, mode = 'center')

	outer_kernel = Gaussian2DKernel(x_stddev = sig2, y_stddev = sig2,
								   factor = factor, mode = 'center')

	M_bg = convolve(cts, outer_kernel, boundary = boundary)
	M_inner = convolve(cts, inner_kernel, boundary = boundary)
	M_conv = (M_inner - M_bg).copy()
	
	S = np.sqrt(4 * np.pi) * sig1 * M_conv / (np.sqrt(M_bg))
	
	floor = np.mean(S)
		
	plt.pcolormesh(x, y, S, cmap = cmap, vmin = vmin, vmax = vmax)
		
	cbar = plt.colorbar(orientation = 'horizontal', location = 'bottom', pad = 0.15)
	
	cbar.ax.set_xlabel('Detection Significance')
	
	plt.gca().invert_xaxis()

	plt.xlabel(r'$\Delta\alpha\cos{\delta}$')
	plt.ylabel(r'$\Delta{\delta}$')

def hess_diff(xcol, ycol, mftable, bgtable, bins = 25, statistic = 'count',
				 xlim = None, ylim = None, vmax = None, vmin = None, inv_y = False, inv_x = False,
				 cmap = 'Greys', smooth = 0):

	x = mftable[xcol]
	y = mftable[ycol]

	x_bg = bgtable[xcol]
	y_bg = bgtable[ycol]
	
	statistic = 'count'#np.nanmedian
	
	if xlim is None:
		bin_range = None
	else:
		bin_range = (xlim, ylim)

	bs = stats.binned_statistic_2d(x, y, y, bins = bins, statistic = statistic,
								  range = bin_range)
	bs_bg = stats.binned_statistic_2d(x_bg, y_bg, y, bins = [bs.x_edge, bs.y_edge], statistic = statistic,
									 range = bin_range)
	
	mf_counts = bs.statistic
	bg_counts = bs_bg.statistic
	
	mf_density = mf_counts / np.sum(mf_counts)
	bg_density = bg_counts / np.sum(bg_counts)

	hdiff = (mf_density - bg_density) * np.sum(bg_counts) #/ np.sqrt(bg_counts)
	#hdiff[np.isnan(hdiff)] = 0.0

	if smooth > 0:
		mf_counts = scipy.ndimage.gaussian_filter(mf_counts, smooth)
		bg_counts = scipy.ndimage.gaussian_filter(bg_counts, smooth)
		hdiff = scipy.ndimage.gaussian_filter(hdiff, smooth)
	

	#hdiff = (bs.statistic - bs_bg.statistic) / bs_bg.statistic

	plt.figure(figsize = (21, 10))

	plt.subplot(131)
	plt.pcolormesh(bs.x_edge, bs.y_edge, mf_counts.T, cmap = cmap);
	cbar = plt.colorbar(location = 'bottom')
	cbar.ax.set_xlabel('Density')

	plt.xlabel(xcol)
	plt.ylabel(ycol)
	plt.title('Selection')
	plt.xlim(xlim)
	if inv_y:
		plt.gca().invert_yaxis()
	if inv_x:
		plt.gca().invert_xaxis()
	
	plt.subplot(132)
	plt.pcolormesh(bs_bg.x_edge, bs_bg.y_edge, bg_counts.T, cmap = cmap);
	cbar = plt.colorbar(location = 'bottom')
	cbar.ax.set_xlabel('Density')

	plt.xlabel(xcol)
	plt.ylabel(ycol)
	plt.title('Background')
	plt.xlim(xlim)
	if inv_y:
		plt.gca().invert_yaxis()
	if inv_x:
		plt.gca().invert_xaxis()

	plt.subplot(133)

	plt.pcolormesh(bs.x_edge, bs.y_edge, (hdiff).T, cmap = cmap, vmin = vmin, vmax = vmax);
	cbar = plt.colorbar(location = 'bottom')
	cbar.ax.set_xlabel('Significance')

	plt.xlabel(xcol)
	plt.ylabel(ycol)
	plt.title('Density Difference')
	plt.xlim(xlim)
	plt.ylim(ylim)
	if inv_y:
		plt.gca().invert_yaxis()
	if inv_x:
		plt.gca().invert_xaxis()

	return hdiff, mf_density, bg_density

def hdiff_cmd(gcol_i, gmag_i, table, bgtable,
			  bin_range = None, bins = 30, statistic = 'count', vmax = None,
			 sigma = 1):
	
	x = table[gcol_i]
	y = table[gmag_i]

	x_bg = bgtable[gcol_i]
	y_bg = bgtable[gmag_i]
	
	if bin_range is None:
		bin_range = [[np.min(x_bg), np.max(x_bg)], [np.min(y_bg), np.max(y_bg)]]

	
	bs = stats.binned_statistic_2d(x, y, y, bins = bins, statistic = statistic,
							  range = bin_range)

	bs_bg = stats.binned_statistic_2d(x_bg, y_bg, y, bins = [bs.x_edge, bs.y_edge], statistic = statistic,
									 range = bin_range)

	mf_density = bs.statistic / np.sum(bs.statistic)
	bg_density = bs_bg.statistic / np.sum(bs_bg.statistic)
	hdiff = (mf_density - bg_density) * np.sum(bs.statistic) #/ np.sqrt(bs.statistic)

	hdiff_smooth = scipy.ndimage.filters.gaussian_filter(hdiff, sigma = sigma)

	plt.pcolormesh(bs.x_edge, bs.y_edge, hdiff_smooth.T, cmap = 'Greys', vmin = 0, vmax = vmax)

	plt.ylim(bin_range[1])
	plt.gca().invert_yaxis()

	return mf_density, bg_density


def query_gaia(ra, dec, radius, constraints = ''):

	gaia_folder = '/Users/vedantchandra/0_research/data/gaia_giants/'
	gaia_files = np.sort(glob.glob(gaia_folder + '*.h5'))
	
	ras = [ra - radius, ra + radius]
	decs = [dec - radius, dec + radius]
	
	c = SkyCoord(ra = ras * u.deg, dec = decs * u.deg)
	ls, bs = c.galactic.l.value, c.galactic.b.value
					  
	if bs.mean() > 0:
		bmin = 20
		bmax = 90
		
	elif bs.mean() < 0:
		bmin = -90
		bmax = -20


	query = '(ra > %.5f) & (ra < %.5f) & (dec > %.5f) & (dec < %.5f) ' % (ra - radius, ra + radius,
																	 dec - radius, dec + radius)
	query = query + constraints

	bs = np.arange(bmin, bmax)

	dfs = [];
	for ii in range(len(bs)-1):
		
		bmin_str = str(bs[ii]).replace('-','m')
		bmax_str = str(bs[ii+1]).replace('-','m')
		filename = gaia_folder + 'gaia_giants_xphot_b_%s_%s.h5' % (bmin_str, bmax_str)

		dfi = pd.read_hdf(filename, where = query)
		
		dfi = dfi.replace(99.0, np.nan) # replace bad values
		
		dfs.append(dfi)

	return Table.from_pandas(pd.concat(dfs))


def nearby_streams(ra, dec, distance = 0, rv = 0, pmra = 0, pmdec = 0,
				   ang_sep = 5, win = 10, pm_win = 5, name = ""):

	import galstreams
	mwsts = galstreams.MWStreams(verbose=False, implement_Off = False)

	nearstr = [];

	for key in mwsts.keys():
		trackra = mwsts[key].track.ra.value
		trackdec = mwsts[key].track.dec.value
		d_str = np.sqrt(((trackra - ra)*np.cos(np.radians(trackdec)))**2 + (trackdec - dec)**2)

		if np.min(d_str) < ang_sep:
			nearstr.append(key)

	plt.figure(figsize = (10, 10))

	### SPATIAL ####
	plt.subplot(221)
	plt.scatter(ra, dec, color = 'C3', s = 350, marker = '*', edgecolor = 'k',
			   zorder = 10)

	for key in nearstr:
		trackra = mwsts[key].track.ra.value
		trackdec = mwsts[key].track.dec.value
		plt.scatter(trackra, trackdec, lw = 1, label = key)
	plt.xlim(ra + win, ra - win)
	plt.ylim(dec - win, dec + win)
	plt.legend(fontsize = 10)
	plt.xlabel("RA")
	plt.ylabel("DEC")
	plt.title(name)

	### DISTANCE ####

	plt.subplot(222)
	plt.scatter(ra, distance, color = 'C3', s = 350, marker = '*', edgecolor = 'k',
			   zorder = 10)

	for key in nearstr:
		trackra = mwsts[key].track.ra.value
		trackdec = mwsts[key].track.dec.value
		trackdist = mwsts[key].track.distance.value
		d_str = np.sqrt(((trackra - ra)*np.cos(np.radians(trackdec)))**2 + (trackdec - dec)**2)
		dsel = d_str < win
		plt.scatter(trackra[dsel], trackdist[dsel])

	plt.xlim(ra + win, ra - win)
	plt.xlabel("RA")
	plt.ylabel("Distance [kpc]")
	
	### RV ####

	plt.subplot(223)
	plt.scatter(ra, rv, color = 'C3', s = 350, marker = '*', edgecolor = 'k',
			   zorder = 10)

	for key in nearstr:
		trackra = mwsts[key].track.ra.value
		trackdec = mwsts[key].track.dec.value
		trackdist = mwsts[key].track.distance.value
		d_str = np.sqrt(((trackra - ra)*np.cos(np.radians(trackdec)))**2 + (trackdec - dec)**2)
		dsel = d_str < win
		plt.scatter(trackra[dsel], mwsts[key].track.radial_velocity[dsel])

	plt.xlim(ra + win, ra - win)
	plt.xlabel("RA")
	plt.ylabel("RV [km/s]")

	### PROPER MOTIONS ####

	plt.subplot(224)
	plt.scatter(pmra, pmdec, color = 'C3', s = 350, marker = '*',
			   edgecolor = 'k')

	for key in nearstr:
		trackra = mwsts[key].track.ra.value
		trackdec = mwsts[key].track.dec.value
		d_str = np.sqrt(((trackra - ra)*np.cos(np.radians(trackdec)))**2 + (trackdec - dec)**2)
		dsel = d_str < win
		plt.scatter(mwsts[key].track.pm_ra_cosdec[dsel], mwsts[key].track.pm_dec[dsel])

	plt.xlabel('$\mu_{RA}$')
	plt.ylabel("$\mu_{DEC}$")
	plt.xlim(pmra + pm_win, pmra - pm_win)
	plt.ylim(pmdec - pm_win, pmdec + pm_win)
	plt.gca().invert_xaxis()

	plt.tight_layout()