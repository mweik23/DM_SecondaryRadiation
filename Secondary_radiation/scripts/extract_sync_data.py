#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 09:37:35 2021

@author: mitchellweikert
"""

import numpy as np
from astropy.io import fits
from os import getcwd
import matplotlib.pyplot as plt

this_path = getcwd()
fits_path = this_path.split('Secondary_radiation')[0] + 'synchrotron_data/'
fits_name = 'm31cm3nthnew.ss.90sec.fits'
hdul = fits.open(fits_path+fits_name)
data = hdul[0].data
hdr = hdul[0].header
dlt1 = abs(hdr['CDELT1'])
dlt2 = abs(hdr['CDELT2'])
N1 = hdr['NAXIS1']
N2 = hdr['NAXIS2']
half_N1 = int((N1-1)/2)
half_N2 = int((N2-1)/2)
ax1_unit = np.linspace(-half_N1, half_N1, N1)
ax2_unit = np.linspace(-half_N2, half_N2, N2)
ax1_deg = dlt1*ax1_unit
ax2_deg = dlt2*ax2_unit

dlt_arcmin = dlt1*60
dlt_rad = dlt1*np.pi/180

AX1, AX2 = np.meshgrid(ax1_deg, ax2_deg)
plt.imshow(data[0], extent=[ax1_deg[0], ax1_deg[-1], ax2_deg[0], ax2_deg[-1]])
plt.xlabel('angle from center (deg)')
plt.ylabel('angle from center (deg)')
plt.colorbar(orientation='horizontal')
figs_dir = this_path.split('scripts')[0] + '/figs/'
plt.savefig(figs_dir + 'M31_real_data_3cm_90arcsecs.pdf')
plt.show()