import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import manipulate_text as mt
import numpy.ma as ma

def get_data_and_info(fits_path, fits_name):
    #extract data and information about data
    hdul = fits.open(fits_path+fits_name)
    data_uJ = hdul[0].data[0]
    data = data_uJ/1000 #mJ/beam
    hdr = hdul[0].header
    dlt_N_deg = abs(hdr['CDELT1'])
    dlt_n_deg = abs(hdr['CDELT2'])
    N = hdr['NAXIS1']
    n = hdr['NAXIS2']
    nu_data = hdr['CRVAL3']
    nu_BW = hdr['CDELT3']
    HPBW_deg = hdr['BMIN']
    return data, dlt_N_deg, dlt_n_deg, N, n, HPBW_deg, nu_data, nu_BW

def make_coords(N, n, dlt_N, dlt_n, loc='centered'):
    if loc=='centered':
        ax_N_unit = np.linspace(-(N-1)/2, (N-1)/2, N)
        ax_n_unit = np.linspace(-(n-1)/2, (n-1)/2, n)
    elif loc=='edges':
        ax_N_unit = np.linspace(-N/2, N/2, N+1)
        ax_n_unit = np.linspace(-n/2, n/2, n+1)
    return dlt_N*ax_N_unit, dlt_n*ax_n_unit

def find_circular_rings(THETA, th_range):
    ind = np.where(np.logical_or(THETA<th_range[0], THETA>th_range[1]))
    return ind

#puts indices in decending order with respect to a certain axis
def sort_ind(ind, axis=0):
    p = ind[0].argsort()[::-1]
    new_ind = ()
    for elem in ind:
        new_ind += (elem[p],)
    return new_ind

def mask_image(image, ind , sort=True):
    sh = image.shape
    #reshape image if neccessary 
    if not len(sh) == 3:
        n = sh[0]
        N = sh[1]
        block_image = np.transpose(image).reshape(N,n)
    else:
        block_image = image

    #convert block_image to a list of arrays
    block_image_ls = list(block_image)
    
    #sort indices if neccesary
    if sort:
        #reorder ind
        ind = sort_ind(ind)
    
    #remove entries
    for i in range(len(ind[0])):
        block_image_ls[ind[1][i]] = np.delete(block_image_ls[ind[1][i]], ind[0][i], 0)
    
    return block_image_ls

#coords = (x, y)                                                                                                                                               
#ellipse_params = (r2, a, thickness/2)                                                                                                                         
def create_mask(THETA, th_range, ps_ind=None, rl=None, coords=None, ellipse_params=None, data=None,  sigma_rms=None, num_sigma_rms=None, sigma_BW=None, num_sigma_BW=None, dlt=None,\
 method='azimuthally_sym'):
    m0 = 0*THETA==0
    m1 = m0
    m2 = m0
    m3 = m0
    if ps_ind is not None:
        m0[ps_ind]=False
    if method=='azimuthally_sym':
        m1 = np.logical_and(THETA>th_range[0], THETA<th_range[1])
    elif method=='ellipse_and_azimuthally_sym':
        m1 = np.logical_and(THETA>th_range[0], THETA<th_range[1])
        elliptic_d = np.sqrt(coords[0]**2+(ellipse_params[0]**2)*coords[1]**2)
        m2 = np.abs(elliptic_d-ellipse_params[1])>ellipse_params[2]
    if rl=='right':
        m3[np.where(AX_N>=0)] = False
    elif rl=='left':
        m3[np.where(AX_N<=0)] = False
    return m0*m1*m2*m3
    
def make_spot(x, w_spot):
    template = w_spot[2]*np.exp(-((x[0]-w_spot[0])**2+(x[1]-w_spot[1])**2)/(2*w_spot[3]**2))
    return template
    
def f(x, w):
    elliptic_d = np.sqrt(x[0]**2+(w[2]**2)*x[1]**2)
    ell_unif = w[0]+w[1]*np.exp(-(elliptic_d-w[3])**2/(2*w[4]**2))
    res = ell_unif
    return res

def im_to_1d(image, kind='list'):
    if kind=='list':
        image_t = tuple(image)
    res = np.hstack(image_t)
    return res

#args = (data, x, sigma, mask)
def chi2(w, *args):
    model = f(args[1], w)
    return np.sum(((args[0]-model)**2/args[2]**2)*args[3])

def remove_ps(data, THETA, cent, ann1, ann2, thresh):
    #find center index
    ind0 = np.where(THETA==0)
    indy0 = ind0[0][0]
    indx0 = ind0[1][0]

    #find indices of annulus
    ind_ann0 = np.where(np.logical_and(THETA>ann1, THETA<ann2))
    dlt_ind_ann = (ind_ann0[0]-indy0, ind_ann0[1]-indx0)

    #find indices of center
    ind_cent0 = np.where(THETA<cent)
    dlt_ind_cent = (ind_cent0[0]-indy0, ind_cent0[1]-indx0)
    
    #prepare for for-loops
    sh = data.shape
    mask_indx = []
    mask_indy = []
    
    #iterate over each pixel, putting the circle and annulus centered over the pixel and computing the average
    #intensity in each. If the circle has an average intensity larger than the annulus by a certain amount
    #then store the indices of the circle for masking
    for i in range(sh[0]):
        #print('i: ', i)
        for j in range(sh[1]):
            ind_of_ind = [k for k in range(len(dlt_ind_ann[0])) \
                                  if j+dlt_ind_ann[1][k] < N and j+dlt_ind_ann[1][k] >= 0\
                                 and i+dlt_ind_ann[0][k] < n and i+dlt_ind_ann[0][k] >= 0]
            ind_ann_x = [int(j+dlt_ind_ann[1][ind]) for ind in ind_of_ind]
            ind_ann_y = [int(i+dlt_ind_ann[0][ind]) for ind in ind_of_ind]
            num_ann = len(ind_of_ind)

            ind_of_ind_cent = [k for k in range(len(dlt_ind_cent[0])) \
                                  if j+dlt_ind_cent[1][k] < N and j+dlt_ind_cent[1][k] >= 0\
                                 and i+dlt_ind_cent[0][k] < n and i+dlt_ind_cent[0][k] >= 0]
            ind_cent_x = [int(j+dlt_ind_cent[1][ind]) for ind in ind_of_ind_cent]
            ind_cent_y = [int(i+dlt_ind_cent[0][ind]) for ind in ind_of_ind_cent]
            num_cent = len(ind_of_ind_cent)

            #compute average intensity in annulus and center
            I_center = np.sum(data[(ind_cent_y, ind_cent_x)])/num_cent
            I_ann = np.sum(data[(ind_ann_y, ind_ann_x)])/num_ann

            if I_center > thresh[i,j]+I_ann: #or I_center < I_ann-thresh[i,j]: (include if voids should be masked)
                mask_indy += list(ind_cent_y)
                mask_indx += list(ind_cent_x)
                
    #remove repeated pairs of indices from the list of indices
    mask_inds_raw = np.array([mask_indy, mask_indx])
    print('raw indices found')
    mask_inds = np.unique(mask_inds_raw, axis=1)
    
    #plot data with masked indices indicated with scatter points
    plt.imshow(data)
    plt.scatter(mask_inds[1], mask_inds[0], s=2, color='r')
    
    return tuple(mask_inds)

#define relevant paths and names
base_path = os.path.realpath(__file__).split('scripts')[0]
fits_path = base_path.split('Secondary_radiation')[0] + 'synchrotron_data/'
fits_name = 'm31cm3nthnew.ss.90sec.fits'

#extract info about data
data, dlt_N_deg, dlt_n_deg, N, n, HPBW_deg, nu_data, nu_BW = get_data_and_info(fits_path, fits_name)
dlt_N = dlt_N_deg*np.pi/180
dlt_n = dlt_n_deg*np.pi/180
HPBW = HPBW_deg*np.pi/180
sigma_BW = HPBW/(2*np.sqrt(2*np.log(2)))
omega_beam = 2*np.pi*sigma_BW**2
print('real data info extracted')

#create coords for data
ax_N, ax_n = make_coords(N, n, dlt_N, dlt_n)
AX_N, AX_n = np.meshgrid(ax_N, ax_n)
THETA = np.sqrt(AX_N**2+AX_n**2)

#create noise map
sigma_rms = [0.25, 0.3]
l = 2/3
b = 2/3
outside = np.logical_or(np.abs(AX_N)>(l/2)*np.pi/180, np.abs(AX_n)>(b/2)*np.pi/180)
inside = np.logical_not(outside)
noise = sigma_rms[-1]*outside + sigma_rms[0]*inside
rl_mask = 'left'

#define dimensionless hyperparams
dless_rad0 = .75
dless_rad1 = 2.25
dless_rad2 = 2.5
num_sigmas = 4

#compute physical hyperparams
cent = dless_rad0*HPBW/2
ann1 = dless_rad1*HPBW/2
ann2 = dless_rad2*HPBW/2
thresh = num_sigmas*noise

#find point sources
mask_inds = remove_ps(data, THETA, cent, ann1, ann2, thresh)

#deterine extent of central region. To make the determination of the central region more accurate,
#I should not include masks of right or left sides
omega_pix = dlt_N*dlt_n
pix_per_beam = omega_beam/omega_pix
min_pix = pix_per_beam
this_mask = create_mask(THETA, [0, np.pi], ps_ind=mask_inds, coords=(AX_N, AX_n))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.imshow(ma.array(data, mask=np.logical_not(this_mask)), vmin=np.min(data), vmax = np.max(data))
#plt.title(r'ellipse mask thickness = '+ '{:.2e}'.format(thickness) + r' rad')
#plt.colorbar(orientation='horizontal')
#plt.savefig(base_path+'figs/ps_mask_data.pdf')
num_bins = 100
th_bins = np.linspace(0, np.max(THETA), num_bins+1)
flux_bin = np.zeros(num_bins)
pix_bin = np.zeros(num_bins)
er_flux = np.zeros(num_bins)
for i in range(num_bins):
    bool_bin = np.logical_and(THETA>=th_bins[i], THETA<th_bins[i+1])
    pix_bin[i] = np.sum(bool_bin*this_mask)
    flux_bin[i] = np.sum(data*bool_bin*this_mask)

er_ring = np.array([np.sqrt(pix_bin[i]/pix_per_beam)*(np.sum(noise*bool_bin*this_mask)/pix_bin[i]) if not pix_bin[i]==0 else 0 \
           for i in range(num_bins)])
mean_bin = np.array([flux_bin[i]/pix_bin[i] if not pix_bin[i]<min_pix else 0 for i in range(num_bins)])
er_beam = np.array([er_ring[i]*(pix_per_beam/pix_bin[i]) if not pix_bin[i]<min_pix else 0 for i in range(num_bins)])
#fig = plt.figure()
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.scatter(th_bins[0:-1], mean_bin, s=4)
#plt.errorbar(th_bins[0:-1], mean_bin, yerr=er_beam, fmt='none')
#ax = plt.gca()
#ax.set_ylim([-0.75, 0.75])
#plt.xlabel(r'$\theta \rm{(rad)}$', size=16)
#plt.ylabel(r'$S_{ave} \rm{(mJy/beam)}$', size=16)
#plt.title(r'residual flux; ellipse mask thickness = '+ '{:.2e}'.format(thickness) + r' rad')
#plt.savefig(base_path + 'figs/resid_flux_vs_radius_real_data_ps_ellipse_mask.pdf')
min_ind = np.where(np.min(mean_bin) == mean_bin)[0][0] 
th_mask = th_bins[min_ind]
print('radius of central mask: ' + str(th_mask) + ' rad')
#print(pix_bin)

#mask ps and center
th_min = th_mask
th_max = np.pi
th_range_in = [th_min, th_max]
this_mask = create_mask(THETA, th_range_in, ps_ind=mask_inds, rl=rl_mask)
ma_data= ma.array(data, mask=np.logical_not(this_mask))
minmin = np.min(data)
maxmax = np.max(data)
#fig = plt.figure()
#plt.imshow(data, vmin=minmin, vmax=maxmax)
#plt.colorbar(orientation='horizontal')
#plt.savefig(base_path+'figs/real_data.pdf')
#fig = plt.figure()
#plt.imshow(ma_data, vmin=minmin, vmax=maxmax)
#plt.colorbar(orientation='horizontal')
#plt.savefig(base_path+'figs/ps_center_masked_data.pdf')

#find best fit ellipse
w_init = np.array([0,  1,  4,  0.015, 3e-3])
print(w_init)
plt.imshow(ma.array(f([AX_N, AX_n], w_init), mask=np.logical_not(this_mask)))

#print initial statistics
this_chi2 = chi2(w_init, *(data, [AX_N, AX_n],  noise, this_mask))
statement1 = 'initial weights are: '
for j in range(len(w_init)):
    statement1 += 'w' + str(j) + ' = ' + str(w_init[j]) + '; '
print(statement1)
print('chi^2 = ' + str(this_chi2))

# update weights
bnds = [(-1, 1), (0, 2), (3, 5), (.01 , 0.02), (1e-3, 1e-2)]
res = minimize(chi2, w_init, args=(data, [AX_N, AX_n], noise, this_mask), bounds=bnds)

w_fin = res.x
print('w_fin: ', w_fin)
print(res.success)
print(res.message)
chi2_final = chi2(w_fin, *(data, [AX_N, AX_n], noise, this_mask))
print('final chi2: '+ str(chi2_final)) 
#fig = plt.figure()
new_model = f([AX_N, AX_n], w_fin)
#plt.imshow(ma.array(new_model, mask=np.logical_not(this_mask))) 
#plt.colorbar(orientation='horizontal')
#plt.savefig(base_path+'figs/best_ring_model_masked.pdf')
#fig = plt.figure()
#plt.imshow(ma_data)
#plt.colorbar(orientation='horizontal')

#set input parameters
mode = 'load'
starting_sample1 = 0
starting_sample2 = 50000
starting_samples= tuple((starting_sample1, starting_sample2))
type1 = 'noring'
type2 = 'ring'
types = tuple((type1, type2))
num_samples = 50000
ell_params = w_fin[1:].copy()
flt_ellipse = False

#compute masks and quantities that are mask dependent but not intensity map dependent
unit_model = f([AX_N, AX_n], [0,1]+list(w_fin[2:]))
model = f([AX_N, AX_n], [0]+list(w_fin[1:]))
unif = np.ones((n, N))
#create masks and compute quantities that depend on mask but not data
num_masks = 50
mask_set = []
mask_thicknesses = np.linspace(0, 3.5*ell_params[3], num_masks)
for thickness in mask_thicknesses:
    this_mask = create_mask(THETA, th_range_in, ps_ind=mask_inds, coords=(AX_N, AX_n), \
                            ellipse_params=[ell_params[1], ell_params[2], thickness],\
                            method='ellipse_and_azimuthally_sym', rl=rl_mask)
    mask_set.append(this_mask)
    

mask_set = np.array(mask_set)
B_set = np.sum(mask_set*(1/noise)**2, axis=(1,2))
C_set = np.sum(mask_set*(unit_model/noise**2), axis=(1,2))
E_set = np.sum(mask_set*(unit_model**2/noise**2), axis=(1,2))
frac = 2.7
beam_cut = 5

#naming convention - w0: uniform background intensity; w1: ellipse intensity; 
#                    r: ring fit; n: no ring fit; 1: simulated data type 1; 
#                    2: simulated data type 2; chi2: chi^2 assuming a particular model
w0n_1 = np.zeros((num_masks, num_samples))
w0n_2 = np.zeros((num_masks, num_samples))
w0n = tuple((w0n_1, w0n_2))
w0r_1 = np.zeros((num_masks, num_samples))
w0r_2 = np.zeros((num_masks, num_samples))
w0r = tuple((w0r_1, w0r_2))
w1r_1 = np.zeros((num_masks, num_samples))
w1r_2 = np.zeros((num_masks, num_samples))
w1r = tuple((w1r_1, w1r_2))
chi2n_1 = np.zeros((num_masks, num_samples))
chi2n_2 = np.zeros((num_masks, num_samples))
chi2n = tuple((chi2n_1, chi2n_2))
chi2r_1 = np.zeros((num_masks, num_samples))
chi2r_2 = np.zeros((num_masks, num_samples))
chi2r = tuple((chi2r_1, chi2r_2))

#initialize data_resid variables
file_name_base = base_path + 'fake_resid_new/sigma_rms_'+ str(sigma_rms[0]) + '_' + str(sigma_rms[-1]) + '_spacing_' + '{:.2e}'.format(dlt_N/frac) + '_samplenum_'

for i in range(num_samples+1):
    print('starting to generate sample number ' + str(i))
    #generate fake data residuals                                                                                    
    file_names = [file_name_base+str(i+ss)+'.npy' for ss in starting_samples]
    if mode == 'save' or mode == 'none':
        if i<num_samples:
            data_resid1, _, _ = gen_fake_data(beam_cut, frac, sigma_rms, sigma_BW, N, n, dlt_N, dlt_n)
            data_resid2, _, _ = gen_fake_data(beam_cut, frac, sigma_rms, sigma_BW, N, n, dlt_N, dlt_n)
            data_resids = [data_resid1, data_resid2]
            if mode == 'save':
                for file, resid in zip(file_names, data_resids):
                    np.save(file, resid)
            print('sample number '+ str(i) + ' generated')
        else:
            data_resid1 = data
            data_resid2 = data
            data_resids = [data_resid1, data_resid2]
    elif mode == 'load':
        if i<num_samples:
            data_resids = (np.load(file) for file in file_names)
        else:
            data_resid1 = data
            data_resid2 = data
            data_resids = [data_resid1, data_resid2]
    else:
        print('value mode is not recognized')
    if i<num_samples:
        fake_data = [resid if t=='noring' else resid+model for resid, t in zip(data_resids, types)]
    else:
        fake_data = data_resids
    
    A_set = [np.sum(mask_set*fdata/noise**2, axis=(1,2)) for fdata in fake_data]
    if flt_ellipse:
        D_set = [np.sum(mask_set*fdata*unit_model/noise**2, axis=(1,2)) for fdata in fake_data]
        this_w0n = [a/B_set for a in A_set]
        this_w0r = [(d-a*E_set/C_set)/(C_set-B_set*E_set/C_set) for a, d, in zip(A_set, D_set)]
        this_w1r = [a/C_set-w*B_set/C_set for a, w in zip(A_set, this_w0r)]
        this_chi2n = [np.sum((mask_set*fdata-mask_set*np.multiply.outer(w, unif))**2/noise**2, axis=(1,2)) for fdata,w in zip(fake_data, this_w0n)]
        this_chi2r = [np.sum((mask_set*fdata-mask_set*(np.multiply.outer(w0, unif)+np.multiply.outer(w1, unit_model)))**2/noise**2, axis=(1,2)) for fdata,w0,w1 in zip(fake_data, this_w0r, this_w1r)]
    else:
        this_w0n = [a/B_set for a in A_set]
        this_w0r = [(a-w_fin[1]*C_set)/B_set for a in A_set]
        this_w1r = [w_fin[1] for k in range(2)]
        this_chi2n = [np.sum((mask_set*fdata-mask_set*np.multiply.outer(w, unif))**2/noise**2, axis=(1,2)) for fdata,w in zip(fake_data, this_w0n)]
        this_chi2r = [np.sum((mask_set*(fdata-w_fin[1]*unit_model)-mask_set*np.multiply.outer(w0, unif))**2/noise**2, axis=(1,2)) for fdata,w0 in zip(fake_data, this_w0r)]

    if i < num_samples:
        for k in range(len(this_chi2r)):
            w0n[k][:, i] = this_w0n[k]
            w0r[k][:, i] = this_w0r[k]
            w1r[k][:, i] = this_w1r[k]
            chi2n[k][:, i] = this_chi2n[k]
            chi2r[k][:, i] = this_chi2r[k]
    else:
        dchi2_data = this_chi2n[0]-this_chi2r[0]
dchi2 = tuple((cn-cr for cn, cr in zip(chi2n, chi2r)))

#modified file names on 11/29
flt_str = ''
if flt_ellipse:
    flt_str = '_float_ellipse'
#and 12/7
flt_str = rl_mask + '_' 
np.save(base_path+'results_compare_rvnr/dchi2_'+types[0]+ flt_str + '.npy', dchi2[0])
np.save(base_path+'results_compare_rvnr/dchi2_'+types[1] + flt_str + '.npy', dchi2[1])
print('dchi^2 computed and saved')

#compute mean of dchi^2 and plot it
mean_dchi2 = tuple((np.mean(dist[:,:-1], axis=-1) for dist in dchi2))
std_dchi2 = tuple((np.std(dist[:,:-1], axis=-1) for dist in dchi2))
np.save(base_path+'results_compare_rvnr/dchi2_mean_'+types[0] +flt_str + '.npy', mean_dchi2[0])
np.save(base_path+'results_compare_rvnr/dchi2_mean_'+types[1] + flt_str + '.npy', mean_dchi2[1])
np.save(base_path+'results_compare_rvnr/dchi2_std_'+types[0] + flt_str + '.npy', std_dchi2[0])
np.save(base_path+'results_compare_rvnr/dchi2_std_'+types[1] + flt_str + '.npy', std_dchi2[1])
np.save(base_path+'results_compare_rvnr/maskthicknesses'+flt_str +'.npy', mask_thicknesses)

#plt.rc('text', usetex=True)                                                                                       
#plt.rc('font', family='serif') 
#plt.errorbar(mask_thicknesses, mean_dchi2[0], yerr=std_dchi2[0]/np.sqrt(num_samples), label=types[0])
#plt.errorbar(mask_thicknesses, mean_dchi2[1], yerr=std_dchi2[1]/np.sqrt(num_samples), label=types[1])
#plt.plot(mask_thicknesses, dchi2_data, label='data')
#plt.xlabel(r'mask thickness (rad)', size=15)
#plt.ylabel(r'$\Delta \chi^2 = \chi^2_n-\chi^2_r$', size=15)
#plt.title(r'Improvement of ring fit over no ring fit')
#plt.yscale('log')
#plt.legend()
#print(base_path+'results_compare_rvnr/dchi2_vs_maskthicknesses.pdf')
#plt.savefig(base_path+'results_compare_rvnr/dchi2_vs_maskthicknesses.pdf')
