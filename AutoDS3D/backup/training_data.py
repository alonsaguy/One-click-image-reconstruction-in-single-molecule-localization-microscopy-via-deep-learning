
import numpy as np
from DS3Dplus.ds3d_utils import ImModelTraining, Sampling
import torch
import os
import pickle
import matplotlib
import shutil
from skimage import io
import scipy.io as sio
import time

matplotlib.use("TkAgg")
np.random.seed(66)
torch.manual_seed(88)

def param_set():

    if torch.cuda.device_count()>1:  # on server
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:  # local PC
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    psf_param_dict = dict(
        device=device,
        # objective
        M=100,  # magnification
        NA=1.49,  # NA
        n_immersion=1.518,  # refractive index of the immersion of the objective
        lamda=0.58,  # wavelength
        n_sample=1.33,  # refractive index of the sample
        f_4f=80e3,  # focal length of 4f system
        ps_camera=15.9,  # pixel size of the camera
        ps_BFP=20,  # pixel size at back focal plane
    )

    mask_dict = sio.loadmat('./laminB1_dh.mat')
    mask_name = list(mask_dict.keys())[3]
    phase_mask = mask_dict[mask_name]
    psf_param_dict['phase_mask'] = phase_mask

    psf_param_dict['g_sigma'] = (0.9*0.86, 1.1*0.86)
    psf_param_dict['NFP'] = 1.1
    psf_param_dict['zrange'] = [0.0, 1.7]


    td_param_dict = dict(
        # 3D volume corresponding to each 2D image input
        H=121,  # size of training images
        W=121,  # size of training images
        us_factor=2,  # up-sampling factor, you may want to have finer xy voxel size
        D=40,  # voxel number in the specified z range

        # SNR
        Nsig_range=[15e4, 50e4],  # photon count range
        baseline=[600, 1400],
        read_std=[300, 600],  # standard deviation of readout noise
        non_uniform_noise_flag=True,  # non-uniform noise std due to non-uniform background

        # molecule density
        num_particles_range=[1, 35],  # emitter count range
        psf_half_size=20,  # unit: pixel, periphery of no molecule, avoid PSF cropping

        # image normalization
        project_01=False,  # 01 image normalization, not recommended

        # folder paths for training
        n_ims=10000,  # the number of training images
        td_folder=os.path.join(os.path.join(os.getcwd(), 'training_data'), 'laminB1_us2'),
        training_result_path=os.path.join(os.getcwd(), 'training_results'),

        # other parameters
        bitdepth=16,
        blob_r=2,
        blob_sigma=0.65,
        blob_maxv=1000,

    )

    param_dict = {**psf_param_dict, **td_param_dict}

    param_dict['HH'] = int(param_dict['H'] * param_dict['us_factor'])
    param_dict['WW'] = int(param_dict['W'] * param_dict['us_factor'])
    param_dict['buffer_HH'] = int(param_dict['psf_half_size'] * param_dict['us_factor'])
    param_dict['buffer_WW'] = int(param_dict['psf_half_size'] * param_dict['us_factor'])
    param_dict['ps_xy'] = param_dict['ps_camera'] / param_dict['M']
    param_dict['vs_xy'] = param_dict['ps_xy'] / param_dict['us_factor']
    param_dict['vs_z'] = ((param_dict['zrange'][1] - param_dict['zrange'][0]) / param_dict['D'])
    print(f"vs_xy: {param_dict['vs_xy']} um, vs_z: {param_dict['vs_z']} um")

    return param_dict


param_dict = param_set()
device = param_dict['device']

# imaging model
model = ImModelTraining(param_dict)
sampling = Sampling(param_dict)

# start
td_folder = param_dict['td_folder']
if os.path.exists(td_folder):  # delete the directory if it exists
    shutil.rmtree(td_folder)
x_folder = td_folder+'/x'
os.makedirs(x_folder)  # make the folder for training data

t0 = time.time()
# labels_dict for training
labels_dict = {}
labels_dict['volume_size'] = (param_dict['D'], param_dict['HH'], param_dict['WW'])
labels_dict['us_factor'] = param_dict['us_factor']
labels_dict['blob_r'] = sampling.blob_r  # radius of each 3D blob representing an emitter in space
labels_dict['blob_maxv'] = sampling.blob_maxv  # maximum value of blobs

ntrain = param_dict['n_ims']
for i in range(ntrain):
    xyzps, xyz_ids, blob3d = sampling.xyzp_batch()
    im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.uint16)
    if param_dict['project_01']:
        im = ((im-im.min())/(im.max()-im.min()))

    x_name = str(i).zfill(5) + '.tif'
    io.imsave(os.path.join(x_folder, x_name), im, check_contrast=False)
    labels_dict[x_name] = (xyz_ids, blob3d)

    if i % (ntrain//10) == 0:
        print('Training Example [%d / %d]' % (i + 1, ntrain))
print('Training Example [%d / %d]' % (ntrain, ntrain))

y_file = os.path.join(td_folder, r'y.pickle')
with open(y_file, 'wb') as handle:
    pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

param_file = os.path.join(td_folder, r'param.pickle')
with open(param_file, 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

t1 = time.time()
print(f'finished generating training data in {t1-t0}s.')
