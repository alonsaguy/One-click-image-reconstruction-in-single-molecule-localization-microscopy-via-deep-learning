
from app_utils import (phase_retrieval, background_removal, mu_std_p, training_data_func, training_func,
                       inference_func1, inference_func2)
from app_utils import show_z_psf
import numpy as np
import torch
import os
import time


# phase retrieval
def func1(M, NA, lamda, n_immersion, n_sample, f_4f, ps_camera, ps_BFP, zstack_file, nfp_text,
          NFP, zrange, device_id, state):
    # assemble a shared parameter dict
    param_dict = dict(
        M=M,  # magnification
        NA=NA,  # NA
        lamda=lamda,  # emission wavelength
        n_immersion=n_immersion,  # refractive index of the immersion of the objective
        n_sample=n_sample,  # refractive index of the sample
        f_4f=f_4f,  # focal length of 4f system
        ps_camera=ps_camera,  # pixel size of the camera
        ps_BFP=ps_BFP,  # pixel size at back focal plane
        # g_sigma=1.0,  # initial std of the gaussian blur kernel
        g_sigma=0.6,  # initial std of the gaussian blur kernel
        device=torch.device('cuda:'+str(device_id) if torch.cuda.is_available() else 'cpu'),
    )

    # a dict for phase retrieval
    nfp_text = nfp_text.split(',')
    nfps = np.linspace(float(nfp_text[0]), float(nfp_text[1]), int(nfp_text[2]))
    pr_dict = dict(
        zstack_file_path=os.path.join(os.getcwd(), zstack_file),
        nfps=nfps,
        r_bead=0.02,  # a default value, not very important
        epoch_num=250,  # optimization iterations
        loss_label=1,  # 1: gauss log likelihood, 2: l2
    )

    phase_mask, g_sigma, ccs = phase_retrieval(param_dict, pr_dict)
    g_sigma = (np.round(0.9*g_sigma, decimals=2), np.round(1.1*g_sigma, decimals=2))
    param_dict['g_sigma'] = g_sigma
    param_dict['phase_mask'] = phase_mask
    print(f'PSF modeling accuracy: average cc of {np.round(np.mean(ccs), decimals=4)}')
    print(f'blur sigma: ({g_sigma[0]}, {g_sigma[1]})')

    # with open('param_dict3.pickle', 'wb') as handle:
    #     pickle.dump(param_dict, handle)

    # with open('param_dict3.pickle', 'rb') as handle:
    #     param_dict = pickle.load(handle)

    # show z-PSF regarding the NFP
    param_dict['NFP'] = NFP
    zrange = zrange.split(',')
    zrange = (float(zrange[0]), float(zrange[1]))
    param_dict['zrange'] = zrange  # a tuple
    param_dict['baseline'] = None   # required by imaging model, None for now
    param_dict['read_std'] = None
    param_dict['bg'] = None
    show_z_psf(param_dict)  # generate PSFs.jpg

    # save results for other blocks
    if 'param_dict' in state.keys():
        state['param_dict'] = {**state['param_dict'], **param_dict}
    else:
        state['param_dict'] = param_dict

    return 'PSF characterization is done. Check phase_retrieval_results.jpg and PSFs.jpg'


# background removal
def func2(raw_image_folder, state):  # preprocessing
    raw_image_folder = os.path.join(os.getcwd(), raw_image_folder)  # assemble the folder directory
    im_br_folder = background_removal(raw_image_folder)
    # im_br_folder = raw_image_folder+'_br'  # for test

    if not 'param_dict' in state.keys():  # in the case of preprocessing images before characterizing PSF
        state['param_dict'] = dict()

    param_dict = state['param_dict']

    param_dict['im_br_folder'] = im_br_folder

    print(f'Images after background removal are in {im_br_folder}.')

    return f'Background removal is done. Check folder {im_br_folder}'


# snr estimation
def func3(photon_roi, max_pv, state):
    param_dict = state['param_dict']

    # option 1, read the ROI from the GUI
    photon_roi = photon_roi.split(',')
    photon_roi = (int(photon_roi[0]), int(photon_roi[1]), int(photon_roi[2]), int(photon_roi[3]))
    # option 2, interactively choose ROI, inside the function

    noise_dict = dict(
        num_ims=1000,  # analyze this number of images at the end of the cleaned blinking images/video
        photon_roi=photon_roi,
        max_pv=max_pv,
    )

    mu, std, p = mu_std_p(param_dict, noise_dict)

    baseline = (np.round(1.0*mu), np.round(1.4*mu))
    read_std = (np.round(1.0*std), np.round(1.4*std))
    Nsig_range = (np.round(0.5*p/1e3)*1e3, np.round(1.1*p/1e3)*1e3)
    param_dict['baseline'] = baseline
    param_dict['read_std'] = read_std
    param_dict['Nsig_range'] = Nsig_range  # the param_dict in state will also be updated
    param_dict['bg'] = 0

    print(f'noise baseline: ({baseline[0]}, {baseline[1]})')
    print(f'noise std: ({read_std[0]}, {read_std[1]})')
    print(f'photon: ({Nsig_range[0]}, {Nsig_range[1]})')

    return 'Obtained noise parameters.'


# training data generation
def func4(num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, projection_01, state):  # training data
    param_dict = state['param_dict']
    # complete the param_dict
    param_dict['H'] = training_im_size
    param_dict['W'] = training_im_size
    param_dict['D'] = num_z_voxel
    param_dict['us_factor'] = us_factor
    param_dict['psf_half_size'] = 20  # pixels
    param_dict['num_particles_range'] = [1, max_num_particles]
    param_dict['blob_r'] = 2
    param_dict['blob_sigma'] = 0.65
    param_dict['blob_maxv'] = 800   # maximum value of network output

    param_dict['HH'] = int(param_dict['H'] * us_factor)  # in case upsampling is needed
    param_dict['WW'] = int(param_dict['W'] * us_factor)
    param_dict['buffer_HH'] = int(param_dict['psf_half_size'] * us_factor)
    param_dict['buffer_WW'] = int(param_dict['psf_half_size'] * us_factor)

    vs_xy = param_dict['ps_camera'] / param_dict['M'] / us_factor  # index of each voxel is at the center of the voxel
    vs_z = ((param_dict['zrange'][1] - param_dict['zrange'][0]) / param_dict['D'])   # no buffer zone in z axis
    param_dict['vs_xy'] = vs_xy
    param_dict['vs_z'] = vs_z

    param_dict['td_folder'] = os.path.join(os.getcwd(), 'training_data')  # where to save the training data
    if projection_01==0:
        param_dict['project_01'] = False  # seems better to not have 01 normalization
    else:
        param_dict['project_01'] = True

    param_dict['n_ims'] = num_training_images  # the number of images for training

    training_data_func(param_dict)

    return f'Data generation is done. data folder: {param_dict["td_folder"]}'


# training
def func5(state):
    param_dict = state['param_dict']
    param_dict['path_save'] = os.path.join(os.getcwd(), 'training_results')

    training_dict = dict(
        batch_size=16,
        lr=0.001,  # 0.005?
        num_epochs=50,
    )

    net_file, fit_file = training_func(param_dict, training_dict)
    param_dict['net_file'] = net_file
    param_dict['fit_file'] = fit_file

    # param_dict['net_file'] = 'net_01-23_17-02.pt'  # for test
    # param_dict['fit_file'] = 'fit_01-23_17-02.pickle'

    return f'Training is done. result folder: {param_dict["path_save"]}'


# inference
def func6_1(threshold, test_idx, state):
    param_dict = state['param_dict'].copy()  # incase
    param_dict['threshold'] = threshold
    inference_func1(param_dict, test_idx)

    return 'Inference test is done. check loss_curves.jpg, sim_loc_gt_rec.jpg, sim_im_gt_rec.jpg, exp_im_gt_rec.jpg'

def func6_2(threshold, state):
    param_dict = state['param_dict'].copy()
    param_dict['threshold'] = threshold

    file_name = inference_func2(param_dict)

    return f'Obtained a complete localization list: {file_name}'


# one click
def func7(M, NA, lamda, n_immersion, n_sample, f_4f, ps_camera, ps_BFP, zstack_file, nfp_text, NFP, zrange, device_id,
          raw_image_folder, photon_roi, max_pv, num_z_voxel, training_im_size, us_factor,
          max_num_particles, num_training_images, projection_01, test_idx, threshold, state):

    t0 = time.time()

    func1(M, NA, lamda, n_immersion, n_sample, f_4f, ps_camera, ps_BFP, zstack_file, nfp_text,
          NFP, zrange, device_id, state)
    func2(raw_image_folder, state)
    func3(photon_roi, max_pv, state)
    func4(num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, projection_01, state)
    func5(state)
    func6_1(threshold, test_idx, state)
    func6_2(threshold, state)

    t1 = time.time()

    print(t1-t0)

    return 'One click is done.'

