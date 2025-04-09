
import numpy as np
from numpy import pi
from skimage import io
import csv
import time
import os
import shutil
import pickle
from scipy import ndimage
from datetime import datetime
import torch
import torch.fft as fft
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DS3D.ds3d_utils import ImModel, Sampling, MyDataset, Volume2XYZ, calc_jaccard_rmse, KDE_loss3D
from DS3D.ds3d_utils import LON as Net
from DS3D.training_utils import TorchTrainer

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
# import matplotlib
# matplotlib.use("TkAgg")

# import tifffile

class ImModel_pr(nn.Module):
    def __init__(self, params):
        """
        a scalar model for air or oil objective in microscopy
        """
        super().__init__()
        device = params['device']
        ################### set parameters: unit:um
        # oil objective
        M = params['M']  # magnification
        NA = params['NA']  # NA
        n_immersion = params['n_immersion']  # refractive index of the immersion of the objective
        lamda = params['lamda']  # wavelength
        n_sample = params['n_sample']  # refractive index of the sample
        f_4f = params['f_4f']  # focal length of 4f system
        ps_camera = params['ps_camera']  # pixel size of the camera
        ps_BFP = params['ps_BFP']  # pixel size at back focal plane

        # image
        H, W = params['H'], params['W']  # FOV size
        g_size = 9  # size of the gaussian blur kernel
        g_sigma = params['g_sigma']  # std of the gaussian blur kernel

        ###################

        N = np.floor(f_4f * lamda / (ps_camera * ps_BFP))  # simulation size
        N = int(N + 1 - (N % 2))  # make it odd
        print(f'Simulation size of the imaging model is {N} which must be larger than image size (PSF z-stack and training images)!')

        # pupil/aperture at back focal plane
        d_pupil = 2 * f_4f * NA / np.sqrt(M ** 2 - NA ** 2)  # diameter [um]
        pn_pupil = d_pupil / ps_BFP  # pixel number of the pupil diameter should be smaller than the simulation size N
        if N < pn_pupil:
            raise Exception('Simulation size is smaller than the pupil!')
        # cartesian and polar grid in BFP
        x_phys = np.linspace(-N / 2, N / 2, N) * ps_BFP
        xi, eta = np.meshgrid(x_phys, x_phys)  # cartesian physical coordinates
        r_phys = np.sqrt(xi ** 2 + eta ** 2)
        pupil = (r_phys < d_pupil / 2).astype(np.float32)

        x_ang = np.linspace(-1, 1, N) * (N / pn_pupil) * (NA / n_immersion)  # angular coordinate
        xx_ang, yy_ang = np.meshgrid(x_ang, x_ang)
        r = np.sqrt(
            xx_ang ** 2 + yy_ang ** 2)  # normalized angular coordinates, s.t. r = NA/n_immersion at edge of E field support

        k_immersion = 2 * pi * n_immersion / lamda  # [1/um]
        sin_theta_immersion = r
        circ_NA = (sin_theta_immersion < (NA / n_immersion)).astype(
            np.float32)  # the same as pupil, NA / n_immersion < 1
        cos_theta_immersion = np.sqrt(1 - (sin_theta_immersion * circ_NA) ** 2) * circ_NA

        k_sample = 2 * pi * n_sample / lamda
        sin_theta_sample = n_immersion / n_sample * sin_theta_immersion
        # note: when circ_sample is smaller than circ_NA, super angle fluorescence apears
        circ_sample = (sin_theta_sample < 1).astype(np.float32)  # if all the frequency of the sample can be captured
        cos_theta_sample = np.sqrt(1 - (sin_theta_sample * circ_sample) ** 2) * circ_sample * circ_NA

        # circular aperture to impose on BFP, SAF is excluded
        circ = circ_NA * circ_sample

        pn_circ = np.floor(np.sqrt(np.sum(circ) / pi) * 2)
        pn_circ = int(pn_circ + 1 - (pn_circ % 2))
        Xgrid = 2 * pi * xi * M / (lamda * f_4f)
        Ygrid = 2 * pi * eta * M / (lamda * f_4f)
        Zgrid = k_sample * cos_theta_sample
        NFPgrid = k_immersion * (-1) * cos_theta_immersion  # -1

        self.device = device
        self.Xgrid = torch.from_numpy(Xgrid).to(device)
        self.Ygrid = torch.from_numpy(Ygrid).to(device)
        self.Zgrid = torch.from_numpy(Zgrid).to(device)
        self.NFPgrid = torch.from_numpy(NFPgrid).to(device)
        self.circ = torch.from_numpy(circ).to(device)
        self.circ_NA = torch.from_numpy(circ_NA).to(device)
        self.circ_sample = torch.from_numpy(circ_sample).to(device)
        self.idx05 = int(N / 2)
        self.N = N
        self.pn_pupil = pn_pupil
        self.pn_circ = pn_circ

        # for a blur kernel
        g_r = int(g_size / 2)
        g_xs = torch.linspace(-g_r, g_r, g_size, device=device).type(torch.float64)
        self.g_xx, self.g_yy = torch.meshgrid(g_xs, g_xs, indexing='xy')

        # crop settings
        self.r0, self.c0 = int(np.round((N-H)/2)), int(np.round((N-W)/2))
        # h05, w05 = int(H / 2), int(W / 2)
        # self.h05, self.w05 = h05, w05
        self.H, self.W = H, W

        self.phase_mask = torch.tensor(circ, device=device, requires_grad=True)
        self.g_sigma = torch.tensor(g_sigma, device=device, requires_grad=True)

    def forward(self, xyzps, NFPs):
        phase_lateral = self.Xgrid * (xyzps[:, 0:1].unsqueeze(1)) + self.Ygrid * (xyzps[:, 1:2].unsqueeze(1))
        phase_axial = self.Zgrid * (xyzps[:, 2:3].unsqueeze(1)) + self.NFPgrid * NFPs.unsqueeze(1).unsqueeze(1)
        ef_bfp = self.circ * torch.exp(1j * (phase_axial + phase_lateral + self.phase_mask))
        psf_field = fft.fftshift(fft.fftn(fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))  # FT
        psfs = torch.abs(psf_field) ** 2

        # blur
        blur_kernel = 1 / (2 * pi * self.g_sigma ** 2) * (
            torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / self.g_sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0), padding='same')
        psfs = psfs.squeeze(1)

        # photon normalization
        psfs = psfs / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(1)  # photon normalization
        # psfs = psfs[:, self.idx05 - self.h05:self.idx05 + self.h05 + 1, self.idx05 - self.w05:self.idx05 + self.w05 + 1]
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]

        return psfs


def calculate_cc(output, target):
    # output: rank 3, target: rank 3
    output_mean = np.mean(output, axis=(1, 2), keepdims=True)
    target_mean = np.mean(target, axis=(1, 2), keepdims=True)
    ccs = (np.sum((output - output_mean) * (target - target_mean), axis=(1, 2)) /
           (np.sqrt(np.sum((output - output_mean) ** 2, axis=(1, 2)) * np.sum((target - target_mean) ** 2,
                                                                              axis=(1, 2))) + 1e-9))
    return ccs


def phase_retrieval(param_dict, pr_dict, fig_flag=True):

    device = param_dict['device']

    file_path = pr_dict['zstack_file_path']

    nfps = pr_dict['nfps']
    r_bead = pr_dict['r_bead']
    epoch_num = pr_dict['epoch_num']
    loss_label = pr_dict['loss_label']

    # read the zstack and set the image size for the imaging model

    zstack = io.imread(file_path)  # axis0 -- z position
    # zstack = tifffile.imread(file_path)
    print(f'zstack shape: ({zstack.shape[1]}, {zstack.shape[2]})')

    corner_size = max(7, int(0.1*zstack.shape[1]))

    param_dict['H'], param_dict['W'] = zstack.shape[1], zstack.shape[2]

    # estimate Gaussian noise: mean and std
    patches = np.concatenate(
        (np.concatenate((zstack[:, :corner_size, :corner_size], zstack[:, :corner_size, -corner_size:]), axis=2),
         np.concatenate((zstack[:, -corner_size:, :corner_size], zstack[:, -corner_size:, -corner_size:]), axis=2)),
        axis=1)
    means = np.mean(patches, axis=(1, 2), keepdims=True)
    stds = np.std(patches, axis=(1, 2), keepdims=True)
    # filtering mask
    zstack = zstack - means
    mask = (zstack > stds)
    # erode and dimate the mask
    struct = ndimage.generate_binary_structure(2, 1)  # raius 1 or 2
    mask = [ndimage.binary_dilation(ndimage.binary_erosion(mask[i, :, :], struct), struct) for i in range(mask.shape[0])]
    # clean zstack
    zstack = zstack * np.array(mask)
    z_photons = np.sum(zstack, axis=(1, 2))

    # build the imaging model
    model = ImModel_pr(param_dict)

    nfps_tensor = torch.tensor(nfps).to(device)
    xyzps_np = np.zeros((nfps.shape[0], 4))
    xyzps_np[:, 2] = r_bead
    xyzps_np[:, 3] = z_photons
    xyzps_tensor = torch.tensor(xyzps_np, device=device)

    std_tensor = torch.tensor(stds, device=device)  # for gaussian likelihood loss

    psf_target = torch.tensor(zstack).to(device)

    optimizer = torch.optim.Adam([{'params': model.phase_mask, 'lr': 1e-1},
                                  {'params': model.g_sigma, 'lr': 1e-2}
                                  ])
    epoch_loss = []
    for i in range(epoch_num):
        psfs = model(xyzps_tensor, nfps_tensor)
        if loss_label == 1:  # gaussian likelihood
            loss = torch.sum((torch.log(torch.sqrt(2 * pi * (psfs + std_tensor ** 2 + 1e-9))) +
                              0.5 * ((psf_target - psfs) ** 2) / (psfs + std_tensor ** 2)) * (psf_target != 0))
        elif loss_label == 2:  # mse loss
            loss = torch.nn.functional.mse_loss(psfs, psf_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    phase_mask = model.phase_mask.detach().cpu().numpy()
    phase_mask = np.angle(np.exp(1j * phase_mask))
    g_sigma = model.g_sigma.item()

    psfs_np = psfs.detach().cpu().numpy()
    ccs = calculate_cc(psfs_np, zstack)

    if fig_flag:
        fig = plt.figure(1, figsize=(7, 4))
        gs = fig.add_gridspec(3, 7)

        ax = fig.add_subplot(gs[:2, :3])
        maskplot = ax.imshow(phase_mask)
        plt.colorbar(maskplot)
        ax.set_title('retrieved phase')

        ax = fig.add_subplot(gs[0, 3:])
        ids = (0, (zstack.shape[0]-1) // 4, (zstack.shape[0]-1) // 2, ((zstack.shape[0]-1)//4)*3, -1)
        im_demo = np.concatenate((zstack[ids[0]], zstack[ids[1]], zstack[ids[2]], zstack[ids[3]],
                                  zstack[ids[4]]), axis=1)
        ax.imshow(im_demo)
        ax.axis('off')
        ax.set_title('exp')

        ax = fig.add_subplot(gs[1, 3:])
        im_demo = np.concatenate((psfs_np[ids[0]], psfs_np[ids[1]], psfs_np[ids[2]], psfs_np[ids[3]],
                                  psfs_np[ids[4]]), axis=1)
        ax.imshow(im_demo)
        ax.axis('off')
        ax.set_title('model')

        ax = fig.add_subplot(gs[2, :])
        ax.plot(nfps, ccs)
        ax.set_xlabel('NFP [um]')
        ax.set_ylabel('CC')
        # ax.set_title('model accuracy')

        plt.savefig('phase_retrieval_results.jpg', bbox_inches='tight', dpi=300)
        plt.clf()
        print(f'phase retrieval results: phase_retrieval_results.jpg')

    return phase_mask, g_sigma, ccs


def show_z_psf(param_dict):
    model = ImModel(param_dict)
    model.model_demo(np.linspace(param_dict['zrange'][0], param_dict['zrange'][1], 5))  # check PSFs


def background_removal(im_folder, num=100):
    save_folder = im_folder + '_br'  # where to save the images after background removal
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    im_files = sorted(os.listdir(im_folder))  # make sure the names are sortable
    n_ims = len(im_files)
    pointer = 0
    for i in range(n_ims//num):
        im_names = [im_files[pointer+j] for j in range(num)]
        im_stack = [io.imread(os.path.join(im_folder, im_files[pointer+j])) for j in range(num)]
        pointer += num
        im_stack = np.array(im_stack)
        im_stack = im_stack-np.min(im_stack, axis=0)

        for j in range(num):  # save
            io.imsave(os.path.join(save_folder, im_names[j]), im_stack[j], check_contrast=False)

    # remainder of n_ims/num
    im_stack = [io.imread(os.path.join(im_folder, im_files[-j])) for j in range(num)]
    im_stack = np.array(im_stack)
    im_min = np.min(im_stack, axis=0)
    for j in range(pointer, n_ims):
        im = io.imread(os.path.join(im_folder, im_files[j]))
        im = im-im_min
        io.imsave(os.path.join(save_folder, im_files[j]), im, check_contrast=False)

    return save_folder

def mu_std_p(param_dict, noise_dict):

    im_br_folder = param_dict['im_br_folder']
    num = noise_dict['num_ims']
    # noise_roi = noise_dict['noise_roi']
    photon_roi = noise_dict['photon_roi']
    max_pv = noise_dict['max_pv']

    im_names = sorted(os.listdir(im_br_folder))
    im_names = im_names[-num:]  # at the end of the video, probably with sparse molecules
    ims = np.array([io.imread(os.path.join(im_br_folder, im_name)) for im_name in im_names])

    ims = ims[:, photon_roi[0]:photon_roi[2], photon_roi[1]:photon_roi[3]]

    max_map = np.max(ims, axis=0)
    mean_map = np.mean(ims, axis=0)
    # median_map = np.median(ims, axis=0)
    std_map = np.std(ims, axis=0)

    bg_threshold = 5.0
    bg = ims[:, max_map < (mean_map + bg_threshold * std_map)]
    bg_means, bg_stds = np.mean(bg, axis=0), np.std(bg, axis=0)
    mu, std = np.min(bg_means), bg_stds[np.argmin(bg_means)]

    # mu, std = np.mean(bg, axis=0).median(), np.std(bg, axis=0).median()

    # mv_threshold = 6
    # mvs = max_map[max_map > (mean_map + mv_threshold * std_map)]

    # patches = ims[:, noise_roi[0]:noise_roi[2], noise_roi[1]:noise_roi[3]]
    # mus, stds = np.mean(patches, axis=(1, 2)), np.std(patches, axis=(1, 2))

    # mu, std = np.median(mus), np.median(stds)  # choose the median
    if max_pv == 0:
        exp_maxv = np.max(max_map)
    else:
        exp_maxv = max_pv
    print(f'Photon count characterization number MPV: {exp_maxv}.')

    model = ImModel(param_dict)

    # zn = noise_dict['zn']
    # xyzps = np.zeros((zn, 4))
    # xyzps[:, 2] = np.linspace(param_dict['zrange'][0], param_dict['zrange'][1], zn)  # set z values
    # xyzps = torch.from_numpy(xyzps).to(param_dict['device'])
    # xyzps[:, 3] = 1e4
    # ims = np.array([model.get_psfs(xyzps[i:i + 1, :]).cpu().numpy()[0] for i in range(zn)])

    photon_count = 1e4
    xyzps = np.array([[0, 0, (param_dict['zrange'][0]+param_dict['zrange'][1])/2, photon_count]])  # take the middle z
    xyzps = torch.from_numpy(xyzps).to(param_dict['device'])
    ims = model.get_psfs(xyzps).cpu().numpy()

    maxvs = np.max(ims, axis=(1, 2))
    mv = np.mean(maxvs) + mu  # model.get_psfs doesn't include noise yet
    p = photon_count/mv * exp_maxv

    return mu, std, p


def training_data_func(param_dict):

    device = param_dict['device']
    # imaging model
    model = ImModel(param_dict)
    # sampling model
    sampling = Sampling(param_dict)
    sampling.show_volume()  # plot volume

    # start
    td_folder = param_dict['td_folder']
    if os.path.exists(td_folder):  # delete the directory if it exists
        shutil.rmtree(td_folder)
    x_folder = td_folder + '/x'
    os.makedirs(x_folder)  # make the folder for training data

    # labels_dict for training
    labels_dict = {}
    labels_dict['volume_size'] = (param_dict['D'], param_dict['HH'], param_dict['WW'])
    labels_dict['us_factor'] = param_dict['us_factor']
    labels_dict['blob_r'] = sampling.blob_r  # radius of each 3D blob representing an emitter in space
    labels_dict['blob_maxv'] = sampling.blob_maxv  # maximum value of blobs

    ntrain = param_dict['n_ims']
    for i in range(ntrain):
        xyzps, xyz_ids, blob3d = sampling.xyzp_batch()
        # im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.float32)
        im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy()
        if param_dict['project_01']:
            # im = ((im - im.min()) / (im.max() - im.min()))
            im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)

        x_name = str(i).zfill(5) + '.tif'
        io.imsave(os.path.join(x_folder, x_name), im, check_contrast=False)
        labels_dict[x_name] = (xyz_ids, blob3d)

        if i % (ntrain // 10) == 0:
            print('Training image [%d / %d]' % (i + 1, ntrain))
    print('Training image [%d / %d]' % (ntrain, ntrain))

    y_file = os.path.join(td_folder, r'y.pickle')
    with open(y_file, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    param_file = os.path.join(td_folder, r'param.pickle')
    with open(param_file, 'wb') as handle:
        pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def training_func(param_dict, training_dict):
    np.random.seed(66)
    torch.manual_seed(88)

    device = param_dict['device']
    torch.backends.cudnn.benchmark = True

    td_folder = param_dict['td_folder']
    path_save = param_dict['path_save']
    if not (os.path.isdir(path_save)):
        os.mkdir(path_save)

    batch_size = training_dict['batch_size']
    lr = training_dict['lr']
    num_epochs = training_dict['num_epochs']

    params_train = {'batch_size': batch_size, 'shuffle': True}
    params_validate = {'batch_size': batch_size, 'shuffle': True}

    x_folder = os.path.join(td_folder, 'x')
    x_list = os.listdir(x_folder)
    num_x = len(x_list)
    with open(os.path.join(td_folder, 'y.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    partition = {'train': x_list[:int(num_x * 0.9)], 'validate': x_list[int(num_x * 0.9):]}
    train_ds = MyDataset(x_folder, partition['train'], labels)
    train_dl = DataLoader(train_ds, **params_train)
    validate_ds = MyDataset(x_folder, partition['validate'], labels)
    validate_dl = DataLoader(validate_ds, **params_validate)

    D, us_factor, maxv = labels['volume_size'][0], labels['us_factor'], labels['blob_maxv']
    model = Net(D=D, us_factor=us_factor, maxv=maxv).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'# of trainable parameters: {n_params}')

    optimizer = Adam(list(model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True,
                                  min_lr=1e-6)  # verbose True
    if param_dict['us_factor']==1:
        my_loss_func = torch.nn.MSELoss()
    else:
        my_loss_func = KDE_loss3D(sigma=0.5*(param_dict['us_factor']/2), device=device)  # 0.5-2, 1.0-4

    trainer = TorchTrainer(model, my_loss_func, optimizer, lr_scheduler=scheduler, device=device)

    time_now = datetime.today().strftime('%m-%d_%H-%M')
    net_file = 'net_' + time_now + '.pt'
    checkpoints = dict(file_name=os.path.join(path_save, net_file),
                       net=Net(D=D, us_factor=us_factor, maxv=maxv),
                       state_dict=None,
                       note=' '
                       )

    t0 = time.time()
    fit_results = trainer.fit(train_dl, validate_dl, num_epochs=num_epochs, checkpoints=checkpoints, early_stopping=4)
    fit_file = 'fit_' + time_now + '.pickle'
    with open(os.path.join(path_save, fit_file), 'wb') as handle:
        pickle.dump(fit_results, handle)

    t1 = time.time()
    print(f'training results in {net_file} and {fit_file}')
    print(f'finished training in {t1 - t0}s.')

    return net_file, fit_file


def inference_func1(param_dict, test_idx, fig_flag=True):  # simulation and try one exp image

    np.random.seed(11)
    torch.manual_seed(11)

    device = param_dict['device']
    path_save = param_dict['path_save']
    net_file = param_dict['net_file']
    fit_file = param_dict['fit_file']

    # training performance
    with open(os.path.join(path_save, fit_file), 'rb') as handle:
        fit_result = pickle.load(handle)
    train_loss = fit_result.train_loss
    test_loss = fit_result.test_loss

    if fig_flag:
        num_epochs = len(train_loss)
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(num_epochs), train_loss)
        plt.plot(np.arange(num_epochs), test_loss)
        plt.title('training loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.savefig('loss_curves.jpg', bbox_inches='tight', dpi=300)
        plt.clf()

        print('Training loss curves: loss_curves.jpg')

    checkpoint = torch.load(os.path.join(path_save, net_file), map_location=device)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])

    # simulated data
    model = ImModel(param_dict)
    sampling = Sampling(param_dict)
    volume2xyz = Volume2XYZ(param_dict)

    # simulation
    xyzps, _, _ = sampling.xyzp_batch()
    im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.float32)
    if param_dict['project_01']:
        im = ((im - im.min()) / (im.max() - im.min()))

    with torch.no_grad():
        net.eval()
        vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
    xyz_rec, conf_rec = volume2xyz(vol)

    if xyz_rec is not None:
        xyz_gt = xyzps[:, :-1]
        jaccard_index, RMSE_xy, RMSE_z, _ = calc_jaccard_rmse(xyz_gt, xyz_rec, 0.1)
        jaccard_index, RMSE_xy, RMSE_z = np.round(jaccard_index, decimals=2), np.round(RMSE_xy*1000, decimals=2), np.round(RMSE_z*1000, decimals=2)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], c='b', marker='o', label='GT', depthshade=False)
        ax.scatter(xyz_rec[:, 0], xyz_rec[:, 1], xyz_rec[:, 2], c='r', marker='^', label='Rec', depthshade=False)
        ax.set_xlabel('X [um]')
        ax.set_ylabel('Y [um]')
        ax.set_zlabel('Z [um]')
        if RMSE_xy is not None:
            plt.title(f'Found {xyz_rec.shape[0]} / {xyz_gt.shape[0]}, j_idx: {jaccard_index}, r_xy: {RMSE_xy} nm, r_z: {RMSE_z} nm')
        else:
            plt.title(f'Found {xyz_rec.shape[0]} emitters out of {xyz_gt.shape[0]}')
        plt.legend()
        plt.savefig('sim_loc_gt_rec.jpg', dpi=300)
        plt.clf()

        nphotons_rec = 1e4 * np.ones(xyz_rec.shape[0])
        psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy()
        im_rec = np.sum(psfs_rec, axis=0)
        im_rec = (im_rec-im_rec.min())/(im_rec.max()-im_rec.min())
        im = (im-im.min())/(im.max()-im.min())

        fig = plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(im, cmap='gray')
        plt.title('im')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(im_rec, cmap='gray')
        plt.title('im_rec')
        plt.axis('off')

        mask = np.max(psfs_rec, axis=0)
        mask = (mask-mask.min())/(mask.max()-mask.min())
        mask = 1-mask
        transparency = 0.2+mask*0.8
        im_overlay = np.stack((im, im, im, transparency), axis=-1)
        im_overlay[:, :, 1] = im_overlay[:, :, 1] * mask
        plt.subplot(1, 3, 3)
        plt.imshow(im_overlay)
        plt.title('overlay')
        plt.axis('off')

        plt.savefig('sim_im_gt_rec.jpg', bbox_inches='tight', dpi=300)
        plt.clf()

        print('Network inference on simulated an image: sim_im_gt_rec.jpg')



    exp_imgs_path = param_dict['im_br_folder']
    img_names = sorted(os.listdir(exp_imgs_path))

    im = io.imread(os.path.join(exp_imgs_path, img_names[test_idx])).astype(np.float32)  # read the test image
    if param_dict['project_01']:
        im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
    with torch.no_grad():
        net.eval()
        vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
    xyz_rec, conf_rec = volume2xyz(vol)


    H, W = im.shape
    param_dict['H'], param_dict['W'] = H, W
    model = ImModel(param_dict)

    if H > param_dict['phase_mask'].shape[0] or W > param_dict['phase_mask'].shape[1]:
        sf = max(H // param_dict['phase_mask'].shape[0]+1, W // param_dict['phase_mask'].shape[1]+1)
        param_dict['ps_BFP'] /= sf
        phase_mask = param_dict['phase_mask']

        # H_, W_ = phase_mask.shape[0]*sf, phase_mask.shape[1]*sf
        # H_ = H_ + 1 - H_ % 2
        # W_ = W_ + 1 - W_ % 2

        HW = np.floor(param_dict['f_4f'] * param_dict['lamda'] / (param_dict['ps_camera'] * param_dict['ps_BFP']))  # simulation size
        HW = int(HW + 1 - (HW % 2))  # make it odd

        phase_mask = interpolate(torch.tensor(phase_mask).unsqueeze(0).unsqueeze(1), size=(HW, HW))
        param_dict['phase_mask'] = phase_mask.squeeze(0).squeeze(1).numpy()
        model = ImModel(param_dict)

    nphotons_rec = 1e4 * np.ones(xyz_rec.shape[0])
    psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy()

    im_rec = np.sum(psfs_rec, axis=0)
    im_rec = (im_rec - im_rec.min()) / (im_rec.max() - im_rec.min())

    im = (im - im.min()) / (im.max() - im.min())

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap='gray')
    plt.title('im')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(im_rec, cmap='gray')
    plt.title(f'im_rec, found {xyz_rec.shape[0]} emitters')
    plt.axis('off')

    mask = np.max(psfs_rec, axis=0)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = 1 - mask
    transparency = 0.2 + mask * 0.8

    im_overlay = np.stack((im, im, im, transparency), axis=-1)

    # mask = mask<0.1
    im_overlay[:, :, 1] = im_overlay[:, :, 1] * mask
    plt.subplot(1, 3, 3)
    plt.imshow(im_overlay)
    plt.title('overlay')
    plt.axis('off')


    plt.savefig('exp_im_gt_rec.jpg', bbox_inches='tight', dpi=300)

    print('Network inference on a test experimental image: exp_im_gt_rec.jpg')


def inference_func2(param_dict):

    device = param_dict['device']
    path_save = param_dict['path_save']
    net_file = param_dict['net_file']
    checkpoint = torch.load(os.path.join(path_save, net_file), map_location=device)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])

    # simulated data
    volume2xyz = Volume2XYZ(param_dict)

    exp_imgs_path = param_dict['im_br_folder']
    img_names = sorted(os.listdir(exp_imgs_path))
    num_imgs = len(img_names)

    tall_start = time.time()
    results = np.array(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
    with torch.no_grad():
        net.eval()
        for im_ind, im_name in enumerate(img_names):
            # print current image number
            print('Processing Image [%d/%d]' % (im_ind + 1, num_imgs))

            tfrm_start = time.time()

            im = io.imread(os.path.join(exp_imgs_path, im_name)).astype(np.float32)
            if param_dict['project_01']:
                im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
            vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
            xyz_rec, conf_rec = volume2xyz(vol)

            tfrm_end = time.time() - tfrm_start

            # if this is the first image, get the dimensions and the relevant center for plotting
            H, W = im.shape
            ch, cw = H / 2, W / 2

            # if prediction is empty then set number fo found emitters to 0
            # otherwise generate the frame column and append results for saving
            if xyz_rec is None:
                nemitters = 0
            else:
                nemitters = xyz_rec.shape[0]
                frm_rec = (im_ind + 1) * np.ones(nemitters)
                xnm = (xyz_rec[:, 0] + cw * param_dict['vs_xy']*param_dict['us_factor']) * 1000
                ynm = (xyz_rec[:, 1] + ch * param_dict['vs_xy']*param_dict['us_factor']) * 1000
                znm = (xyz_rec[:, 2] - param_dict['zrange'][0]) * 1000  # make sure they are above 0
                xyz_save = np.c_[xnm, ynm, znm]

                results = np.vstack((results, np.column_stack((frm_rec, xyz_save, conf_rec))))

            print('Single frame complete in {:.6f}s, found {:d} emitters'.format(tfrm_end, nemitters))

    # print the time it took for the entire analysis
    tall_end = time.time() - tall_start
    print('=' * 50)
    print('Analysis complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        tall_end // 3600, np.floor((tall_end / 3600 - tall_end // 3600) * 60), tall_end % 60))
    print('=' * 50)

    # write the results to a csv file named "localizations.csv" under the exp img folder
    row_list = results.tolist()
    curr_dt = datetime.now()
    curr_date, curr_time = f'{curr_dt.month}_{curr_dt.day}', f'{curr_dt.hour}_{curr_dt.minute}'
    file_name = os.path.join(os.getcwd(), f'localizations' + '_' + curr_date + '_' + curr_time + '.csv')
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)

    print(f'Localization list: {file_name}')

    return file_name





