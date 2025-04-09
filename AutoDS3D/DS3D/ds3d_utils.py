import numpy as np
from math import pi
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.fft as fft
import torch.nn.functional as F
import os
from skimage import io
from torch.utils.data import Dataset
from torch.nn import MaxPool3d, ConstantPad3d
from torch.nn.functional import conv3d, interpolate
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment


class ImModel(nn.Module):
    def __init__(self, params):
        """
        a scalar model for air or oil objective in microscopy
        """
        super().__init__()

        ################### set parameters: unit:um
        device = params['device']
        # oil objective
        M = params['M']  # magnification
        NA = params['NA']  # NA
        n_immersion = params['n_immersion']  # refractive index of the immersion of the objective
        lamda = params['lamda']  # wavelength
        n_sample = params['n_sample']  # refractive index of the sample
        f_4f = params['f_4f']  # focal length of 4f system
        ps_camera = params['ps_camera']  # pixel size of the camera
        ps_BFP = params['ps_BFP']  # pixel size at back focal plane
        NFP = params['NFP']  # location of the nominal focal plane

        # mask at BFP
        phase_mask = params['phase_mask']

        # image
        H, W = params['H'], params['W']  # FOV size
        g_size = 9  # size of the gaussian blur kernel
        g_sigma = params['g_sigma']  # std of the gaussian blur kernel
        bg = params['bg']  # photon counts of background noise
        baseline = params['baseline']  # cannot be really certain, so should be a range
        read_std = params['read_std']  # standard deviation of readout noise
        ###################

        N = np.floor(f_4f * lamda / (ps_camera * ps_BFP))  # simulation size
        N = int(N + 1 - (N % 2))  # make it odd0

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
        r = np.sqrt(xx_ang ** 2 + yy_ang ** 2)  # normalized angular coordinates, s.t. r = NA/n_immersion at edge of E field support

        k_immersion = 2 * pi * n_immersion / lamda  # [1/um]
        sin_theta_immersion = r
        circ_NA = (sin_theta_immersion < (NA / n_immersion)).astype(np.float32)  # the same as pupil, NA / n_immersion < 1
        cos_theta_immersion = np.sqrt(1 - (sin_theta_immersion * circ_NA) ** 2) * circ_NA

        k_sample = 2 * pi * n_sample / lamda
        sin_theta_sample = n_immersion / n_sample * sin_theta_immersion
        # note: when circ_sample is smaller than circ_NA, super angle fluorescence apears
        circ_sample = (sin_theta_sample < 1).astype(np.float32)  # if all the frequency of the sample can be captured
        cos_theta_sample = np.sqrt(1 - (sin_theta_sample * circ_sample) ** 2) * circ_sample * circ_NA

        # circular aperture to impose on BFP, SAF is excluded
        circ = circ_NA * circ_sample

        pn_circ = np.floor(np.sqrt(np.sum(circ)/pi)*2)
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
        self.phase_NFP = self.NFPgrid * NFP
        if phase_mask is not None:
            self.phase_mask = torch.from_numpy(phase_mask).to(device)
        else:
            self.phase_mask = torch.from_numpy(circ).to(device)
        self.pn_pupil = pn_pupil
        self.pn_circ = pn_circ

        # build a blur kernel
        g_r = int(g_size / 2)
        g_xs = np.linspace(-g_r, g_r, g_size)
        g_xx, g_yy = np.meshgrid(g_xs, g_xs)
        self.g_xx, self.g_yy = torch.from_numpy(g_xx).to(device), torch.from_numpy(g_yy).to(device)
        self.g_sigma = g_sigma
        # crop settings
        # h05, w05 = int(H / 2), int(W / 2)
        # self.h05, self.w05 = h05, w05
        self.r0, self.c0 = int(np.round((N - H) / 2)), int(np.round((N - W) / 2))
        self.H, self.W = H, W

        # noise settings, background, shot, and readout
        self.bg = bg
        self.baseline = baseline
        self.read_std = read_std
        # image bitdepth
        self.bitdepth = 16

    def get_psfs(self, xyzps):  # each batch can only have the same number of particles
        phase_lateral = self.Xgrid * (xyzps[:, 0:1].unsqueeze(1)) + self.Ygrid * (xyzps[:, 1:2].unsqueeze(1))
        phase_axial = self.Zgrid * (xyzps[:, 2:3].unsqueeze(1)) + self.phase_NFP
        ef_bfp = self.circ * torch.exp(1j * (phase_axial + phase_lateral + self.phase_mask))
        psf_field = fft.fftshift(fft.fftn(fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))  # FT
        psfs = torch.abs(psf_field) ** 2
        # blur
        sigma = self.g_sigma[0]+torch.rand(1).to(self.device)*(self.g_sigma[1]-self.g_sigma[0])
        blur_kernel = 1/(2*pi*sigma ** 2)*(torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0), padding='same')
        psfs = psfs.squeeze(1)
        # photon normalization
        psfs = psfs / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(1)  # photon normalization
        # psfs = psfs[:, self.idx05 - self.h05:self.idx05 + self.h05 + 1, self.idx05 - self.w05:self.idx05 + self.w05 + 1]
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]

        return psfs

    def forward(self, xyzps):
        """
        image of point sources
        :param xyzps: spatial locations and photon counts, tensor, rank 2 [n 4]
        :return: tensor, image
        """
        psfs = self.get_psfs(xyzps)

        im = torch.sum(psfs, dim=0)

        # noise: background, shot, readout
        im = torch.poisson(im + self.bg)  # rounded

        read_baseline = self.baseline[0]+torch.rand(1).to(device=self.device)*(self.baseline[1]-self.baseline[0])
        read_std = self.read_std[0]+torch.rand(1).to(device=self.device)*(self.read_std[1]-self.read_std[0])
        im = im + torch.round(read_baseline + torch.randn(im.shape, device=self.device) * read_std)

        im[im < 0] = 0
        max_adu = 2**self.bitdepth - 1
        im[im > max_adu] = max_adu
        im = im.type(torch.int16)

        return im

    def show_circs(self):
        """
        plot several windows/circles in BFP
        :return: plot the windows
        """
        plt.figure(figsize=(4, 3))
        plt.plot(self.circ_NA.cpu().numpy()[self.idx05, :] + 0.5)
        plt.plot(self.circ_sample.cpu().numpy()[self.idx05, :] + 0.25)
        plt.plot(self.circ.cpu().numpy()[self.idx05, :])
        plt.plot(self.phase_mask.cpu().numpy()[self.idx05, :])
        plt.legend(['immersion', 'sample', 'aper', 'mask'])
        plt.title('circles in BFP')
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        plt.show()

    def model_demo(self, zs):
        xyzps = np.c_[np.zeros(zs.shape[0]), np.zeros(zs.shape[0]), zs, np.ones(zs.shape[0])*1e4]
        zstack = self.get_psfs(torch.from_numpy(xyzps).to(self.device)).cpu()
        plt.figure(figsize=(6, 2))
        plt.imshow(torch.cat([zstack[i] for i in range(5)], dim=1))
        plt.title(f'z positions [um]: {zs}')
        plt.axis('off')
        plt.savefig('PSFs.jpg', bbox_inches='tight', dpi=300)
        plt.clf()
        print('imaging model: PSFs.jpg')


class Sampling():
    def __init__(self, params):
        # define the reconstruction domain
        self.D = params['D']  # voxel number in z
        self.HH = params['HH']  # voxel number in y
        self.WW = params['WW']  # voxel number in x
        self.buffer_HH = params['buffer_HH']  # buffer in y, place Gaussian blobs and avoid PSF cropping
        self.buffer_WW = params['buffer_WW']  # buffer in x, place Gaussian blobs and avoid PSF cropping
        self.vs_xy, self.vs_z = params['vs_xy'], params['vs_z']
        self.zrange = params['zrange']

        self.Nsig_range = params['Nsig_range']  # photon count range
        self.num_particles_range = params['num_particles_range']  # emitter count range
        self.blob_maxv = params['blob_maxv']
        self.blob_r = params['blob_r']

        # define Gaussian blobs
        self.sigma = params['blob_sigma']
        pn = self.blob_r*2+1  # the number of pixels of the Gaussian blob
        xs = np.linspace(-self.blob_r, self.blob_r, pn)
        self.zz, self.yy, self.xx = np.meshgrid(xs, xs, xs, indexing='ij')
        self.normal_factor1 = 1 / (np.sqrt(2 * pi * self.sigma ** 2)) ** 3
        self.normal_factor2 = self.blob_maxv / self.Nsig_range[1]

    def xyzp_batch(self):  # one batch

        num_particles = np.random.randint(self.num_particles_range[0], self.num_particles_range[1]+1)

        # integers at center of voxels, starting from 0
        x_ids = np.random.randint(self.buffer_WW, self.WW - self.buffer_WW, num_particles)
        y_ids = np.random.randint(self.buffer_HH, self.HH - self.buffer_HH, num_particles)
        z_ids = np.random.randint(0, self.D, num_particles)
        xyz_ids = np.c_[x_ids, y_ids, z_ids]  # where to place 3D Gaussian blobs

        x_local = np.random.uniform(-0.49, 0.49, num_particles)
        y_local = np.random.uniform(-0.49, 0.49, num_particles)
        z_local = np.random.uniform(-0.49, 0.49, num_particles)
        xyz_local = np.c_[x_local, y_local, z_local]

        xyz = xyz_ids+xyz_local  # voxel

        xyz[:, 0] = (xyz[:, 0] - (self.WW-1) / 2) * self.vs_xy
        xyz[:, 1] = (xyz[:, 1] - (self.HH-1) / 2) * self.vs_xy
        xyz[:, 2] = (xyz[:, 2]+0.5) * self.vs_z + self.zrange[0]

        Nphotons = np.random.randint(self.Nsig_range[0], self.Nsig_range[1], num_particles)
        xyzps = np.c_[xyz, Nphotons]

        blob3d = np.exp(-0.5 * ((self.xx - xyz_local[:, 0, np.newaxis, np.newaxis, np.newaxis]) ** 2 +
                                (self.yy - xyz_local[:, 1, np.newaxis, np.newaxis, np.newaxis]) ** 2 +
                                (self.zz - xyz_local[:, 2, np.newaxis, np.newaxis, np.newaxis]) ** 2) / self.sigma ** 2)

        # blob3d = blob3d * self.normal_factor1 * xyzps[:, 3][:, np.newaxis, np.newaxis, np.newaxis]
        blob3d = blob3d * self.normal_factor2 * xyzps[:, 3][:, np.newaxis, np.newaxis, np.newaxis]

        return xyzps, xyz_ids, blob3d


    def show_volume(self, ):
        _, xyz_ids, blob3d = self.xyzp_batch()
        y = np.zeros((self.D, self.HH, self.WW))
        # assemble the representation of emitters
        y = np.pad(y, self.blob_r)
        for i in range(xyz_ids.shape[0]):
            xidx, yidx, zidx = xyz_ids[i, 0], xyz_ids[i, 1], xyz_ids[i, 2]
            y[zidx:zidx + 2 * self.blob_r + 1, yidx:yidx + 2 * self.blob_r + 1, xidx:xidx + 2 * self.blob_r + 1] += blob3d[i]
        y = y[self.blob_r:-self.blob_r, self.blob_r:-self.blob_r, self.blob_r:-self.blob_r]

        xy_proj = np.max(y, axis=0)
        xz_proj = np.max(y, axis=1)
        # yz_proj = np.max(y, axis=2)

        plt.figure(figsize=(4, 6))
        plt.subplot(2, 1, 1)
        plt.imshow(xy_proj)
        plt.title('xy max projection')

        plt.subplot(2, 1, 2)
        plt.imshow(xz_proj)
        plt.title('xz max projection')

        # plt.show()
        plt.savefig('volume_projection.jpg', bbox_inches='tight', dpi=300)
        plt.clf()
        print('Volume (network output) example: volume_projection.jpg')

class Volume2XYZ(nn.Module):
    def __init__(self, params):
        super().__init__()
        # define the reconstruction volume
        self.blob_r = params['blob_r']  # buffer in z, place Gaussian blobs, radius of 3D gaussian blobs
        self.ps_xy = params['vs_xy']
        self.ps_z = params['vs_z']
        self.zrange = params['zrange']
        self.threshold = params['threshold']
        self.device = params['device']

        self.r = self.blob_r  # radius of the blob
        self.maxpool = MaxPool3d(kernel_size=2 * self.r + 1, stride=1, padding=self.r)
        self.pad = ConstantPad3d(self.r, 0.0)
        self.zero = torch.FloatTensor([0.0]).to(self.device)

        # construct the local average filters
        filt_vec = np.arange(-self.r, self.r + 1)
        yfilter, zfilter, xfilter = np.meshgrid(filt_vec, filt_vec, filt_vec)
        xfilter = torch.FloatTensor(xfilter).unsqueeze(0).unsqueeze(0)
        yfilter = torch.FloatTensor(yfilter).unsqueeze(0).unsqueeze(0)
        zfilter = torch.FloatTensor(zfilter).unsqueeze(0).unsqueeze(0)
        sfilter = torch.ones_like(xfilter)
        self.local_filter = torch.cat((sfilter, xfilter, yfilter, zfilter), 0).to(self.device)

        # blob catch
        offsets = torch.arange(0, self.r * 2 + 1, device=self.device)
        grid_z, grid_y, grid_x = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
        self.grid_z = grid_z.flatten()
        self.grid_y = grid_y.flatten()
        self.grid_x = grid_x.flatten()

    def local_avg(self, xbool, ybool, zbool, pred_vol_pad):
        num_pts = len(zbool)
        all_z = zbool.unsqueeze(1) + self.grid_z
        all_y = ybool.unsqueeze(1) + self.grid_y
        all_x = xbool.unsqueeze(1) + self.grid_x
        pred_vol_all_ = pred_vol_pad[0][all_z, all_y, all_x].view(num_pts, self.r*2+1, self.r*2+1, self.r*2+1)

        conf_rec = torch.sum(pred_vol_all_, dim=(1, 2, 3))   # sum of the 3D sub-volume

        pred_vol_all = pred_vol_all_.unsqueeze(1)
        # convolve it using conv3d
        sums = conv3d(pred_vol_all, self.local_filter)
        # squeeze the sums and convert them to local perturbations
        xloc = torch.squeeze(sums[:, 1] / sums[:, 0])
        yloc = torch.squeeze(sums[:, 2] / sums[:, 0])
        zloc = torch.squeeze(sums[:, 3] / sums[:, 0])
        return xloc, yloc, zloc, conf_rec

    def forward(self, pred_vol):
        # threshold
        pred_thresh = torch.where(pred_vol > self.threshold, pred_vol, self.zero)

        # apply the 3D maxpooling to find local maxima
        conf_vol = self.maxpool(pred_thresh)
        conf_vol = torch.where((conf_vol > self.zero) & (conf_vol == pred_thresh), conf_vol, self.zero)  # ~0.001s
        conf_vol = torch.squeeze(conf_vol)
        batch_indices = torch.nonzero(conf_vol, as_tuple=True)  # ~0.006s
        zbool, ybool, xbool = batch_indices[0], batch_indices[1], batch_indices[2]

        # if the prediction is empty return None otherwise convert to list of locations
        if len(zbool) == 0:
            xyz_rec = None
            conf_rec = None
        else:
            # pad the result with radius_px 0's for average calc.
            pred_vol_pad = self.pad(pred_vol)
            # for each point calculate local weighted average
            xloc, yloc, zloc, conf_rec_sum = self.local_avg(xbool, ybool, zbool, pred_vol_pad)  # ~0.001

            D, H, W = conf_vol.size()
            # calculate the recovered positions assuming mid-voxel
            xrec = (xbool + xloc - ((W-1) / 2)) * self.ps_xy
            yrec = (ybool + yloc - ((H-1) / 2)) * self.ps_xy
            zrec = (zbool + zloc + 0.5) * self.ps_z + self.zrange[0]
            xyz_rec = torch.stack((xrec, yrec, zrec), dim=1).cpu().numpy()

            conf_rec = conf_vol[zbool, ybool, xbool]  # use the peak
            conf_rec = conf_rec.cpu().numpy()  # conf_rec is the sum of each 3D blob

        return xyz_rec, conf_rec


class MyDataset(Dataset):
    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels):
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels

        self.r = labels['blob_r']
        self.maxv = labels['blob_maxv']
        self.volume_size = labels['volume_size']

    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)

    # sampling one example from the data
    def __getitem__(self, index):
        # select sample
        ID = self.list_IDs[index]
        # load tiff image
        x_file = os.path.join(self.root_dir, ID)
        x = io.imread(x_file)

        x = x[np.newaxis, :, :].astype(np.float32)

        y = np.zeros(self.volume_size)
        y = np.pad(y, self.r)
        xyz_ids, blob3d = self.labels[ID][0], self.labels[ID][1]
        for i in range(xyz_ids.shape[0]):
            xidx, yidx, zidx = xyz_ids[i, 0], xyz_ids[i, 1], xyz_ids[i, 2]
            y[zidx:zidx + 2 * self.r + 1, yidx:yidx + 2 * self.r + 1, xidx:xidx + 2 * self.r + 1] += blob3d[i]
        y = (y[self.r:-self.r, self.r:-self.r, self.r:-self.r]).astype(np.float32)

        return x, y


class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out


class LON(nn.Module):
    def __init__(self, D, us_factor, maxv):
        super(LON, self).__init__()
        C = 64
        self.us_factor = us_factor
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, C, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(C + 1, C, 3, 1, 1, 0.2)
        self.layer3 = Conv2DLeakyReLUBN(C + 1, C, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = Conv2DLeakyReLUBN(C + 1, C, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = Conv2DLeakyReLUBN(C + 1, C, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = Conv2DLeakyReLUBN(C + 1, C, 3, (16, 16), (16, 16), 0.2)
        self.deconv1 = Conv2DLeakyReLUBN(C + 1, C, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(C, C, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(C, D, 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(D, D, 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(D, D, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(D, D, kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=maxv)


    def forward(self, im):
        # extract multi-scale features
        im = self.norm(im)
        out = self.layer1(im)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out
        features = torch.cat((out, im), 1)

        if self.us_factor == 1:
            out = self.deconv1(features)
            out = self.deconv2(out)
        elif self.us_factor == 2:
            out = interpolate(features, scale_factor=2)
            out = self.deconv1(out)
            out = self.deconv2(out)
        elif self.us_factor == 4:
            out = interpolate(features, scale_factor=2)
            out = self.deconv1(out)
            out = interpolate(out, scale_factor=2)
            out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


def GaussianKernel(shape=(7, 7, 7), sigma=1.0, normfactor=1):
    """
    3D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma]) in 3D
    """
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    y, x, z = np.ogrid[-m:m + 1, -n:n + 1, -p:p + 1]
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma ** 2))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    """
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
        h = h * normfactor
    """
    maxh = h.max()
    if maxh != 0:
        h /= maxh
        h = h * normfactor

    h = torch.from_numpy(h).type(torch.float32)
    h = h.unsqueeze(0)
    h = h.unsqueeze(1)

    return h



class KDE_loss3D(nn.Module):
    def __init__(self, sigma, device):
        super(KDE_loss3D, self).__init__()
        self.kernel = GaussianKernel(sigma=sigma).to(device)

    def forward(self, pred_bol, target_bol):
        # extract kernel dimensions
        _, _, D, _, _ = self.kernel.size()

        # extend prediction and target to have a single channel
        target_bol = target_bol.unsqueeze(1)
        pred_bol = pred_bol.unsqueeze(1)

        # KDE for both input and ground truth spikes
        Din = F.conv3d(pred_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
        Dtar = F.conv3d(target_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))

        # kde loss
        kde_loss = nn.functional.mse_loss(Din, Dtar)

        # final loss
        final_loss = kde_loss

        return final_loss


def calc_jaccard_rmse(xyz_gt, xyz_rec, radius):
    # if the net didn't detect anything return None's
    if xyz_rec is None:
        print("Empty Prediction!")
        return 0.0, None, None, None

    else:

        # calculate the distance matrix for each GT to each prediction
        C = pairwise_distances(xyz_rec, xyz_gt, 'euclidean')

        # number of recovered points and GT sources
        num_rec = xyz_rec.shape[0]
        num_gt = xyz_gt.shape[0]

        # find the matching using the Hungarian algorithm
        rec_ind, gt_ind = linear_sum_assignment(C)

        # number of matched points
        num_matches = len(rec_ind)

        # run over matched points and filter points radius away from GT
        indicatorTP = [False] * num_matches
        for i in range(num_matches):

            # if the point is closer than radius then TP else it's FP
            if C[rec_ind[i], gt_ind[i]] < radius:
                indicatorTP[i] = True

        # resulting TP count
        TP = sum(indicatorTP)

        # resulting jaccard index
        jaccard_index = TP / (num_rec + num_gt - TP)

        # if there's TP
        if TP:

            # pairs of TP
            rec_ind_TP = (rec_ind[indicatorTP]).tolist()
            gt_ind_TP = (gt_ind[indicatorTP]).tolist()
            xyz_rec_TP = xyz_rec[rec_ind_TP, :]
            xyz_gt_TP = xyz_gt[gt_ind_TP, :]

            # calculate mean RMSE in xy, z, and xyz
            RMSE_xy = np.sqrt(np.mean(np.sum((xyz_rec_TP[:, :2] - xyz_gt_TP[:, :2]) ** 2, 1)))
            RMSE_z = np.sqrt(np.mean(np.sum((xyz_rec_TP[:, 2:] - xyz_gt_TP[:, 2:]) ** 2, 1)))
            RMSE_xyz = np.sqrt(np.mean(np.sum((xyz_rec_TP - xyz_gt_TP) ** 2, 1)))

            return jaccard_index, RMSE_xy, RMSE_z, RMSE_xyz
        else:
            return jaccard_index, None, None, None