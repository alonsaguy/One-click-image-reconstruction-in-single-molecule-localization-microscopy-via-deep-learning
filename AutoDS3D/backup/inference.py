
import numpy as np
import torch
import os
from DS3Dplus.ds3d_utils import ImModelTraining, Sampling, calc_jaccard_rmse, Volume2XYZ
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import pickle
from skimage import io
from torch.nn.functional import interpolate
import time
from datetime import datetime
import csv


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(66)
torch.manual_seed(66)

# prepare the trained model
path_save = os.getcwd() + '/training_results'
# net_file = 'net_06-01_13-59.pt'
# fit_file = 'fit_06-01_13-59.pickle'

# net_file = 'net_06-01_14-42.pt'
# fit_file = 'fit_06-01_14-42.pickle'

net_file = 'net_06-06_15-47.pt'
fit_file = 'fit_06-06_15-47.pickle'

path_param_dict = os.path.join(os.path.join(os.path.join(os.getcwd(), 'training_data'), 'laminB1_us1'), 'param.pickle')
with open(path_param_dict, 'rb') as handle:
    param_dict = pickle.load(handle)

# exp_imgs_path = None
exp_imgs_path = r'.\laminB1_dense_dh_br'
demo_1_exp_image = False

param_dict['device'] = device
param_dict['blob_r'] = 2
param_dict['threshold'] = 40  # 80-40
photons_rec = (param_dict['Nsig_range'][0]+param_dict['Nsig_range'][1])/2

# training curves
with open(os.path.join(path_save, fit_file), 'rb') as handle:
    fit_result = pickle.load(handle)
num_epochs = fit_result.num_epochs
train_loss = fit_result.train_loss
test_loss = fit_result.test_loss
plt.figure(figsize=(5, 2))
plt.plot(np.arange(1, num_epochs+1), train_loss, label='training')
plt.plot(np.arange(1, num_epochs+1), test_loss, label='validation')
plt.legend()
plt.title('loss curves')
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.show()

checkpoint = torch.load(os.path.join(path_save, net_file), map_location=device)
net = checkpoint['net']
net.load_state_dict(checkpoint['state_dict'])

model = ImModelTraining(param_dict)
sampling = Sampling(param_dict)
volume2xyz = Volume2XYZ(param_dict)

if exp_imgs_path is None:  # simulation
    xyzps, _, _ = sampling.xyzp_batch()
    im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.float32)
    if param_dict['project_01']:
        im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
    with torch.no_grad():
        net.eval()
        vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
        xyz_rec, conf_rec = volume2xyz(vol)

    if xyz_rec is not None:
        xyz_gt = xyzps[:, :-1]
        jaccard_index, RMSE_xy, RMSE_z, _ = calc_jaccard_rmse(xyz_gt, xyz_rec, 0.1)   # set the radius
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

        nphotons_rec = 1e4 * np.ones(xyz_rec.shape[0])
        psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy()
        im_rec = np.sum(psfs_rec, axis=0)
        im_rec = (im_rec-im_rec.min())/(im_rec.max()-im_rec.min())
        im = (im-im.min())/(im.max()-im.min())

        ps_xy = param_dict['vs_xy']*param_dict['us_factor']
        h, w = im.shape
        ch, cw = (h - 1) / 2, (w - 1) / 2
        fig = plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(im, cmap='gray')
        plt.plot(xyz_rec[:, 0] / ps_xy + cw, xyz_rec[:, 1] / ps_xy + ch, 'r+')
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

        plt.show()
    else:
        print('xyz_rec is empty.')

else:

    img_names = sorted(os.listdir(exp_imgs_path))
    num_imgs = len(img_names)

    if demo_1_exp_image:
        im = io.imread(os.path.join(exp_imgs_path, img_names[0])).astype(np.float32)
        if param_dict['project_01']:
            im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)

        with torch.no_grad():
            net.eval()
            vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
        tpost_start = time.time()
        xyz_rec, conf_rec = volume2xyz(vol)
        tpost_elapsed = time.time() - tpost_start
        print('Post-processing complete in {:.6f}s'.format(tpost_elapsed))

        tinf_start = time.time()
        with torch.no_grad():
            net.eval()
            vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
        tinf_elapsed = time.time() - tinf_start
        print('Inference complete in {:.6f}s'.format(tinf_elapsed))

        H, W = im.shape
        param_dict['H'], param_dict['W'] = H, W
        model = ImModelTraining(param_dict)

        if H > param_dict['phase_mask'].shape[0] or W > param_dict['phase_mask'].shape[1]:
            sf = max(H // param_dict['phase_mask'].shape[0] + 1, W // param_dict['phase_mask'].shape[1] + 1)
            param_dict['ps_BFP'] /= sf
            phase_mask = param_dict['phase_mask']
            HW = np.floor(param_dict['f_4f'] * param_dict['lamda'] / (
                        param_dict['ps_camera'] * param_dict['ps_BFP']))  # simulation size
            HW = int(HW + 1 - (HW % 2))  # make it odd

            phase_mask = interpolate(torch.tensor(phase_mask).unsqueeze(0).unsqueeze(1), size=(HW, HW))
            param_dict['phase_mask'] = phase_mask[0, 0].numpy()
            model = ImModelTraining(param_dict)


        nphotons_rec = photons_rec * np.ones(xyz_rec.shape[0])
        psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy()

        im_rec = np.sum(psfs_rec, axis=0)
        im_rec = (im_rec - im_rec.min()) / (im_rec.max() - im_rec.min())

        im = (im - im.min()) / (im.max() - im.min())

        ps_xy = param_dict['vs_xy'] * param_dict['us_factor']
        h, w = im.shape
        ch, cw = (h - 1) / 2, (w - 1) / 2

        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(im, cmap='gray')
        plt.plot(xyz_rec[:, 0] / ps_xy + cw, xyz_rec[:, 1] / ps_xy + ch, 'r+')
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

        plt.show()

    else:

        # time the entire dataset analysis
        tall_start = time.time()
        # needed pixel-size for plotting if only few images are in the folder
        pixel_size_FOV = param_dict['vs_xy']*param_dict['us_factor']  # FOV size, pixel size/magnification
        # process all experimental images
        net.eval()
        results = np.array(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
        with torch.no_grad():
            for im_ind, im_name in enumerate(img_names):

                # print current image number
                print('Processing Image [%d/%d]' % (im_ind + 1, num_imgs))

                # time each frame
                tfrm_start = time.time()

                im = io.imread(os.path.join(exp_imgs_path, im_name)).astype(np.float32)
                if param_dict['project_01']:
                    im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
                vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
                xyz_rec, conf_rec = volume2xyz(vol)

                # time it takes to analyze a single frame
                tfrm_end = time.time() - tfrm_start

                # if this is the first image, get the dimensions and the relevant center for plotting
                if im_ind == 0:
                    H, W = im.shape
                    ch, cw = H / 2, W / 2

                # if prediction is empty then set number fo found emitters to 0
                # otherwise generate the frame column and append results for saving
                if xyz_rec is None:
                    nemitters = 0
                else:
                    nemitters = xyz_rec.shape[0]
                    frm_rec = (im_ind + 1) * np.ones(nemitters)

                    xnm = (xyz_rec[:, 0] + cw * pixel_size_FOV) * 1000
                    ynm = (xyz_rec[:, 1] + ch * pixel_size_FOV) * 1000
                    znm = (xyz_rec[:, 2] - param_dict['zrange'][0]) * 1000
                    xyz_save = np.c_[xnm, ynm, znm]

                    results = np.vstack((results, np.column_stack((frm_rec, xyz_save, conf_rec))))

                # visualize the first 10 images regardless of the number of expeimental frames
                visualize_flag = True if im_ind < 10 else False

                # if the number of imgs is small then plot each image in the loop with localizations
                if visualize_flag:

                    # show input image
                    fig100 = plt.figure(100)

                    imfig = plt.imshow(im, cmap='gray')
                    plt.plot(xyz_rec[:, 0] / pixel_size_FOV + cw-0.5, xyz_rec[:, 1] / pixel_size_FOV + ch-0.5, 'r+')
                    plt.title('Single frame complete in {:.2f}s, found {:d} emitters'.format(tfrm_end, nemitters))
                    fig100.colorbar(imfig)

                    plt.draw()
                    plt.pause(0.1)
                    plt.clf()

                    # plt.show()

                else:

                    # print status
                    print('Single frame complete in {:.6f}s, found {:d} emitters'.format(tfrm_end, nemitters))

        # print the time it took for the entire analysis
        tall_end = time.time() - tall_start
        print('=' * 50)
        print('Analysis complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            tall_end // 3600, np.floor((tall_end / 3600 - tall_end // 3600) * 60), tall_end % 60))
        print('=' * 50)

        # write the results to a csv file named "localizations.csv" under the exp img folder
        time_now = datetime.today().strftime('%m-%d_%H-%M')

        file_name = os.path.join(os.getcwd(), 'localizations_' + time_now + '.csv')
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results.tolist())
        print(f'{file_name} is saved.')




