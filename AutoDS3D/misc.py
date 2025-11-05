
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
import math
import PIL
import scipy.optimize as opt


def reshape_mask(mask, target_aperture):
    """
    resize phase mask according to a given aperture
    :param mask: centered phase matrix, ndarray, rank 2
    :param target_aperture: target aperture, 0-1 ndarray, rank 2
    :return: resized phase matrix, has the same aperture size as the target_aperture
    """
    Rc, Cc = target_aperture.shape
    RN, CN = mask.shape
    mask_mr = mask[int(np.ceil((RN-1)/2)), :]  # middle row
    mask_mc = mask[:, int(np.ceil((CN-1)/2))]  # middle column
    aper_mr = target_aperture[int(np.ceil((Rc - 1) / 2)), :]
    aper_mc = target_aperture[:, int(np.ceil((Cc - 1) / 2))]
    # for mask
    x1 = next((i for i, x in enumerate(mask_mr) if x), None)  # index of the first nonzero value
    x2 = next((i for i, x in enumerate(np.flipud(mask_mr)) if x), None)  # index of the last nonzero value, flipud!
    x2 = RN - x2 + 1
    y1 = next((i for i, x in enumerate(mask_mc) if x), None)
    y2 = next((i for i, x in enumerate(np.flipud(mask_mc)) if x), None)
    y2 = RN - y2 + 1
    # for aperture
    x1_C = next((i for i, x in enumerate(aper_mr) if x), None)
    x2_C = next((i for i, x in enumerate(np.flipud(aper_mr)) if x), None)
    x2_C = Rc - x2_C + 1
    y1_C = next((i for i, x in enumerate(aper_mc) if x), None)
    y2_C = next((i for i, x in enumerate(np.flipud(aper_mc)) if x), None)
    y2_C = Cc - y2_C + 1

    W = np.floor((x2_C - x1_C) / (x2 - x1) * RN)  # target width
    H = np.floor((y2_C - y1_C) / (y2 - y1) * CN)  # target height
    im = PIL.Image.fromarray(mask)
    mask_res = np.array(im.resize([int(H), int(W)], PIL.Image.BICUBIC))  # resized mask

    # pad or crop to get same shape as aperture
    R, C = mask_res.shape
    if R < Rc or C < Cc:  # padding
        mask = np.zeros((Rc, Cc))
        mask[int(np.round(Rc/2-R/2)):int(np.round(Rc/2-R/2))+R,
        int(np.round((Cc/2-C/2))):int(np.round((Cc/2-C/2)))+C] = mask_res
    elif R > Rc or C > Cc:
        mask = mask_res[int(np.round(R/2-Rc/2)):int(np.round(R/2-Rc/2))+Rc,
               int(np.round(C/2-Cc/2)):int(np.round(C/2-Cc/2))+Cc]

    return np.array(mask)


def show_imset(imset, title=None, rc=None):
    """
    show all the matrices (more than 1) in the imset
    :param imset: a tuple with images(ndarray or tensor)
    :param title: title of the imset
    :param rc: tuple to set row and column number
    :return: show them
    """
    im_num = len(imset)
    if rc is None:
        r = int(np.floor(np.sqrt(im_num)))
        c = int(np.ceil(im_num/r))
    else:
        r, c = rc[0], rc[1]
    plt.figure(1)
    for i in range(im_num):
        plt.subplot(r, c, i + 1)
        plt.imshow(imset[i])
        plt.axis('off')
        # plt.colorbar(shrink=0.5)
    if title is not None:
        plt.gca().set_title(title)
    plt.show()


def circular_aper(N, d_ratio):
    """
    creat a circular aperture/window
    :param N: size(how many pixels) of the aperture
    :param d_ratio: diameter ratio, the maximum is 1
    :return: aperture, ndarray
    """
    xs = np.linspace(-1, 1, N)
    x, y = np.meshgrid(xs, xs, indexing='xy')
    r = np.sqrt(x**2 + y**2)
    aperture = r < (d_ratio)
    aperture = aperture.astype('float64')
    aperture[r == d_ratio] = 0.5
    return aperture


def square_window(N, d_ratio):
    """
    square window/aperture
    :param N: size of the square matrix
    :param d_ratio: ratio of diameter/width to N, the maximum is 1
    :return: window matrix W, ndarray
    """
    xs = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    X, Y = np.abs(X), np.abs(Y)
    W1 = (X < d_ratio).astype('float64')
    W1[X == d_ratio/2] = 0.5
    W2 = (Y < d_ratio).astype('float64')
    W2[Y == d_ratio / 2] = 0.5
    return W1*W2


def square_window2(N, rx, ry, d_ratio):
    """
    square window/aperture at (rx, ry). （0, 0） is the center.
    :param N: size of the square matrix
    :param rx and ry: relative coordinates[-1, 1]
    :param d_ratio: ratio of diameter/width to N
    :return: window matrix W, ndarray
    """
    xs = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    X = X - rx
    Y = Y - ry
    X, Y = np.abs(X), np.abs(Y)
    W1 = (X < d_ratio).astype('float64')
    W2 = (Y < d_ratio).astype('float64')

    return W1*W2


def phase2voltage(phase, mapping_curve):
    """
    transform phase mask to voltage mask, numpy
    :param phase: to be transformed, [-pi, pi]
    :param mapping_curve: calibration curve of a certain wavelength, rank 1
    :return: voltage mask [0, 255]
    """
    phase = phase + pi  # to [0, 2pi]
    r, c = phase.shape[0], phase.shape[1]
    voltage = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            pv = phase[i, j]
            ind = np.argmin(np.abs(mapping_curve-pv))
            voltage[i, j] = ind
    voltage = voltage.astype('uint8')
    return voltage


def vortex_phase(M, N, pn):
    """
    vortex phase generation
    :param M: row #
    :param N: column #
    :param pn: how many periods
    :return: vortex phase, nd array
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, M)
    X, Y = np.meshgrid(x, y, indexing='xy')
    phase = np.angle(X + 1j*Y)
    phase = np.mod(phase, 2*pi/pn) * pn
    return phase


class ZernikeBasis(object):
    """
    https://www.gatinel.com/recherche-formation/wavefront-sensing/zernike-polynomials/
    https://mathworld.wolfram.com/ZernikePolynomial.html
    """
    def __init__(self, N):
        # N: matrix size
        x_lin = np.linspace(-1, 1, N)  # create grid
        [X, Y] = np.meshgrid(x_lin, x_lin)
        self.r = np.sqrt(X**2 + Y**2)  # polar coordinates
        self.theta = np.arctan2(Y, X)
        self.circ = (X ** 2 + Y ** 2) <= 1
        
    def polynomial(self, n, m):
        # n: lower index, order 
        # m: upper index, angular frequency 
        Rmn = 0
        for s in range(int((n-np.abs(m))/2)+1):
            Rmn = Rmn + ((-1) ** s) * (np.math.factorial(n - s)) * (self.r ** (n - 2 * s)) / \
            ((np.math.factorial(s)) * (np.math.factorial(int((n + np.abs(m))/2) - s)) *(np.math.factorial(int((n - np.abs(m))/2) - s)))

        Nnm = np.sqrt(2*(n+1)/(1+int(m==0)))

        if m >= 0:
            M = np.cos(m*self.theta)
        else:
            M = np.sin(np.abs(m)*self.theta)

        Znm = Nnm*Rmn*M*self.circ
        
        return Znm
    
    def polynomials(self, n_max):
        # n_max: maximum order 
        Z = []
        for n in range(n_max+1):
            for m in np.arange(-n, n+1, 2):
                Znm = self.polynomial(n, m)
                Z.append(Znm)

        return np.array(Z)

    def orthonomality_check(self, nm1, nm2):
        # check the orthogality and normality of the basis functions
        # nm1: the first tuple of n and m
        # nm2: the second ruple of n and m
        znm1 = self.polynomial(*nm1)
        znm2 = self.polynomial(*nm2)
        return np.sum(znm1*znm2)/np.sum(self.circ)
    


def Zernike_basis_numpy(aper_size, N, M=5):
    """
    aper_size: pixel number of the aperture
    N: size of the output, larger than the aper_size
    M: maximum order
    return: N N number (-1, 1), count
    order: zero, piston, tip, tilt, defocus, astigatism, astigmatism, coma, coma, trefoil, trefoil,
    spherical,....
    """

    x_lin = np.linspace(-1, 1, aper_size)  # create grid
    [X, Y] = np.meshgrid(x_lin, x_lin)

    r = np.sqrt(X ** 2 + Y ** 2)  # polar coordinates
    phi = np.arctan2(Y, X)

    mask_circ = (X ** 2 + Y ** 2) < 1
    count = 1
    D = np.zeros((N, N, M ** 2))
    for n in np.arange(0, M + 1):
        for m in np.arange(0, n + 1):
            Rmn = 0
            if (np.mod(n - m, 2) == 0):
                for k in np.arange(0, int((n - m) / 2) + 1):
                    Rmn = Rmn + ((-1) ** k) * (math.factorial(n - k)) * (r ** (n - 2 * k)) / \
                          ((math.factorial(k)) * (math.factorial(int((n + m) / 2) - k)) * 
                           (math.factorial(int((n - m)/2) - k)))
                if m == 0:
                    tmp = Rmn * mask_circ
                    pad_val = int(np.round((N - aper_size) / 2))
                    tmp = np.pad(tmp, [[pad_val, N - aper_size - pad_val], [pad_val, N - aper_size - pad_val]])
                    if tmp.shape[0] < D.shape[0]:
                        tmp = np.pad(tmp, [[0, 1], [0, 1]])
                    D[:, :, count] = tmp
                    count = count + 1
                else:
                    pad_val = int(np.round((N - aper_size) / 2))
                    tmp = Rmn * np.cos(m * phi) * mask_circ
                    tmp = np.pad(tmp, [[pad_val, N - aper_size - pad_val], [pad_val, N - aper_size - pad_val]])
                    if tmp.shape[0] < D.shape[0]:
                        tmp = np.pad(tmp, [[0, 1], [0, 1]])
                    D[:, :, count] = tmp
                    count = count + 1

                    tmp = Rmn * np.sin(m * phi) * mask_circ
                    tmp = np.pad(tmp, [[pad_val, N - aper_size - pad_val], [pad_val, N - aper_size - pad_val]])
                    if tmp.shape[0] < D.shape[0]:
                        tmp = np.pad(tmp, [[0, 1], [0, 1]])
                    D[:, :, count] = tmp
                    count = count + 1
    return D


def gaussian(N, sigma):
    """
    gaussian distribution generation
    :param N: size
    :param sigma: standard deviation. Its square is the variance.
    :return: normalized gaussian distribution, 2d ndarray
    """
    xl = np.linspace(-10, 10, N)
    x, y = np.meshgrid(xl, xl, indexing='xy')
    g = np.exp(-(x**2 + y**2)/(2*sigma**2))
    g = (g-np.min(g))/(np.max(g)-np.min(g))
    return g


def fresnel_zone_plate(f, wvl, D, N):
    """
    generate the binary fresnel zone plate according to focal length, wavelength, diameter and size
    reference: http://zoneplate.lbl.gov/theory
    :param f: focal length
    :param wvl: wave length
    :param D: diameter
    :param N: how many pixels are included
    :return: a matrix, ndarray
    """
    xl = np.linspace(-D/2, D/2, N)
    x, y = np.meshgrid(xl, xl, indexing='xy')
    r = np.sqrt(x**2 + y**2)
    fzp = np.zeros_like(r)
    n = 1
    while True:
        rn = np.sqrt(n*wvl*(f+n*wvl/4))
        if rn > D/2:
            break
        else:
            rn_ = np.sqrt((n-1)*wvl*(f+(n-1)*wvl/4))
            fzp[(r>=rn_) & (r<rn)] = 1.
            n = n+2
    return fzp


def twoD_GaussianScaledAmp(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array
    xy: a tuple of x and y meshgrid
    for the fitting function--getFWHM_GaussianFitScaledAmp
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude * np.exp(- (((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2)))
    return g.ravel()


def twoD_GaussianScaledAmp_(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array
    xy: a tuple of x and y meshgrid
    for generating 2D gaussian pattern
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude * np.exp(- (((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2)))
    return g  # return an image


def getFWHM_GaussianFitScaledAmp(img):
    """2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns:
        FWHMs in pixels, a list [xo, yo, sigma_x, sigma_y, amplitude, offset]
    """
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    # Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    initial_guess = (img.shape[1] / 2, img.shape[0] / 2, 10, 10, 1, 0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img, 5)
    img_scaled = np.clip((img - bg) / (img.max() - bg), 0, 1)
    popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, (x, y),
                               img_scaled.ravel(), p0=initial_guess,
                               bounds=((img.shape[1] * 0.4, img.shape[0] * 0.4, 1, 1, 0.5, -0.1),
                                       (img.shape[1] * 0.6, img.shape[0] * 0.6, img.shape[1] / 2, img.shape[0] / 2, 1.5,
                                        0.5)))
    return popt



def mask_resize_reshape(mask, resize_factor, final_shape):
    # resize
    im = PIL.Image.fromarray(mask)
    N_resize = int(np.round(mask.shape[0]*resize_factor))
    mask1 = np.array(im.resize([N_resize, N_resize], PIL.Image.Resampling.BILINEAR))  # resized mask
    if final_shape>=N_resize:
        # zero padding
        before = int((final_shape-N_resize)/2)
        after = final_shape-N_resize-before
        mask2 = np.pad(mask1, (before, after))
    else:
        # crop
        before = int((N_resize-final_shape)/2)
        mask2 = mask1[before:before+final_shape, before:before+final_shape]
        
    return mask2
