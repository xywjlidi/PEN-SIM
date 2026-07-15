import torch
import os

import cv2
import numpy as np
import scipy.io as scio
import torch.nn.functional as F
import matplotlib.pyplot as plt

#Bayesian Loss
def VSR_loss(x, y):
    mean = x[:, 0:1, :, :]
    std = x[:, 1:2, :, :]
    loss = torch.div(torch.abs(y - mean), std + 1e-6) + torch.log(std + 1e-6)
    l1loss = F.l1_loss(mean, y)
    return loss, l1loss

def L1Norm(x):
    # B,C,H,W = x.shape
    # x = normalize(x, 1, 0)
    regular = torch.mean(torch.abs(x))
    return regular


def Logsum(x):
    # B,C,H,W = x.shape
    # x = normalize(x, 1, 0)
    regular = torch.log(torch.sum(torch.abs(x)))
    return regular


def Conv_OTF_MSE(x, y, OTF):
    x_conv_otf = torch.real(torch.fft.ifft2(torch.fft.ifftshift((OTF * torch.fft.fftshift(torch.fft.fft2(x))))))
    y_conv_otf = torch.real(torch.fft.ifft2(torch.fft.ifftshift((OTF * torch.fft.fftshift(torch.fft.fft2(y))))))
    return CharbonnierLoss(x_conv_otf, y_conv_otf)


def normalize(img, max, min, test):
    # ?minibatch or single
    img_max = torch.max(img)
    img_min = torch.min(img)
    if img_max == img_min:
        if not test:
            print("error")
        return img
    img = (img - img_min) / (img_max - img_min) * (max - min) + min
    return img


def normalize_np(img, max, min):
    # ?minibatch or single
    img_max = np.max(img)
    img_min = np.min(img)
    if img_max == img_min:
        print("error")
        return img
    img = (img - img_min) / (img_max - img_min) * (max - min) + min
    return img


def normalize_zip(img, max, min):
    B, _, _, _ = img.shape
    for i in range(B):
        img[i] = normalize(img[i], max, min)
    return img


def get_gauss_kernel(kernel_size, sigma, k):
    X = np.linspace(-k, k, kernel_size)
    Y = np.linspace(-k, k, kernel_size)
    x, y = np.meshgrid(X, Y)

    kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x - 0) ** 2 + (y - 0) ** 2) / (2 * sigma ** 2))
    kernel = normalize_np(kernel, 1, 0)
    return kernel


def Conv_Kernel_MSE(x, y, Kernel, test=False):
    ##*psf
    _, _, L, _ = Kernel.shape
    x = F.conv2d(x, Kernel, padding=L // 2)
    y = F.conv2d(y, Kernel, padding=L // 2)
    ##1 clamp 2 normalize
    x = normalize(torch.clamp(x, 0), 1, 0, test)
    y = normalize(torch.clamp(y, 0), 1, 0, test)
    # x = normalize_zip(torch.clamp(x, 0), 1, 0)
    # y = normalize_zip(torch.clamp(y, 0), 1, 0)
    return CharbonnierLoss(x, y)


def Conv_Kernel_MSE_3D(x, y, Kernel, test=False):
    ##*psf
    _, _, L, _ = Kernel.shape
    B, C, Z, H, W = x.shape
    x = x.view(B * Z, 1, H, W)
    y = y.view(B * Z, 1, H, W)
    x = F.conv2d(x, Kernel, padding=L // 2)
    y = F.conv2d(y, Kernel, padding=L // 2)
    ##1 clamp 2 normalize
    x = normalize(torch.clamp(x, 0), 1, 0, test)
    y = normalize(torch.clamp(y, 0), 1, 0, test)
    # x = normalize_zip(torch.clamp(x, 0), 1, 0)
    # y = normalize_zip(torch.clamp(y, 0), 1, 0)
    return CharbonnierLoss(x, y)


def CharbonnierLoss(x, y):
    eps = 1e-3
    diff = x - y
    loss = torch.mean(torch.sqrt((diff * diff) + (eps * eps)))
    return loss


def PSNR(re_img, tar_img):
    imdff = torch.clamp(re_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    return ps


def batch_PSNR(restored, target):
    PSNR_list = []
    for im1, im2 in zip(restored, target):
        psnr = PSNR(im1, im2)
        PSNR_list.append(psnr)
    return sum(PSNR_list)


def linear_transform(transform, target, device):
    size = transform.shape
    length = size[0] * size[1] * size[2]
    x = transform.reshape((length, 1))
    y = target.reshape((length, 1))
    X = torch.cat([x, torch.ones(length, 1).to(device)], 1)
    XT = X.permute(1, 0)
    c = torch.solve(XT.mm(y), XT.mm(X))[0]
    re = (c[0] * x + c[1]).reshape(size)
    return re


def batch_PSNR_linear_transform(restored, target, device):
    PSNR_list = []
    for re, t in zip(restored, target):
        if torch.max(re) == torch.min(re):
            print("zero in val restore")
        else:
            re = linear_transform(re, t, device)
        psnr = PSNR(re, t)
        PSNR_list.append(psnr)
    return sum(PSNR_list)

def batch_PSNR_linear_transform_3D(restored, target, device):
    PSNR_list = []
    for re, t in zip(restored, target):
        if torch.max(re) == torch.min(re):
            print("zero in val restore")
        else:
            re = linear_transform(re, t, device)
        psnr = PSNR(re, t)
        PSNR_list.append(psnr)
    return sum(PSNR_list)


def load_tif_img(filepath):
    img = cv2.imread(filepath, -1)
    img = img.astype(np.float32)
    return img


if __name__ == "__main__":
    OTF_path = '../data/OTF/OTF.mat'

    OTF = scio.loadmat(OTF_path)['OTF']
    PSF = scio.loadmat(OTF_path)['PSF']
    L = 25
    H, W = PSF.shape
    start = (H - L) // 2 + 1
    psf = PSF[start:start + L, start:start + L]
    plt.figure()
    plt.suptitle("PSF")
    plt.subplot(2, 2, 1)
    plt.imshow(psf, 'gray')
    # print(psf)

    # x = load_tif_img("../data/train_bead/TrainData_Simul_r100nm_up4_dense10/train/gt/00001.tif")
    # plt.suptitle("beads")
    gauss = get_gauss_kernel(L, 4.9046, L / 2)
    plt.subplot(2, 2, 2)
    plt.imshow(gauss, 'gray')
    plt.show()

    x_conv_otf = torch.from_numpy(x)
    OTF = torch.from_numpy(OTF)
    x_conv_otf = torch.real(
        torch.fft.ifft2(torch.fft.ifftshift((OTF * torch.fft.fftshift(torch.fft.fft2(x_conv_otf))))))
    plt.suptitle("beads_conv_otf")
    plt.subplot(2, 2, 3)
    plt.imshow(x_conv_otf.numpy(), 'gray')

    y = load_tif_img("../data/train_bead/TrainData_Simul_r100nm_up4_dense10/train/input/00001.tif")
    x = torch.from_numpy(x[np.newaxis, np.newaxis, :, :])
    y = torch.from_numpy(y[np.newaxis, np.newaxis, :, :])
    psf = torch.from_numpy(psf[np.newaxis, np.newaxis, :, :])

    x_conv_psf = F.conv2d(x, psf, padding=L // 2)
    plt.suptitle("x_conv_psf")
    plt.subplot(2, 2, 4)
    plt.imshow(x_conv_psf.squeeze().numpy(), 'gray')
    plt.colorbar()

    # mse = Conv_PSF_MSE(x,y,psf)
    # print(mse)

    plt.show()
