import numpy as np
import os
import torch
from read_utils.read_utils import load_tif_img, is_tif_file, normalize, load_mrc_dataset, is_specify_file, \
    load_3d_mrc_dataset, load_tif_img_origin, load_3d_mrc_dataset_origin, load_mrc_dataset_origin, prctile_norm
import matplotlib.pyplot as plt
import cv2
from utils.dataset_baseline import Dataset_baseline, input3d_guass_blur, input_guass_blur
from utils.dataset_baseline import get_mask_2d as get_mask

## Transform for Augumentation
def randomRotateAndCrop(img, cropsize, random):
    le = cropsize // 2

    # theta = np.pi/2/(2/np.abs(np.random.randn())+1)
    theta = np.pi / 2 * random[0]
    x = [np.cos(theta) * le - np.sin(theta) * le, np.cos(theta) * le - np.sin(theta) * (-le),
         np.cos(theta) * (-le) - np.sin(theta) * le]
    y = [np.sin(theta) * le + np.cos(theta) * le, np.sin(theta) * le + np.cos(theta) * (-le),
         np.sin(theta) * (-le) + np.cos(theta) * le]
    xc = random[1] * (img.shape[0] - 2 * np.max(x)) + np.max(x)
    yc = random[2] * (img.shape[1] - 2 * np.max(x)) + np.max(x)
    x1 = np.float32(x).reshape((3, 1)) + xc
    y1 = np.float32(y).reshape((3, 1)) + yc

    srcTri = np.concatenate((x1, y1), axis=1)
    dstTri = np.float32([[cropsize, cropsize], [cropsize, 0], [0, cropsize]])
    #     print(srcTri)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    re = cv2.warpAffine(img, warp_mat, (cropsize, cropsize))

    if random[3] > 0.5:
        re = re[::-1, :]
    if random[4] > 0.5:
        re = re[:, ::-1]

    # print(flip)
    # print(theta,srcTri)
    return re


## Transform for 3-d image Augumentation
def randomRotateAndCrop_3d(img, cropsize, random):
    nz, nx, ny = img.shape
    re = np.zeros([nz, cropsize, cropsize])
    for i in range(nz):
        re[i] = randomRotateAndCrop(img[i],cropsize, random)

    return re


def RotateAndCrop_3d_by_center(img, cropsize, xc, yc, random):
    nz, nx, ny = img.shape

    re = np.zeros([nz, cropsize, cropsize])
    for i in range(nz):
        re[i] = RotateAndCrop_by_center(img[i], cropsize, xc, yc, random)
        if (np.max(re[i]) == 0):
            print("oops")

    # print(flip)
    # print(theta,srcTri)
    return re


def RotateAndCrop_by_center(img, cropsize, xc, yc, random):
    le = cropsize // 2

    # theta = np.pi/2/(2/np.abs(np.random.randn())+1)
    theta = np.pi / 2 * random[0]
    x = [np.cos(theta) * le - np.sin(theta) * le, np.cos(theta) * le - np.sin(theta) * (-le),
         np.cos(theta) * (-le) - np.sin(theta) * le]
    y = [np.sin(theta) * le + np.cos(theta) * le, np.sin(theta) * le + np.cos(theta) * (-le),
         np.sin(theta) * (-le) + np.cos(theta) * le]
    # xc = random[1] * (nx - 2 * np.max(x)) + np.max(x)
    # yc = random[2] * (ny - 2 * np.max(x)) + np.max(x)
    x1 = np.float32(x).reshape((3, 1)) + xc
    y1 = np.float32(y).reshape((3, 1)) + yc

    srcTri = np.concatenate((y1, x1), axis=1)
    dstTri = np.float32([[cropsize, cropsize], [cropsize, 0], [0, cropsize]])
    #     print(srcTri)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)

    re = cv2.warpAffine(img, warp_mat, (cropsize, cropsize))
    if (np.max(re) == 0):
        print("oops")

    if random[1] > 0.5:
        re = re[::-1, :]
    if random[2] > 0.5:
        re = re[:, ::-1]

    # print(flip)
    # print(theta,srcTri)
    return re


## Transform for Validate Denoising
def randomCrop(img, cropsize, random):
    x = round(random[0] * (img.shape[0] - cropsize))
    y = round(random[1] * (img.shape[1] - cropsize))

    re = img[y:y + cropsize, x:x + cropsize]

    if random[2] > 0.5:
        re = re[::-1, :]
    if random[3] > 0.5:
        re = re[:, ::-1]
    #     print(x,y,flip)
    # print(flip)
    # print(theta,srcTri)
    return re


## Transform for 3-d image Validate Denoising
def randomCrop_3d(img, cropsize, random):
    nz, nx, ny = img.shape
    x = round(random[0] * (nx - cropsize))
    y = round(random[1] * (ny - cropsize))

    re = img[:, y:y + cropsize, x:x + cropsize]

    if random[2] > 0.5:
        re = re[:, ::-1, :]
    if random[3] > 0.5:
        re = re[:, :, ::-1]
    #     print(x,y,flip)
    # print(flip)
    # print(theta,srcTri)
    return re


def crop_3d_by_center(img, cropsize, xc, yc, random):
    le = cropsize // 2

    re = img[:, yc - le:yc + le, xc - le:xc + le]

    if random[2] > 0.5:
        re = re[:, ::-1, :]
    if random[3] > 0.5:
        re = re[:, :, ::-1]
    #     print(x,y,flip)
    # print(flip)
    # print(theta,srcTri)
    return re


def crop_by_center(img, cropsize, xc, yc, random):
    le = cropsize // 2

    re = img[yc - le:yc + le, xc - le:xc + le]

    if random[2] > 0.5:
        re = re[::-1, :]
    if random[3] > 0.5:
        re = re[:, ::-1]
    #     print(x,y,flip)
    # print(flip)
    # print(theta,srcTri)
    return re


def randomCrop_noflip(img, cropsize, random):
    x = round(random[0] * (img.shape[0] - cropsize))
    y = round(random[1] * (img.shape[1] - cropsize))

    re = img[y:y + cropsize, x:x + cropsize]

    return re


def randomCrop_noflip_3d(img, cropsize, random):
    nz, nx, ny = img.shape
    x = round(random[0] * (nx - cropsize))
    y = round(random[1] * (ny - cropsize))

    re = img[:, y:y + cropsize, x:x + cropsize]

    return re


def crop_noflip_3d_by_center(img, cropsize, xc, yc, random):
    le = cropsize // 2

    re = img[:, yc - le:yc + le, xc - le:xc + le]

    return re


def crop_noflip_by_center(img, cropsize, xc, yc, random):
    le = cropsize // 2

    re = img[yc - le:yc + le, xc - le:xc + le]

    return re


class Dataset():
    def __init__(self, base_dir, opt, shuffle=True, rotate=False, flip=False):
        self.shuffle = shuffle
        if opt.img_type == '.mrc':
            load_function = load_mrc_dataset
        else:
            load_function = load_tif_img

        self.clean_files = sorted(os.listdir(os.path.join(base_dir, opt.gt_dir)))
        self.noisy_files = sorted(os.listdir(os.path.join(base_dir, opt.input_dir)))

        self.clean_filenames = [os.path.join(base_dir, opt.gt_dir, x) for x in self.clean_files if
                                is_specify_file(x, opt.img_type)]
        self.noisy_filenames = [os.path.join(base_dir, opt.input_dir, x) for x in self.noisy_files if
                                is_specify_file(x, opt.img_type)]

        self.clean_imgs = np.float32([load_function(x, opt.input_img_neg) for x in self.clean_filenames])
        self.noisy_imgs = np.float32([load_function(x, opt.input_img_neg) for x in self.noisy_filenames])

        self.length = len(self.clean_filenames)

        self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = [self.clean_imgs[x] for x in ilist]
        noisy_imgs = [self.noisy_imgs[x] for x in ilist]

        clean_imgs = torch.from_numpy(np.array(clean_imgs)[:, np.newaxis, :, :])
        noisy_imgs = torch.from_numpy(np.array(noisy_imgs)[:, np.newaxis, :, :])
        return clean_imgs, noisy_imgs

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        return [self.clean_files[x] for x in ilist], [self.noisy_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_normal():
    def __init__(self, base_dir, gt_dir, input_dir, batch_size, opt, shuffle=True):
        self.shuffle = shuffle

        self.clean_files = sorted(os.listdir(os.path.join(base_dir, gt_dir)))
        self.noisy_files = sorted(os.listdir(os.path.join(base_dir, input_dir)))

        self.clean_filenames = [os.path.join(base_dir, gt_dir, x) for x in self.clean_files if is_tif_file(x)]
        self.noisy_filenames = [os.path.join(base_dir, input_dir, x) for x in self.noisy_files if is_tif_file(x)]

        self.clean_imgs = [normalize(load_tif_img(x), opt.normal_max, opt.normal_min) for x in self.clean_filenames]
        self.noisy_imgs = [normalize(load_tif_img(x), opt.normal_max, opt.normal_min) for x in self.noisy_filenames]

        self.length = len(self.clean_filenames)

        self.iters = np.arange(0, self.length)

        self.batch_size = batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = np.array(np.float32([self.clean_imgs[x] for x in ilist]))
        noisy_imgs = np.array(np.float32([self.noisy_imgs[x] for x in ilist]))

        clean_imgs = torch.from_numpy(clean_imgs[:, np.newaxis, :, :])
        noisy_imgs = torch.from_numpy(noisy_imgs[:, np.newaxis, :, :])
        return clean_imgs, noisy_imgs

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        return [self.clean_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_with_label():
    def __init__(self, base_dir, gt_dir, input_dir, label, batch_size, shuffle=True):
        self.shuffle = shuffle

        self.label_num = len(label)

        self.clean_files, self.noisy_files, self.clean_filenames, self.noisy_filenames, self.label = [], [], [], [], []
        for i in range(self.label_num):
            clean_files = sorted(os.listdir(os.path.join(base_dir[i], gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(base_dir[i], input_dir)))
            self.clean_files += clean_files
            self.noisy_files += noisy_files
            self.clean_filenames += [os.path.join(base_dir[i], gt_dir, x) for x in clean_files if is_tif_file(x)]
            self.noisy_filenames += [os.path.join(base_dir[i], input_dir, x) for x in noisy_files if is_tif_file(x)]
            self.label += [label[i] for j in range(len(clean_files))]

        self.length = len(self.clean_filenames)

        self.iters = np.arange(0, self.length)

        self.batch_size = batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        clean_imgs = np.array([np.float32(load_tif_img(self.clean_filenames[x])) for x in ilist])
        noisy_imgs = np.array([np.float32(load_tif_img(self.noisy_filenames[x])) for x in ilist])

        clean_imgs = torch.from_numpy(clean_imgs[:, np.newaxis, :, :])
        noisy_imgs = torch.from_numpy(noisy_imgs[:, np.newaxis, :, :])

        label = torch.tensor([self.label[x] for x in ilist])
        return clean_imgs, noisy_imgs, label

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        return [self.clean_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_augment():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True):
        self.shuffle = shuffle
        self.rotate = rotate
        self.label_num = len(opt.label)
        self.augment_size = opt.augment_size
        self.noise_crop = opt.noise_crop
        self.clean_crop = opt.clean_crop

        self.clean_files, self.noisy_files, self.clean_filenames, self.noisy_filenames, self.label = [], [], [], [], []
        for i in range(self.label_num):
            clean_files = sorted(os.listdir(os.path.join(base_dir[i], opt.gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(base_dir[i], opt.input_dir)))
            self.clean_files += clean_files
            self.noisy_files += noisy_files
            self.clean_filenames += [os.path.join(base_dir[i], opt.gt_dir, x) for x in clean_files if is_tif_file(x)]
            self.noisy_filenames += [os.path.join(base_dir[i], opt.input_dir, x) for x in noisy_files if is_tif_file(x)]
            self.label += [opt.label[i] for j in range(len(clean_files))]

        self.length = len(self.clean_filenames) * self.augment_size

        self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        ##resize to file length
        ilist = ilist // self.augment_size
        clean_imgs = [load_tif_img(self.clean_filenames[x]) for x in ilist]
        noisy_imgs = [load_tif_img(self.noisy_filenames[x]) for x in ilist]
        clean_crop_imgs = np.zeros((self.batch_size, self.clean_crop, self.clean_crop), np.float32)
        noisy_crop_imgs = np.zeros((self.batch_size, self.noise_crop, self.noise_crop), np.float32)
        # todo augment
        if self.rotate:
            for j in range(self.batch_size):
                random = np.random.rand(5)
                clean_crop_imgs[j] = randomRotateAndCrop(clean_imgs[j], self.clean_crop, random)
                noisy_crop_imgs[j] = randomRotateAndCrop(noisy_imgs[j], self.noise_crop, random)
        else:
            for j in range(self.batch_size):
                random = np.random.rand(4)
                clean_crop_imgs[j] = randomCrop(clean_imgs[j], self.clean_crop, random)
                noisy_crop_imgs[j] = randomCrop(noisy_imgs[j], self.noise_crop, random)

        clean_imgs = torch.from_numpy(clean_crop_imgs[:, np.newaxis, :, :])
        noisy_imgs = torch.from_numpy(noisy_crop_imgs[:, np.newaxis, :, :])

        label = torch.tensor([self.label[x] for x in ilist])
        return clean_imgs, noisy_imgs, label

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        return [self.clean_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_augment_in_cache():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True, flip=True):
        self.shuffle = shuffle
        if rotate:
            crop = randomRotateAndCrop
        elif flip:
            crop = randomCrop
        else:
            crop = randomCrop_noflip

        if opt.img_type == '.mrc':
            load_function = load_mrc_dataset
        else:
            load_function = load_tif_img
        self.haslabel = opt.haslabel
        self.augment_size = opt.augment_size
        self.noise_crop = opt.noise_crop
        self.clean_crop = opt.clean_crop

        self.clean_files, self.noisy_files, self.clean_filenames, self.noisy_filenames, self.label = [], [], [], [], []
        for i in range(len(base_dir)):
            clean_files = sorted(os.listdir(os.path.join(base_dir[i], opt.gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(base_dir[i], opt.input_dir)))
            self.clean_files += clean_files
            self.noisy_files += noisy_files
            self.clean_filenames += [os.path.join(base_dir[i], opt.gt_dir, x) for x in clean_files if
                                     is_specify_file(x, opt.img_type)]
            self.noisy_filenames += [os.path.join(base_dir[i], opt.input_dir, x) for x in noisy_files if
                                     is_specify_file(x, opt.img_type)]
            if self.haslabel:
                self.label += [opt.label[i] for _ in range(len(clean_files))]

        self.length = len(self.clean_filenames) * self.augment_size

        self.clean_crop_imgs = []
        self.noisy_crop_imgs = []
        f_th = len(opt.check_black_list) / opt.black_region_threshold
        cut = 0
        for i in range(len(self.clean_filenames)):
            # print('augment ' + self.clean_files[i])
            clean_imgs = input_guass_blur(load_function(self.clean_filenames[i], opt.input_img_neg).squeeze())
            noisy_imgs = input_guass_blur(load_function(self.noisy_filenames[i], opt.input_img_neg).squeeze())
            mask = get_mask(clean_imgs)
            for j in range(self.augment_size):
                while True:
                    random = np.random.rand(5)
                    mask_crop = crop(mask, self.clean_crop, random)
                    flag = 0
                    for (xl, xh, yl, yh) in opt.check_black_list:
                        tmp_sum = np.sum(mask_crop[round(self.clean_crop * xl): round(self.clean_crop * xh),
                                         round(self.clean_crop * yl): round(self.clean_crop * yh)])
                        if tmp_sum > opt.black_threshold:
                            flag += 1
                    if flag > f_th:
                        break
                    cut += 1
                self.clean_crop_imgs.append(crop(clean_imgs, self.clean_crop, random))
                self.noisy_crop_imgs.append(crop(noisy_imgs, self.noise_crop, random))
            print(str(cut / (i + 1) / self.augment_size))
        self.clean_crop_imgs = np.float32(self.clean_crop_imgs)
        self.noisy_crop_imgs = np.float32(self.noisy_crop_imgs)

        self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = np.array([self.clean_crop_imgs[x] for x in ilist])
        noisy_imgs = np.array([self.noisy_crop_imgs[x] for x in ilist])

        clean_imgs = torch.from_numpy(clean_imgs[:, np.newaxis, :, :])
        noisy_imgs = torch.from_numpy(noisy_imgs[:, np.newaxis, :, :])
        ##resize to file length
        ilist = ilist // self.augment_size
        if self.haslabel:
            label = torch.tensor([self.label[x] for x in ilist])
            return clean_imgs, noisy_imgs, label
        else:
            return clean_imgs, noisy_imgs

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        ##resize to file length
        ilist = ilist // self.augment_size
        return [self.clean_files[x] for x in ilist], [self.noisy_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_2d_augment_in_cache():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True, flip=True):
        self.shuffle = shuffle
        if opt.img_type == '.mrc':
            load_function = load_mrc_dataset
        else:
            load_function = load_tif_img

        if opt.rotate:
            crop = randomRotateAndCrop
            crop_mask = randomRotateAndCrop
        elif flip:
            crop = randomCrop
            crop_mask = randomCrop
        else:
            crop = randomCrop_noflip
            crop_mask = randomCrop_noflip

        # self.haslabel = opt.haslabel

        self.noise_crop = opt.noise_crop
        self.clean_crop = opt.clean_crop

        self.clean_files = sorted(os.listdir(os.path.join(base_dir, opt.gt_dir)))
        self.noisy_files = sorted(os.listdir(os.path.join(base_dir, opt.input_dir)))
        self.clean_filenames, self.noisy_filenames = [], []

        self.clean_filenames = [os.path.join(base_dir, opt.gt_dir, x) for x in self.clean_files if
                                is_specify_file(x, opt.img_type)]
        self.noisy_filenames = [os.path.join(base_dir, opt.input_dir, x) for x in self.noisy_files if
                                is_specify_file(x, opt.img_type)]
        self.augment_size = opt.augment_size // len(self.clean_filenames)
        self.length = len(self.clean_filenames) * self.augment_size

        self.clean_crop_imgs = []
        self.noisy_crop_imgs = []
        if opt.frontground_filter:
            f_th = len(opt.check_black_list) / opt.black_region_threshold
            cut = 0
        for i in range(len(self.clean_filenames)):
            # print('augment ' + self.clean_files[i])
            # if is_specify_file(self.clean_files[i],'.mrc'):
            clean_imgs = load_function(self.clean_filenames[i], opt.input_img_neg)

            noisy_imgs = load_function(self.noisy_filenames[i], opt.input_img_neg)
            if opt.frontground_filter:
                mask = get_mask(clean_imgs)

            for j in range(self.augment_size):
                if opt.frontground_filter:
                    while True:
                        random = np.random.rand(5)
                        mask_crop = crop_mask(mask, self.clean_crop, random)
                        flag = 0
                        for (xl, xh, yl, yh) in opt.check_black_list:
                            tmp_sum = np.sum(mask_crop[round(self.clean_crop * xl): round(self.clean_crop * xh),
                                             round(self.clean_crop * yl): round(self.clean_crop * yh)])
                            if tmp_sum > opt.black_threshold:
                                flag += 1
                        if flag > f_th:
                            break
                        cut += 1
                else:
                    random = np.random.rand(5)

                self.clean_crop_imgs.append(crop(clean_imgs, self.clean_crop, random))
                self.noisy_crop_imgs.append(crop(noisy_imgs, self.noise_crop, random))
            if opt.frontground_filter:
                print(str(cut / (i + 1) / self.augment_size))
        self.clean_crop_imgs = np.float32(self.clean_crop_imgs)
        self.noisy_crop_imgs = np.float32(self.noisy_crop_imgs)

        self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = np.array([self.clean_crop_imgs[x] for x in ilist])
        noisy_imgs = np.array([self.noisy_crop_imgs[x] for x in ilist])

        clean_imgs = torch.from_numpy(clean_imgs[:, np.newaxis])
        noisy_imgs = torch.from_numpy(noisy_imgs[:, np.newaxis])
        ##resize to file length
        # ilist = ilist // self.augment_size

        return clean_imgs, noisy_imgs

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        ##resize to file length
        ilist = ilist // self.augment_size
        return [self.clean_files[x] for x in ilist], [self.noisy_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_2d_augment_every_iter():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True, flip=True):
        self.shuffle = shuffle
        if opt.img_type == '.mrc':
            load_function = load_mrc_dataset_origin
        else:
            load_function = load_tif_img_origin

        if opt.rotate:
            self.crop_function = RotateAndCrop_by_center
            # crop_mask = randomRotateAndCrop
        elif opt.flip:
            self.crop_function = crop_by_center
            # crop_mask = randomCrop
        else:
            self.crop_function = crop_noflip_by_center
            # crop_mask = randomCrop_noflip

        # self.haslabel = opt.haslabel

        self.noise_crop = opt.noise_crop
        self.clean_crop = opt.clean_crop
        self.upfactor = self.clean_crop // self.noise_crop
        self.noisy_half = self.noise_crop // 2 * np.sqrt(2)
        if not opt.rotate:
            self.noisy_half = self.noise_crop // 2

        self.clean_files = sorted(os.listdir(os.path.join(base_dir, opt.gt_dir)))
        self.noisy_files = sorted(os.listdir(os.path.join(base_dir, opt.input_dir)))
        self.clean_filenames, self.noisy_filenames = [], []

        self.clean_filenames = [os.path.join(base_dir, opt.gt_dir, x) for x in self.clean_files if
                                is_specify_file(x, opt.img_type)]
        self.noisy_filenames = [os.path.join(base_dir, opt.input_dir, x) for x in self.noisy_files if
                                is_specify_file(x, opt.img_type)]

        self.length = opt.augment_size

        self.clean_imgs = []
        self.noisy_imgs = []
        self.noisy_mask_imgs = []

        for i in range(len(self.clean_filenames)):
            # print('augment ' + self.clean_files[i])
            # if is_specify_file(self.clean_files[i],'.mrc'):
            clean_imgs = load_function(self.clean_filenames[i])
            noisy_imgs = load_function(self.noisy_filenames[i])
            if not opt.input_img_neg:
                clean_imgs = np.where(clean_imgs < 0, 0, clean_imgs)
                noisy_imgs = np.where(noisy_imgs < 0, 0, noisy_imgs)
            if opt.normal:
                clean_imgs /= np.max(np.abs(clean_imgs))
                noisy_imgs /= np.max(np.abs(noisy_imgs))

            self.clean_imgs.append(clean_imgs)

            self.noisy_imgs.append(noisy_imgs)

            mask = get_mask(noisy_imgs)
            center_list = np.array(np.where(mask == 1)).transpose([1, 0])
            self.noisy_mask_imgs.append(center_list)
            # length_mask = len(center_list)
            # _, x, y = clean_imgs.shape

        # self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_postion(self):
        while True:
            index = int(np.random.rand() * len(self.clean_filenames))
            center_list = self.noisy_mask_imgs[index]
            clean_imgs = self.clean_imgs[index]
            noisy_imgs = self.noisy_imgs[index]

            length_mask = len(center_list)
            x, y = noisy_imgs.shape

            iter = 0
            while True:
                c_x, c_y = center_list[np.int16(length_mask * np.random.rand())]
                if c_x - self.noisy_half > 0 and c_x + self.noisy_half < x and c_y - self.noisy_half > 0 and c_y + self.noisy_half < y:
                    return clean_imgs, noisy_imgs, c_x, c_y
                if iter > 10:
                    break
                iter += 1

    def get_crop(self):
        noisy_crop_imgs = []
        clean_crop_imgs = []
        for i in range(self.batch_size):
            clean_imgs, noisy_imgs, c_x, c_y = self.get_postion()
            random = np.random.rand(5)
            clean_crop_imgs.append(
                self.crop_function(clean_imgs, self.clean_crop, c_x * self.upfactor, c_y * self.upfactor,
                                   random).astype(np.float32)[np.newaxis])
            noisy_crop_imgs.append(
                self.crop_function(noisy_imgs, self.noise_crop, c_x, c_y, random).astype(np.float32)[np.newaxis])

        return noisy_crop_imgs, clean_crop_imgs

    def get_data(self, i):
        noisy_imgs, clean_imgs = self.get_crop()

        clean_imgs = torch.from_numpy(np.array(clean_imgs))
        noisy_imgs = torch.from_numpy(np.array(noisy_imgs))
        ##resize to file length
        # ilist = ilist // self.augment_size

        return clean_imgs, noisy_imgs

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        ##resize to file length
        ilist = ilist // self.augment_size
        return [self.clean_files[x] for x in ilist], [self.noisy_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_3dnt_augment_in_cache():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True, flip=True):
        self.shuffle = shuffle
        if opt.img_type == '.mrc':
            load_function = load_3d_mrc_dataset_origin
        else:
            load_function = load_tif_img_origin

        if opt.rotate:
            crop = randomRotateAndCrop_3d
            crop_mask = randomRotateAndCrop
        elif flip:
            crop = randomCrop_3d
            crop_mask = randomCrop
        else:
            crop = randomCrop_noflip_3d
            crop_mask = randomCrop_noflip

        # self.haslabel = opt.haslabel

        self.noise_crop = opt.noise_crop
        self.clean_crop = opt.clean_crop

        self.clean_files = sorted(os.listdir(os.path.join(base_dir, opt.gt_dir)))
        self.noisy_files = sorted(os.listdir(os.path.join(base_dir, opt.input_dir)))
        self.clean_filenames = [os.path.join(base_dir, opt.gt_dir, x) for x in self.clean_files if
                                is_specify_file(x, opt.img_type)]
        self.noisy_filenames = [os.path.join(base_dir, opt.input_dir, x) for x in self.noisy_files if
                                is_specify_file(x, opt.img_type)]
        self.augment_size = opt.augment_size // len(self.clean_filenames)
        self.length = len(self.clean_filenames) * self.augment_size

        self.clean_crop_imgs = []
        self.noisy_crop_imgs = []
        if opt.frontground_filter:
            f_th = len(opt.check_black_list) / opt.black_region_threshold
            cut = 0
        for i in range(len(self.clean_filenames)):
            # print('augment ' + self.clean_files[i])
            clean_imgs = input3d_guass_blur(load_function(self.clean_filenames[i]))
            noisy_imgs = input3d_guass_blur(load_function(self.noisy_filenames[i]))
            if opt.img_frame != 0:
                clean_imgs = clean_imgs[:opt.img_frame]
                noisy_imgs = noisy_imgs[:opt.img_frame]
            if not opt.input_img_neg:
                clean_imgs = np.where(clean_imgs < 0, 0, clean_imgs)
                noisy_imgs = np.where(noisy_imgs < 0, 0, noisy_imgs)
            if opt.normal:
                clean_imgs /= np.max(np.abs(clean_imgs))
                noisy_imgs /= np.max(np.abs(noisy_imgs))
            if opt.frontground_filter:
                mask = get_mask(clean_imgs)
                center_list = np.array(np.where(mask == 1)).transpose([1, 0])
            for j in range(self.augment_size):

                if opt.frontground_filter:
                    while True:
                        random = np.random.rand(5)
                        mask_crop = crop_mask(mask, self.clean_crop, random)
                        flag = 0
                        for (xl, xh, yl, yh) in opt.check_black_list:
                            tmp_sum = np.sum(mask_crop[round(self.clean_crop * xl): round(self.clean_crop * xh),
                                             round(self.clean_crop * yl): round(self.clean_crop * yh)])
                            if tmp_sum > opt.black_threshold:
                                flag += 1
                        if flag > f_th:
                            break
                        cut += 1
                else:
                    random = np.random.rand(5)
                self.clean_crop_imgs.append(crop(clean_imgs, self.clean_crop, random))
                self.noisy_crop_imgs.append(crop(noisy_imgs, self.noise_crop, random))
            if opt.frontground_filter:
                print(str(cut / (i + 1) / self.augment_size))
        self.clean_crop_imgs = np.float32(self.clean_crop_imgs)
        self.noisy_crop_imgs = np.float32(self.noisy_crop_imgs)

        self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = np.array([self.clean_crop_imgs[x] for x in ilist])
        noisy_imgs = np.array([self.noisy_crop_imgs[x] for x in ilist])

        clean_imgs = torch.from_numpy(clean_imgs[:, np.newaxis])
        noisy_imgs = torch.from_numpy(noisy_imgs[:, np.newaxis])
        ##resize to file length
        # ilist = ilist // self.augment_size

        return clean_imgs, noisy_imgs

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        ##resize to file length
        ilist = ilist // self.augment_size
        return [self.clean_files[x] for x in ilist], [self.noisy_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length

class Dataset_3dnt_augment_eachit_extend_baseline(Dataset_baseline):
    def __init__(self, base_dir, opt, shuffle=True):
        super().__init__(base_dir, opt, shuffle)
        super().load_imgs()

    def get_data(self, i):
        return super().augment_batch(super().get_img_gt_in)

class Dataset_3dnt_augment_in_cache_extend_baseline(Dataset_baseline):
    def __init__(self, base_dir, opt, shuffle=True):
        super().__init__(base_dir, opt, shuffle)
        super().load_imgs()
        super().augment_in_cache(super().get_img_gt_in)

    def get_data(self, i):
        return super().get_data_from_augment_in_cache(i)

class Dataset_3dnt_augment_eachit_in_cache_extend_baseline_VSR(Dataset_baseline):
    def __init__(self, base_dir, opt, shuffle=True):
        super().__init__(base_dir, opt, shuffle)
        super().load_imgs()

    def get_data(self, i):
        # return super().get_data_from_augment_in_cache(i)
        return super().augment_batch_VSR(super().get_img_gt_in)

class Dataset_n2n_augment_eachit_in_cache_extend_baseline(Dataset_baseline):
    def __init__(self, base_dir, opt, shuffle=True):
        super().__init__(base_dir, opt, shuffle)
        super().load_imgs_prs()
        # super().augment_in_cache()

    def get_data(self, i):
        # return super().get_data_from_augment_in_cache(i)
        return super().augment_batch(super().get_img_prs)

class Dataset_rolling_augment_eachit_in_cache_extend_baseline(Dataset_baseline):
    def __init__(self, base_dir, opt, shuffle=True):
        super().__init__(base_dir, opt, shuffle)
        super().load_imgs_rolling()

    def get_data(self, i):
        return super().augment_batch(super().get_img_rolling)

class Dataset_3dnt_n2n_augment_in_cuda():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True, flip=True):
        self.shuffle = shuffle
        assert opt.upfactor == 1
        if opt.img_type == '.mrc':
            load_function = load_3d_mrc_dataset_origin
        else:
            load_function = load_tif_img_origin

        if opt.frontground_filter:
            if opt.rotate:
                crop = RotateAndCrop_3d_by_center
            elif flip:
                crop = crop_3d_by_center
            else:
                crop = crop_noflip_3d_by_center
        else:
            if opt.rotate:
                crop = randomRotateAndCrop_3d
            elif flip:
                crop = randomCrop_3d
            else:
                crop = randomCrop_noflip_3d

        # self.haslabel = opt.haslabel
        device = opt.device

        self.noise_crop = opt.noise_crop
        self.clean_crop = opt.clean_crop

        self.clean_files = sorted(os.listdir(os.path.join(base_dir, opt.gt_dir)))
        self.noisy_files = sorted(os.listdir(os.path.join(base_dir, opt.input_dir)))
        self.clean_filenames = [os.path.join(base_dir, opt.gt_dir, x) for x in self.clean_files if
                                is_specify_file(x, opt.img_type)]
        self.noisy_filenames = [os.path.join(base_dir, opt.input_dir, x) for x in self.noisy_files if
                                is_specify_file(x, opt.img_type)]
        self.augment_size = opt.augment_size // len(self.clean_filenames)
        self.length = len(self.clean_filenames) * self.augment_size

        self.clean_crop_imgs = []
        self.noisy_crop_imgs = []
        half = self.clean_crop // 2 * np.sqrt(2)

        for i in range(len(self.clean_filenames)):
            # print('augment ' + self.clean_files[i])
            imgs = load_function(self.clean_filenames[i])
            imgs = np.transpose(imgs, opt.input_map)  # axis: diff z x y
            length = imgs.shape[0]
            p1 = np.int8(np.random.rand() * length)
            p2 = np.int8(np.random.rand() * length)
            if p2 == p1:
                p2 = (p2 + 1) % length
            clean_imgs = input3d_guass_blur(imgs[p1])
            noisy_imgs = input3d_guass_blur(imgs[p2])
            if opt.img_frame != 0:
                clean_imgs = clean_imgs[:opt.img_frame]
                noisy_imgs = noisy_imgs[:opt.img_frame]
            if opt.frontground_filter:
                mask = get_mask(clean_imgs)
                center_list = np.array(np.where(mask == 1)).transpose([1, 0])
                length_mask = len(center_list)
                _, x, y = clean_imgs.shape

            iter = 0
            for j in range(self.augment_size):
                if opt.frontground_filter:
                    while True:
                        c_x, c_y = center_list[np.int16(length_mask * np.random.rand())]
                        if c_x - half > 0 and c_x + half < x and c_y - half > 0 and c_y + half < y:
                            break
                    random = np.random.rand(3)
                    self.clean_crop_imgs.append(
                        torch.from_numpy(
                            crop(clean_imgs, self.clean_crop, c_x, c_y, random).astype(np.float32)[
                                np.newaxis, np.newaxis]).to(device))
                    self.noisy_crop_imgs.append(
                        torch.from_numpy(
                            crop(noisy_imgs, self.noise_crop, c_x, c_y, random).astype(np.float32)[
                                np.newaxis, np.newaxis]).to(device))
                else:
                    random = np.random.rand(5)
                    self.clean_crop_imgs.append(
                        torch.from_numpy(
                            crop(clean_imgs, self.clean_crop, random).astype(np.float32)[np.newaxis, np.newaxis]).to(
                            device))
                    self.noisy_crop_imgs.append(
                        torch.from_numpy(
                            crop(noisy_imgs, self.noise_crop, random).astype(np.float32)[np.newaxis, np.newaxis]).to(
                            device))

                if iter > self.augment_size * 5:
                    self.augment_size = (opt.augment_size - len(self.clean_crop_imgs)) // (
                            len(self.clean_filenames) - i - 1)
                    break  # delete bad input
                iter += 1

        # self.clean_crop_imgs = np.float32(self.clean_crop_imgs)
        # self.noisy_crop_imgs = np.float32(self.noisy_crop_imgs)

        self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = torch.cat([self.clean_crop_imgs[x] for x in ilist], 0)
        noisy_imgs = torch.cat([self.noisy_crop_imgs[x] for x in ilist], 0)
        ##resize to file length
        # ilist = ilist // self.augment_size

        return clean_imgs, noisy_imgs

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        ##resize to file length
        ilist = ilist // self.augment_size
        return [self.clean_files[x] for x in ilist], [self.noisy_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_2d_multi_channel_augment_in_cache():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True, flip=True):
        self.shuffle = shuffle
        if opt.img_type == '.mrc':
            load_function = load_3d_mrc_dataset
        else:
            load_function = load_tif_img

        if rotate:
            crop = randomRotateAndCrop_3d
            crop_mask = randomRotateAndCrop
        elif flip:
            crop = randomCrop_3d
            crop_mask = randomCrop
        else:
            crop = randomCrop_noflip_3d
            crop_mask = randomCrop_noflip

        # self.haslabel = opt.haslabel

        self.noise_crop = opt.noise_crop
        self.clean_crop = opt.clean_crop

        self.input_channel = opt.input_channel
        self.output_channel = opt.output_channel

        self.clean_files = sorted(os.listdir(os.path.join(base_dir, opt.gt_dir)))
        self.noisy_files = sorted(os.listdir(os.path.join(base_dir, opt.input_dir)))
        self.clean_filenames = [os.path.join(base_dir, opt.gt_dir, x) for x in self.clean_files if
                                is_specify_file(x, opt.img_type)]
        self.noisy_filenames = [os.path.join(base_dir, opt.input_dir, x) for x in self.noisy_files if
                                is_specify_file(x, opt.img_type)]
        self.augment_size = opt.augment_size // len(self.clean_filenames)
        self.length = len(self.clean_filenames) * self.augment_size

        self.clean_crop_imgs = []
        self.noisy_crop_imgs = []
        if opt.frontground_filter:
            f_th = len(opt.check_black_list) / opt.black_region_threshold
            cut = 0
        for i in range(len(self.clean_filenames)):
            # print('augment ' + self.clean_files[i])
            clean_imgs = load_function(self.clean_filenames[i], opt.input_img_neg)
            noisy_imgs = load_function(self.noisy_filenames[i], opt.input_img_neg)
            if self.input_channel == 1:
                noisy_imgs = noisy_imgs[np.newaxis]
            if self.output_channel == 1:
                clean_imgs = clean_imgs[np.newaxis]
            if opt.img_frame != 0:
                clean_imgs = clean_imgs[:opt.img_frame]
                noisy_imgs = noisy_imgs[:opt.img_frame]
            if opt.frontground_filter:
                mask = get_mask(clean_imgs)
            for j in range(self.augment_size):
                if opt.frontground_filter:
                    while True:
                        random = np.random.rand(5)
                        mask_crop = crop_mask(mask, self.clean_crop, random)
                        flag = 0
                        for (xl, xh, yl, yh) in opt.check_black_list:
                            tmp_sum = np.sum(mask_crop[round(self.clean_crop * xl): round(self.clean_crop * xh),
                                             round(self.clean_crop * yl): round(self.clean_crop * yh)])
                            if tmp_sum > opt.black_threshold:
                                flag += 1
                        if flag > f_th:
                            break
                        cut += 1
                else:
                    random = np.random.rand(5)
                self.clean_crop_imgs.append(crop(clean_imgs, self.clean_crop, random))
                self.noisy_crop_imgs.append(crop(noisy_imgs, self.noise_crop, random))
            if opt.frontground_filter:
                print(str(cut / (i + 1) / self.augment_size))
        self.clean_crop_imgs = np.float32(self.clean_crop_imgs)
        self.noisy_crop_imgs = np.float32(self.noisy_crop_imgs)

        self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = np.array([self.clean_crop_imgs[x] for x in ilist])
        noisy_imgs = np.array([self.noisy_crop_imgs[x] for x in ilist])

        clean_imgs = torch.from_numpy(clean_imgs)
        noisy_imgs = torch.from_numpy(noisy_imgs)
        ##resize to file length
        # ilist = ilist // self.augment_size

        return clean_imgs, noisy_imgs

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        ##resize to file length
        ilist = ilist // self.augment_size
        return [self.clean_files[x] for x in ilist], [self.noisy_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_3dnt_augment_in_cuda():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True, flip=True):
        self.device = torch.device(opt.device)
        if opt.img_type == '.mrc':
            load_function = load_3d_mrc_dataset
        else:
            load_function = load_tif_img

        if rotate:
            self.crop = randomRotateAndCrop_3d
            self.crop_mask = randomRotateAndCrop
        elif flip:
            self.crop = randomCrop_3d
            self.crop_mask = randomCrop
        else:
            self.crop = randomCrop_noflip_3d
            self.crop_mask = randomCrop_noflip

        self.noise_crop = opt.noise_crop
        self.clean_crop = opt.clean_crop

        clean_files = sorted(os.listdir(os.path.join(base_dir, opt.gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(base_dir, opt.input_dir)))
        self.clean_filenames = [os.path.join(base_dir, opt.gt_dir, x) for x in clean_files if
                                is_specify_file(x, opt.img_type)]
        self.noisy_filenames = [os.path.join(base_dir, opt.input_dir, x) for x in noisy_files if
                                is_specify_file(x, opt.img_type)]
        self.length = opt.augment_size

        self.clean_imgs, self.noisy_imgs, self.clean_imgs_mask = [], [], []
        for i in range(len(self.clean_filenames)):
            # print('augment ' + self.clean_files[i])
            clean_imgs_temp = input3d_guass_blur(load_function(self.clean_filenames[i], opt.input_img_neg))
            noisy_imgs_temp = input3d_guass_blur(load_function(self.noisy_filenames[i], opt.input_img_neg))
            if opt.img_frame != 0:
                clean_imgs_temp = clean_imgs_temp[:opt.img_frame]
                noisy_imgs_temp = noisy_imgs_temp[:opt.img_frame]
            self.clean_imgs.append(clean_imgs_temp)
            self.noisy_imgs.append(noisy_imgs_temp)
            self.clean_imgs_mask.append(get_mask(clean_imgs_temp))
        self.clean_imgs = np.array(self.clean_imgs, np.float32)
        self.noisy_imgs = np.array(self.noisy_imgs, np.float32)
        self.clean_imgs_mask = np.array(self.clean_imgs_mask, np.float32)

        # self.iters = np.arange(0, self.length)
        self.batch_size = opt.batch_size

        self.check_black_list = opt.check_black_list
        self.black_threshold = opt.black_threshold
        self.black_region_threshold = opt.black_region_threshold

    #     return clean_imgs,noisy_imgs

    def augment(self):
        self.clean_crop_imgs, self.noisy_crop_imgs = [], []
        f_th = len(self.check_black_list) / self.black_region_threshold
        cut = 0
        for i in range(self.length):
            # print('augment ' + self.clean_files[i])
            while True:
                index = np.int16(np.random.rand() * len(self.clean_imgs))
                random = np.random.rand(5)
                mask_crop = self.crop_mask(self.clean_imgs_mask[index], self.clean_crop, random)
                flag = 0
                for (xl, xh, yl, yh) in self.check_black_list:
                    tmp_sum = np.sum(mask_crop[round(self.clean_crop * xl): round(self.clean_crop * xh),
                                     round(self.clean_crop * yl): round(self.clean_crop * yh)])
                    if tmp_sum > self.black_threshold:
                        flag += 1
                if flag > f_th:
                    break
                cut += 1
            self.clean_crop_imgs.append(self.crop(self.clean_imgs[index], self.clean_crop, random))
            self.noisy_crop_imgs.append(self.crop(self.noisy_imgs[index], self.noise_crop, random))
        print(str(cut / self.length))
        self.clean_crop_imgs = torch.from_numpy(np.float32(self.clean_crop_imgs)[:, np.newaxis]).to(self.device)
        self.noisy_crop_imgs = torch.from_numpy(np.float32(self.noisy_crop_imgs)[:, np.newaxis]).to(self.device)

    def get_data(self, i):
        if i == 0:
            self.augment()
        start = i * self.batch_size
        end = (i + 1) * self.batch_size
        if (i + 1) * self.batch_size >= self.length:
            end = self.length

        clean_imgs = self.clean_crop_imgs[start:end]
        noisy_imgs = self.noisy_crop_imgs[start:end]
        ##resize to file length
        # ilist = ilist // self.augment_size

        return clean_imgs, noisy_imgs

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_N2N_augment_in_cache():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True):
        self.shuffle = shuffle
        if rotate:
            crop = randomRotateAndCrop
        else:
            crop = randomCrop
        self.augment_size = opt.augment_size

        self.files, self.filenames = [], []
        walk = os.walk(base_dir)

        for path, dir_lsit, file_list in walk:
            # self.group_list = dir_lsit
            for dir_name in dir_lsit:
                files = sorted(os.listdir(os.path.join(base_dir, dir_name)))
                self.files.append(files)
                self.filenames.append([os.path.join(base_dir, dir_name, x) for x in files if is_tif_file(x)])
        self.filenames = np.array(self.filenames)
        self.pairs = self.filenames.shape[1]
        self.length = len(self.filenames) * self.augment_size * (self.pairs * (self.pairs - 1))

        self.clean_crop_imgs = np.zeros((self.length, opt.clean_crop, opt.clean_crop), np.float32)
        self.noisy_crop_imgs = np.zeros((self.length, opt.noise_crop, opt.noise_crop), np.float32)

        p = 0
        for groups in range(len(self.filenames)):
            load_imgs = [load_tif_img(self.filenames[groups][x]) for x in range(self.pairs)]
            for p1 in range(self.pairs):
                for p2 in range(self.pairs):
                    if p1 == p2:
                        continue
                    for j in range(self.augment_size):
                        random = np.random.rand(5)
                        self.clean_crop_imgs[p] = crop(load_imgs[p1], opt.clean_crop, random)
                        self.noisy_crop_imgs[p] = crop(load_imgs[p2], opt.noise_crop, random)
                        p += 1

        self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = np.array([self.clean_crop_imgs[x] for x in ilist])
        noisy_imgs = np.array([self.noisy_crop_imgs[x] for x in ilist])

        clean_imgs = torch.from_numpy(clean_imgs[:, np.newaxis, :, :])
        noisy_imgs = torch.from_numpy(noisy_imgs[:, np.newaxis, :, :])
        ##resize to file length
        # ilist = ilist // self.augment_size

        return clean_imgs, noisy_imgs

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        ##resize to file length
        ilist = ilist // self.augment_size
        groups = ilist // self.pairs // (self.pairs - 1)
        input_i = ilist // len(self.filenames) // (self.pairs - 1)
        gt_i = ilist // len(self.filenames) // self.pairs
        for i in range(len(input_i)):
            if input_i[i] <= gt_i[i]:
                gt_i[i] += 1
        input_filenames = [self.filenames[x][y] for x, y in zip(groups, input_i)]
        target_filenames = [self.filenames[x][y] for x, y in zip(groups, gt_i)]
        return input_filenames, target_filenames

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length


class Dataset_augment_normal():
    def __init__(self, base_dir, opt, rotate=True, shuffle=True):

        self.shuffle = shuffle
        if rotate:
            crop_function = randomRotateAndCrop
        else:
            crop_function = randomCrop
        self.haslabel = opt.haslabel
        if self.haslabel:
            self.label_num = len(opt.label)
        self.augment_size = opt.augment_size
        self.noise_crop = opt.noise_crop
        self.clean_crop = opt.clean_crop

        self.clean_files, self.noisy_files, self.clean_filenames, self.noisy_filenames = [], [], [], []
        # if self.haslabel:
        self.label = []
        for i in range(len(base_dir)):
            clean_files = sorted(os.listdir(os.path.join(base_dir[i], opt.gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(base_dir[i], opt.input_dir)))
            self.clean_files += clean_files
            self.noisy_files += noisy_files
            self.clean_filenames += [os.path.join(base_dir[i], opt.gt_dir, x) for x in clean_files if is_tif_file(x)]
            self.noisy_filenames += [os.path.join(base_dir[i], opt.input_dir, x) for x in noisy_files if is_tif_file(x)]
            # if self.haslabel:
            self.label += [opt.label[i] for j in range(len(clean_files))]

        self.length = len(self.clean_filenames) * self.augment_size

        self.clean_crop_imgs = np.zeros((self.length, self.clean_crop, self.clean_crop), np.float32)
        self.noisy_crop_imgs = np.zeros((self.length, self.noise_crop, self.noise_crop), np.float32)
        for i in range(len(self.clean_filenames)):
            clean_imgs = normalize(load_tif_img(self.clean_filenames[i]), opt.normal_max, opt.normal_min)
            noisy_imgs = normalize(load_tif_img(self.noisy_filenames[i]), opt.normal_max, opt.normal_min)
            mean = np.mean(noisy_imgs)
            for j in range(self.augment_size):
                random = np.random.rand(5)
                croped = crop_function(noisy_imgs, self.noise_crop, random)
                while np.mean(croped) < mean:
                    random = np.random.rand(5)
                    croped = crop_function(noisy_imgs, self.noise_crop, random)
                self.noisy_crop_imgs[i * self.augment_size + j] = croped
                self.clean_crop_imgs[i * self.augment_size + j] = crop_function(clean_imgs, self.clean_crop, random)

        self.iters = np.arange(0, self.length)

        self.batch_size = opt.batch_size

    #     return clean_imgs,noisy_imgs

    def get_data(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = np.array([self.clean_crop_imgs[x] for x in ilist])
        noisy_imgs = np.array([self.noisy_crop_imgs[x] for x in ilist])

        clean_imgs = torch.from_numpy(clean_imgs[:, np.newaxis, :, :])
        noisy_imgs = torch.from_numpy(noisy_imgs[:, np.newaxis, :, :])
        ##resize to file length
        ilist = ilist // self.augment_size
        # if self.haslabel:
        label = torch.tensor([self.label[x] for x in ilist])
        return [clean_imgs, noisy_imgs, label, len(ilist)]
        # else:
        #     return [clean_imgs, noisy_imgs]

    def get_filename(self, i):
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]
        ##resize to file length
        ilist = ilist // self.augment_size
        return [self.clean_files[x] for x in ilist]

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length

# def get_dataset_by_name(datname):
