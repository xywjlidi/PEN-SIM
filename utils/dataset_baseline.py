import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from read_utils.read_utils import load_tif_img, is_tif_file, normalize, load_mrc_dataset, is_specify_file, \
    load_tif_2d_img_origin, load_tif_img_origin, load_3d_mrc_dataset_origin, load_mrc_dataset_origin, prctile_norm
from scipy.ndimage.filters import gaussian_filter

class randomCropFunction:
    def __init__(self, is3D, filter, rotate, rotate90, flip):
        self.is3D = is3D
        if filter:
            self.crop_function = self.crop_function_center
            if rotate:
                self.crop = self.rotateAndCropByCenter
            elif rotate90:
                self.crop = self.rotate90AndFlipByCenter
            elif flip:
                self.crop = self.cropAndFlipByCenter
            else:
                self.crop = self.cropByCenter
        else:
            self.crop_function = self.crop_function_random
            if rotate:
                self.crop = self.randomRotateAndCrop
            elif rotate90:
                self.crop = self.randomRotato90AndFlip
            elif flip:
                self.crop = self.randomCropAndFlip
            else:
                self.crop = self.randomCrop

    def crop_function_random(self, img, cropsize, xc=0, yc=0, zc=0, random=np.zeros(6), crop_depth=None):
        if not self.is3D:
            return self.crop(img, cropsize, random)
        nz, nx, ny = img.shape
        if crop_depth is None:
            re = np.zeros([nz, cropsize, cropsize])
            for i in range(nz):
                re[i] = self.crop(img[i], cropsize, random)
        else:
            start = np.floor(random[5] * (nz - crop_depth)).astype(np.int16)
            re = np.zeros([crop_depth, cropsize, cropsize])
            for i in range(crop_depth):
                re[i] = self.crop(img[i+start], cropsize, random[:5])
        return re


    def crop_function_center(self, img, cropsize, xc, yc, zc, random, crop_depth=None):
        #TODO x y in rotation reversely compared with mask calculation
        if not self.is3D:
            return self.crop(img, cropsize, yc, xc, random)
        nz, nx, ny = img.shape
        if crop_depth is None:
            re = np.zeros([nz, cropsize, cropsize])
            for i in range(nz):
                re[i] = self.crop(img[i], cropsize, yc, xc, random)
        else:
            start = zc - crop_depth // 2
            re = np.zeros([crop_depth, cropsize, cropsize])
            for i in range(crop_depth):
                re[i] = self.crop(img[i+start], cropsize, yc, xc, random)
        return re

    def randomRotateAndCrop(self,img, cropsize, random):
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

    def rotateAndCropByCenter(self, img, cropsize, xc, yc, random):
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

    def randomRotato90AndFlip(self, img, cropsize, random):
        y = round(random[0] * (img.shape[0] - cropsize))
        x = round(random[1] * (img.shape[1] - cropsize))

        re = img[y:y + cropsize, x:x + cropsize]

        if random[2] > 0.5:
            re = re[::-1, :]
        if random[3] > 0.5:
            re = re[:, ::-1]
        if random[4] > 0.5:
            re = np.rot90(re, axes=(0,1)).copy()
        #     print(x,y,flip)
        # print(flip)
        # print(theta,srcTri)
        return re

    def randomCropAndFlip(self, img, cropsize, random):
        y = round(random[0] * (img.shape[0] - cropsize))
        x = round(random[1] * (img.shape[1] - cropsize))

        re = img[y:y + cropsize, x:x + cropsize]

        if random[2] > 0.5:
            re = re[::-1, :]
        if random[3] > 0.5:
            re = re[:, ::-1]
        #     print(x,y,flip)
        # print(flip)
        # print(theta,srcTri)
        return re

    def rotate90AndFlipByCenter(self, img, cropsize, xc, yc, random):
        le = cropsize // 2

        re = img[yc - le:yc + le, xc - le:xc + le]

        if random[2] > 0.5:
            re = re[::-1, :]
        if random[3] > 0.5:
            re = re[:, ::-1]
        if random[4] > 0.5:
            re = np.rot90(re, axes=(0,1)).copy()
        #     print(x,y,flip)
        # print(flip)
        # print(theta,srcTri)
        return re

    def cropAndFlipByCenter(self, img, cropsize, xc, yc, random):
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

    def randomCrop(self, img, cropsize, random):
        y = round(random[0] * (img.shape[0] - cropsize))
        x = round(random[1] * (img.shape[1] - cropsize))

        re = img[y:y + cropsize, x:x + cropsize]

        return re

    def cropByCenter(self, img, cropsize, xc, yc, random):
        le = cropsize // 2

        re = img[yc - le:yc + le, xc - le:xc + le]

        return re


def get_mask_2d(img, th=0.05):
    if np.ndim(img) == 3:
        img = img[0]
    img = prctile_norm(np.float32(img), 0.1, 99.9)
    img_guass = cv2.GaussianBlur(img, (3, 3), 3) - cv2.GaussianBlur(img, (25, 25), 50)
    x, y = img_guass.shape
    # sorted = np.sort(img_guass.reshape((x * y)))
    # threshold = sorted[round(len(sorted) * 0.75)]

    while True:
        mask = np.where(img_guass > th, 1, 0)
        if np.sum(mask) < x * y * 0.005:
            th *= 0.8
        else:
            break

    return mask

def get_mask_3d(img, th=0.05, zleft=0, zright=0):#TODO check
    nz, nx, ny = img.shape
    if zright==0: zright = nz
    img = prctile_norm(np.float32(img), 0.1, 99.9)
    img_guass = np.zeros(img.shape)
    for i in range(nz):
        img_guass[i] = cv2.GaussianBlur(img[i], (3, 3), 3) - cv2.GaussianBlur(img[i], (25, 25), 50)
    # sorted = np.sort(img_guass.reshape((x * y)))
    # threshold = sorted[round(len(sorted) * 0.75)]
    img_guass[:zleft,:,:] = 0
    img_guass[zright:nz,:,:] = 0

    while True:
        mask = np.where(img_guass > th, 1, 0)
        if np.sum(mask) < nz * nx * ny * 0.003:
            th *= 0.8
        else:
            break

    return mask


def input3d_guass_blur(img):
    nz, _, _ = img.shape
    for i in range(nz):
        img[i] = cv2.GaussianBlur(img[i], (3, 3), 0.6, 0.6)
    return img


def input_guass_blur(img):
    img = gaussian_filter(img, sigma=0.6)
    return img


def check_blank_crop_3d(img):
    # plt.figure()
    # plt.imshow(img[0], 'gray')
    img_sort = np.sort(np.reshape(img[0], img.shape[1] * img.shape[2]))
    intensity_len = img_sort[len(img_sort) // 10 * 9] - img_sort[len(img_sort) // 10]
    # plt.title(intensity_len)
    # plt.close()
    if (intensity_len < 1e-2):
        return True
    return False


def check_blank_crop(img):
    if (np.var(img) < 1e-5):
        return True
    return False



class Dataset_baseline():
    def __init__(self, base_dir, opt, shuffle=True):
        self.shuffle = shuffle

        # TODO: maybe add crop on depth
        self.noise_crop = opt.noise_crop
        self.upfactor = opt.upfactor
        self.zupfactor = opt.zupfactor
        self.clean_crop = self.noise_crop * self.upfactor
        self.crop_depth = opt.crop_depth

        self.noisy_half = self.noise_crop // 2 * np.sqrt(2)
        if not opt.rotate:
            self.noisy_half = self.noise_crop // 2

        self.base_dir = base_dir
        self.length = opt.augment_size
        self.batch_size = opt.batch_size
        self.opt = opt

        self.crop_function = randomCropFunction(is3D=opt.is3D, filter=opt.frontground_filter, rotate=opt.rotate,
                                                rotate90=opt.rotate90, flip=opt.flip).crop_function

        self.get_crop = self.get_random_crop
        if opt.frontground_filter:
            self.get_crop = self.get_crop_from_filter

        if self.opt.img_type == '.mrc':
            if self.opt.is3D:
                self.load_function = load_3d_mrc_dataset_origin
            else:
                self.load_function = load_mrc_dataset_origin
        else:
            if self.opt.is3D:
                self.load_function = load_tif_img_origin
            else:
                self.load_function = load_tif_2d_img_origin

        self.nzhalf_left = 0
        self.nzhalf_right = 0
        if self.crop_depth is None:
            self.get_position = self.get_position_2d
            self.get_mask = self.get_mask_2d
        elif opt.is3D:
            self.get_position = self.get_position_3d
            self.get_mask = self.get_mask_3d
            self.nzhalf_left = self.crop_depth // 2
            self.nzhalf_right = self.crop_depth - self.nzhalf_left

    def get_mask_2d(self, img, th=0.05):
        if np.ndim(img) == 3:
            img = img[0]
        img = prctile_norm(np.float32(img), 0.1, 99.9)
        img_guass = cv2.GaussianBlur(img, (3, 3), 3) - cv2.GaussianBlur(img, (25, 25), 50)
        nx, ny = img_guass.shape
        # sorted = np.sort(img_guass.reshape((x * y)))
        # threshold = sorted[round(len(sorted) * 0.75)]

        clean_half = self.noisy_half * self.upfactor

        img_guass[0:clean_half, :] = 0
        img_guass[nx - clean_half:nx, :] = 0
        img_guass[:, 0:clean_half] = 0
        img_guass[:, ny - clean_half:ny] = 0

        while True:
            mask = np.where(img_guass > th, 1, 0)
            if np.sum(mask) < nx * ny * 0.005:
                th *= 0.8
            else:
                break

        return mask

    def get_mask_3d(self, img, th=0.05):  # TODO check
        nz, nx, ny = img.shape
        img = prctile_norm(np.float32(img), 0.1, 99.9)
        img_guass = np.zeros(img.shape)
        for i in range(nz):
            img_guass[i] = cv2.GaussianBlur(img[i], (3, 3), 3) - cv2.GaussianBlur(img[i], (25, 25), 50)
        # sorted = np.sort(img_guass.reshape((x * y)))
        # threshold = sorted[round(len(sorted) * 0.75)]
        clean_half = self.noisy_half * self.upfactor

        img_guass[:self.nzhalf_left, :, :] = 0
        img_guass[nz-self.nzhalf_right+1:nz, :, :] = 0
        img_guass[:, 0:clean_half, :] = 0
        img_guass[:, nx - clean_half:nx, :] = 0
        img_guass[:, :, 0:clean_half] = 0
        img_guass[:, :, ny - clean_half:ny] = 0

        while True:
            mask = np.where(img_guass > th, 1, 0)
            if np.sum(mask) < nz * nx * ny * 0.001 and np.sum(mask) < 200:
                th *= 0.8
            else:
                break

        return mask

    def load_imgs(self):

        clean_files = sorted(os.listdir(os.path.join(self.base_dir, self.opt.gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(self.base_dir, self.opt.input_dir)))

        clean_filenames = [os.path.join(self.base_dir, self.opt.gt_dir, x) for x in clean_files if
                                is_specify_file(x, self.opt.img_type)]
        noisy_filenames = [os.path.join(self.base_dir, self.opt.input_dir, x) for x in noisy_files if
                                is_specify_file(x, self.opt.img_type)]

        self.clean_imgs = []
        self.noisy_imgs = []
        self.clean_mask_imgs = []

        for i in range(len(clean_filenames)):
            clean_imgs = input_guass_blur(self.load_function(clean_filenames[i]))
            noisy_imgs = input_guass_blur(self.load_function(noisy_filenames[i]))

            if self.opt.is3D:
                if self.opt.img_frame != 0:
                    clean_imgs = clean_imgs[:self.opt.img_frame]
                    noisy_imgs = noisy_imgs[:self.opt.img_frame]

            if not self.opt.input_img_neg:
                clean_imgs = np.where(clean_imgs < 0, 0, clean_imgs)
                noisy_imgs = np.where(noisy_imgs < 0, 0, noisy_imgs)
            if self.opt.normal:
                clean_imgs /= np.max(np.abs(clean_imgs))
                noisy_imgs /= np.max(np.abs(noisy_imgs))

            self.clean_imgs.append(clean_imgs)
            self.noisy_imgs.append(noisy_imgs)

            if self.opt.frontground_filter:
                mask = self.get_mask(clean_imgs, self.opt.fg_th)
                center_list = np.array(np.where(mask == 1)).transpose([1, 0])
                self.clean_mask_imgs.append(center_list)
                # length_mask = len(center_list)
                # _, x, y = clean_imgs.shape

        self.length_origin = len(self.clean_imgs)

    def load_imgs_prs(self):

        self.noisy_imgs = []
        self.clean_mask_imgs = []

        shape_pair = None
        # shape = []
        for pair in self.opt.pairs:
            noisy_filenames = []
            for dir in self.opt.input_dir:
                noisy_files = sorted([x for x in os.listdir(os.path.join(self.base_dir, dir))
                                 if x.find(pair) >= 0])
                noisy_filenames += [os.path.join(self.base_dir, dir, x) for x in noisy_files if
                                is_specify_file(x, self.opt.img_type)]

            noisy_imgs_pair = []
            clean_mask_imgs_pair = []
            for i in range(len(noisy_filenames)):
                if self.opt.is3D:
                    noisy_imgs = input_guass_blur(self.load_function(noisy_filenames[i]))
                else:
                    noisy_imgs = input3d_guass_blur(self.load_function(noisy_filenames[i]))

                if self.opt.is3D:
                    if self.opt.img_frame != 0:
                        noisy_imgs = noisy_imgs[:self.opt.img_frame]

                if not self.opt.input_img_neg:
                    noisy_imgs = np.where(noisy_imgs < 0, 0, noisy_imgs)
                if self.opt.normal:
                    for i in range(noisy_imgs.shape[0]):
                        noisy_imgs[i] /= np.max(np.abs(noisy_imgs[i]))

                # for i in range(noisy_imgs.shape[0]):
                #     noisy_imgs[i] = input_guass_blur(noisy_imgs[i])

                # if pair == self.opt.pairs[0]:
                #     shape.append(np.array(noisy_imgs).shape)
                # else:
                #     assert shape[i] == np.array(noisy_imgs).shape

                noisy_imgs_pair.append(noisy_imgs)

                if self.opt.frontground_filter:
                    mask = self.get_mask(noisy_imgs[0], self.opt.fg_th)
                    center_list = np.array(np.where(mask == 1)).transpose([1, 0])
                    clean_mask_imgs_pair.append(center_list)
                    # length_mask = len(center_list)
                    # _, x, y = clean_imgs.shape

            if shape_pair is None:
                shape_pair = len(noisy_imgs_pair)
            else:
                assert len(noisy_imgs_pair) == shape_pair

            self.noisy_imgs.append(noisy_imgs_pair)
            if self.opt.frontground_filter:
                self.clean_mask_imgs.append(clean_mask_imgs_pair)

        self.noisy_imgs = self.noisy_imgs
        self.length_origin = len(self.noisy_imgs[0])
        self.pairs = len(self.noisy_imgs)

    def load_imgs_rolling(self):
        if self.opt.img_type == '.mrc':
            self.load_function = load_3d_mrc_dataset_origin
        else:
            self.load_function = load_tif_img_origin

        self.noisy_imgs = []
        self.clean_mask_imgs = []

        # shape = []

        noisy_filenames = []
        for dir in self.opt.input_dir:
            noisy_files = sorted([x for x in os.listdir(os.path.join(self.base_dir, dir))
                             if x.find(self.opt.input_img_name) >= 0])
            noisy_filenames += [os.path.join(self.base_dir, dir, x) for x in noisy_files if
                            is_specify_file(x, self.opt.img_type)]

        self.noisy_imgs, self.clean_mask_imgs = [], []
        for i in range(len(noisy_filenames)):
            noisy_imgs = self.load_function(noisy_filenames[i])
            for j in range(noisy_imgs.shape[0]):
                noisy_imgs_t = input_guass_blur(noisy_imgs[i])

                if self.opt.is3D:
                    if self.opt.img_frame != 0:
                        noisy_imgs_t = noisy_imgs_t[:, self.opt.img_frame]

                if not self.opt.input_img_neg:
                    noisy_imgs_t = np.where(noisy_imgs_t < 0, 0, noisy_imgs_t)

                if self.opt.normal:
                        noisy_imgs_t[i] /= np.max(np.abs(noisy_imgs_t[i]))

                noisy_imgs[i] = noisy_imgs_t


            self.noisy_imgs.append(noisy_imgs)

            if self.opt.frontground_filter:
                mask = self.get_mask(noisy_imgs[0], self.opt.fg_th)
                center_list = np.array(np.where(mask == 1)).transpose([1, 0])
                self.clean_mask_imgs.append(center_list)
                # length_mask = len(center_list)
                # _, x, y = clean_imgs.shape


        self.noisy_imgs = self.noisy_imgs
        self.length_origin = len(self.noisy_imgs)

    def get_position_2d(self, get_img):
        while True:
            idx = int(np.random.rand() * self.length_origin)
            clean_imgs, noisy_imgs = get_img(idx=idx)
            nx, ny = noisy_imgs.shape
            center_list = self.clean_mask_imgs[idx]
            length_mask = len(center_list)

            iter = 0
            while True:
                c_x, c_y = center_list[np.int16(length_mask * np.random.rand())]
                c_x = c_x // self.upfactor
                c_y = c_y // self.upfactor
                if c_x - self.noisy_half > 0 and c_x + self.noisy_half < nx and c_y - self.noisy_half > 0 and c_y + self.noisy_half < ny:
                    return clean_imgs, noisy_imgs, c_x, c_y, 0
                if iter > 10:
                    break
                iter += 1

    def get_position_3d(self, get_img): #TODO check
        while True:
            idx = int(np.random.rand() * self.length_origin)
            clean_imgs, noisy_imgs = get_img(idx=idx)
            nz, nx, ny = noisy_imgs.shape
            center_list = self.clean_mask_imgs[idx]
            length_mask = len(center_list)

            iter = 0
            while True:
                c_z, c_x, c_y = center_list[np.int16(length_mask * np.random.rand())]
                c_z = c_z // self.zupfactor
                c_x = c_x // self.upfactor
                c_y = c_y // self.upfactor
                if c_x - self.noisy_half > 0 and c_x + self.noisy_half < nx and \
                        c_y - self.noisy_half > 0 and c_y + self.noisy_half < ny and \
                    c_z - self.nzhalf_left >= 0 and c_z + self.nzhalf_right < nz:
                    return clean_imgs, noisy_imgs, c_x, c_y, c_z
                if iter > 10:
                    break
                iter += 1

    def get_img_gt_in(self, idx):
        return self.clean_imgs[idx], self.noisy_imgs[idx]

    def get_img_prs(self, idx):
        imgs = [self.noisy_imgs[i][idx] for i in range(self.pairs)]
        p1 = np.int8(np.random.rand() * self.pairs)
        p2 = np.int8(np.random.rand() * self.pairs)
        if p2 == p1:
            p2 = (p2 + np.int8(np.random.rand() * (self.pairs-1))) % self.pairs
        return imgs[p1], imgs[p2]

    def get_img_rolling(self, idx):
        imgs = self.noisy_imgs[idx]
        pairs = len(imgs)
        p1 = np.int16(np.random.rand() * pairs)
        p2 = p1 + ((-1) ** np.int16(np.random.rand()*pairs)) * np.ceil(np.random.rand() * self.opt.rolling_step).astype(np.int16)
        if p2 < 0:
            p2 = 0 if p1 != 0 else 1
        elif p2 >= pairs:
            p2 = pairs - 1 if p1 != pairs -1 else pairs-2
        return imgs[p1], imgs[p2]


    def get_crop_from_filter(self, get_img):
        # random choose positon
        clean_imgs, noisy_imgs, c_x, c_y, c_z = self.get_position(get_img)
        random = np.random.rand(6)
        clean_crop_imgs = self.crop_function(clean_imgs, self.clean_crop, xc=c_x * self.upfactor,
                                             yc=c_y * self.upfactor, zc=c_z * self.zupfactor,
                                             random=random, crop_depth=self.crop_depth//self.zupfactor).astype(np.float32)
        noisy_crop_imgs = self.crop_function(noisy_imgs, self.noise_crop, xc=c_x, yc=c_y, zc=c_z,
                                             random=random, crop_depth=self.crop_depth).astype(np.float32)
        return noisy_crop_imgs, clean_crop_imgs

    def get_random_crop(self, get_img):
        idx = int(np.random.rand() * self.length_origin)
        clean_imgs, noisy_imgs = get_img(idx=idx)
        random = np.random.rand(6)
        clean_crop_imgs = self.crop_function(clean_imgs, self.clean_crop, random=random, crop_depth=
            self.crop_depth//self.zupfactor if self.crop_depth is not None and self.zupfactor is not None else None
                                             ).astype(np.float32)
        noisy_crop_imgs = self.crop_function(noisy_imgs, self.noise_crop, random=random, crop_depth=self.crop_depth).astype(np.float32)
        return noisy_crop_imgs, clean_crop_imgs

    def augment_in_cache(self, get_img):
        self.clean_crop_imgs = []
        self.noisy_crop_imgs = []
        for i in range(self.length):
            noisy_crop_imgs, clean_crop_imgs = self.get_crop(get_img)
            self.noisy_crop_imgs.append(noisy_crop_imgs)
            self.clean_crop_imgs.append(clean_crop_imgs)

        self.iters = np.arange(0, self.length)

    def augment_batch(self, get_img):
        clean_crop_imgs = []
        noisy_crop_imgs = []
        for i in range(self.batch_size):
            n, c = self.get_crop(get_img)
            noisy_crop_imgs.append(n)
            clean_crop_imgs.append(c)

        clean_imgs = torch.from_numpy(np.array(clean_crop_imgs)[:, np.newaxis]) # add channel axis
        noisy_imgs = torch.from_numpy(np.array(noisy_crop_imgs)[:, np.newaxis])

        return clean_imgs, noisy_imgs

    def augment_batch_VSR(self, get_img):
        gt_start = (self.crop_depth - self.opt.output_frame) // 2

        clean_crop_imgs = []
        noisy_crop_imgs = []
        for i in range(self.batch_size):
            n, c = self.get_crop(get_img)
            noisy_crop_imgs.append(n)
            clean_crop_imgs.append(c)

        # B T C X Y
        # add channel axis
        clean_imgs = torch.from_numpy(np.array(clean_crop_imgs)[:, gt_start:gt_start + self.opt.output_frame, np.newaxis])
        noisy_imgs = torch.from_numpy(np.array(noisy_crop_imgs)[:, :, np.newaxis])

        return clean_imgs, noisy_imgs

    def augment_batch_swin(self, get_img):
        gt_start = (self.crop_depth - self.opt.output_frame) // 2

        clean_crop_imgs = []
        noisy_crop_imgs = []
        for i in range(self.batch_size):
            n, c = self.get_crop(get_img)
            noisy_crop_imgs.append(n)
            clean_crop_imgs.append(c)

        # B T C X Y
        # add channel axis
        clean_imgs = torch.from_numpy(np.array(clean_crop_imgs)[:, gt_start:gt_start + self.opt.output_frame])
        noisy_imgs = torch.from_numpy(np.array(noisy_crop_imgs))

        return clean_imgs, noisy_imgs

    def get_data_from_augment_in_cache(self, i):
        if self.shuffle and i == 0:
            np.random.shuffle(self.iters)
        if (i + 1) * self.batch_size < self.length:
            ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
        else:
            ilist = self.iters[i * self.batch_size: self.length]

        clean_imgs = np.array([self.clean_crop_imgs[x] for x in ilist])
        noisy_imgs = np.array([self.noisy_crop_imgs[x] for x in ilist])

        clean_imgs = torch.from_numpy(clean_imgs[:, np.newaxis]) # add channel axis
        noisy_imgs = torch.from_numpy(noisy_imgs[:, np.newaxis])

        return clean_imgs, noisy_imgs

    def get_len(self):
        length = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            length += 1
        return length

    def get_full_len(self):
        return self.length

    # def get_filename(self, i):
    #     if (i + 1) * self.batch_size < self.length:
    #         ilist = self.iters[i * self.batch_size: (i + 1) * self.batch_size]
    #     else:
    #         ilist = self.iters[i * self.batch_size: self.length]
    #     ##resize to file length
    #     ilist = ilist // self.augment_size
    #     return [self.clean_files[x] for x in ilist], [self.noisy_files[x] for x in ilist]
