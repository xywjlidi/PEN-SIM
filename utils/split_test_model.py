import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import tifffile
import os

def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


## UNET: input mod 32 == 0
#
# split 1536
#    4 4; 8 2; 16 1
# => 480; 288; 192
#
# split 1024
#
def split_test_model_3d(model, device, img, split_num, split_padding, upfactor=1):
    nz, nx, ny = img.shape
    if nz == 1:
        return
    inte_x = nx // split_num
    inte_y = ny // split_num
    pad_x = inte_x // split_padding
    pad_y = inte_y // split_padding

    output = np.zeros(img.shape, np.dtype(np.float32))
    for i in range(split_num):
        for j in range(split_num):
            # position in total output
            a = i * inte_x
            b = i * inte_x + inte_x
            c = j * inte_y
            d = j * inte_y + inte_y
            # cut off partition output
            x, y, z, w = [+ pad_x, - pad_x, + pad_y, - pad_y]
            # check edge
            if i == 0:
                x = 0
            elif i == inte_x:
                y = 0
            if j == 0:
                z = 0
            elif j == inte_y:
                w = 0
            # cut off partition input
            e, f, g, h = [a - x, b - y, c - z, d - w]
            # test model
            input_ = img[np.newaxis, np.newaxis, :, e: f, g: h]
            # print(input_.shape)
            out = model(torch.tensor(input_).to(device)).squeeze().cpu().numpy()

            # test splice
            # out = img[:, e: f, g: h]
            # print(out.shape)

            a *= upfactor
            b *= upfactor
            c *= upfactor
            d *= upfactor
            x *= upfactor
            y *= upfactor

            output[:, a:b, c:d] = out[:, x:x + b - a, z:z + d - c]

    return output


def kernel_test_model_3d_prctile(model, device, img, kernel, padding, upfactor=1, zupfactor=1, min_prc=0, max_prc=100):
    nz, nx, ny = img.shape
    # if nz == 1:
    #     return
    kx, ky, kz = kernel
    px, py, pz = padding
    # if padding == 0:
    #     padding = kernel // 8
    embed_x = kx - 2 * px
    embed_y = ky - 2 * py
    embed_z = kz - 2 * pz

    split_x = math.ceil(nx / embed_x)
    split_y = math.ceil(ny / embed_y)
    split_z = math.ceil(nz / embed_z)

    margin_x = split_x * embed_x + 2 * px
    margin_y = split_y * embed_y + 2 * py
    margin_z = split_z * embed_z + 2 * pz
    img_margin = np.zeros((1, 1, margin_z, margin_x, margin_y))
    img_margin[:, :, pz:pz + nz, px:px + nx, py:py + ny] = img

    output = torch.zeros((margin_z * zupfactor, margin_x * upfactor, margin_y * upfactor))

    for i in range(split_x):
        for j in range(split_y):
            for k in range(split_z):
                # test splice
                # out = img[zd:zd + kz, xd: xd + kx, yd: yd + ky]
                # if out.shape != (kz, kx, ky):
                #     print("nz: %d row %d col %d shape %s" % (k, i, j, out.shape))
                # if f - e != kernel or h - g != kernel:
                #     print("row %d col %d shape %s" % (i, j, out.shape))
                # print(input_.shape)
                # print(out.shape)

                # test model
                input_ = prctile_norm(img_margin[:, :, k * embed_z:k * embed_z + kz,
                                      i * embed_x: i * embed_x + kx,
                                      j * embed_y: j * embed_y + ky],
                                      min_prc, max_prc)

                out = model(torch.FloatTensor(input_).to(device))
                out = out.squeeze().cpu()

                output[(pz + k * embed_z) * zupfactor:(pz + (k + 1) * embed_z) * zupfactor,
                (px + i * embed_x) * upfactor:(px + (i + 1) * embed_x) * upfactor,
                (py + j * embed_y) * upfactor:(py + (j + 1) * embed_y) * upfactor] = \
                    out[pz * zupfactor:(kz - pz) * zupfactor,
                    px * upfactor:(kx - px) * upfactor, py * upfactor:(ky - py) * upfactor]

    return output[pz * zupfactor:(pz + nz) * zupfactor, px * upfactor:(px + nx) * upfactor,
           py * upfactor:(py + ny) * upfactor]


def kernel_test_model_3d(model, device, img, kernel, padding, upfactor=1, zupfactor=1):
    # save_path = './img/localization/300nm-degradation-100nm-TFM/test_split'

    nz, nx, ny = img.shape
    # if nz == 1:
    #     return
    kx, ky, kz = kernel
    px, py, pz = padding
    # if padding == 0:
    #     padding = kernel // 8
    embed_x = kx - 2 * px
    embed_y = ky - 2 * py
    embed_z = kz - 2 * pz

    split_x = math.ceil(nx / embed_x)
    split_y = math.ceil(ny / embed_y)
    split_z = math.ceil(nz / embed_z)

    margin_x = split_x * embed_x + 2 * px
    margin_y = split_y * embed_y + 2 * py
    margin_z = split_z * embed_z + 2 * pz
    img_margin = np.zeros((1, 1, margin_z, margin_x, margin_y))

    # fill in frame with zero
    for f_i in range(pz):
        img_margin[:, :, f_i, px:px + nx, py:py + ny] = img[0]
    for f_i in range(pz + nz, margin_z):
        img_margin[:, :, f_i, px:px + nx, py:py + ny] = img[nz - 1]
    img_margin[:, :, pz:pz + nz, px:px + nx, py:py + ny] = img

    output = torch.zeros((margin_z * zupfactor, margin_x * upfactor, margin_y * upfactor))

    for i in range(split_x):
        for j in range(split_y):
            for k in range(split_z):
                # test splice
                # out = img[zd:zd + kz, xd: xd + kx, yd: yd + ky]
                # if out.shape != (kz, kx, ky):
                #     print("nz: %d row %d col %d shape %s" % (k, i, j, out.shape))
                # if f - e != kernel or h - g != kernel:
                #     print("row %d col %d shape %s" % (i, j, out.shape))
                # print(input_.shape)
                # print(out.shape)

                # test model
                input_ = img_margin[:, :, k * embed_z:k * embed_z + kz,
                         i * embed_x: i * embed_x + kx,
                         j * embed_y: j * embed_y + ky]

                # tifffile.imsave(os.path.join(save_path,"x_{}_y_{}_z{}_{}.tif".format(str(i),str(j),str(k),"in")), input_[0,0], dtype=np.float32)

                out = model(torch.FloatTensor(input_).to(device))
                out = out.squeeze().cpu()

                # tifffile.imsave(os.path.join(save_path, "x_{}_y_{}_z{}_{}.tif".format(str(i),str(j),str(k), "out")), out.numpy(), dtype=np.float32)

                output[(pz + k * embed_z) * zupfactor:(pz + (k + 1) * embed_z) * zupfactor,
                (px + i * embed_x) * upfactor:(px + (i + 1) * embed_x) * upfactor,
                (py + j * embed_y) * upfactor:(py + (j + 1) * embed_y) * upfactor] = \
                    out[pz * zupfactor:(kz - pz) * zupfactor,
                    px * upfactor:(kx - px) * upfactor, py * upfactor:(ky - py) * upfactor]

    return output[pz * zupfactor:(pz + nz) * zupfactor, px * upfactor:(px + nx) * upfactor,
           py * upfactor:(py + ny) * upfactor]


def kernel_test_model_2d(model, device, img, kernel, overlap, upfactor=1, norm=False, min_prc=0, max_prc=100):
    if img.ndim == 3:
        nz, nx, ny = img.shape
    else:
        nx, ny = img.shape
        nz = 1
    # if nz == 1:
    #     return
    kx, ky = kernel
    px, py = overlap
    # if padding == 0:
    #     padding = kernel // 8
    embed_x = kx - 2 * px
    embed_y = ky - 2 * py

    split_x = math.ceil(nx / embed_x)
    split_y = math.ceil(ny / embed_y)

    margin_x = split_x * embed_x + 2 * px
    margin_y = split_y * embed_y + 2 * py

    img_margin = np.zeros((1, nz, margin_x, margin_y))
    img_margin[:, :, px:px + nx, py:py + ny] = img

    output = torch.zeros((margin_x * upfactor, margin_y * upfactor))

    for i in range(split_x):
        for j in range(split_y):
            # test model
            input_ = img_margin[:, :, i * embed_x: i * embed_x + kx, j * embed_y: j * embed_y + ky]
            if norm:
                input_ = prctile_norm(input_, min_prc, max_prc)

            out = model(torch.FloatTensor(input_).to(device))
            out = out.squeeze().cpu()

            output[(px + i * embed_x) * upfactor:(px + (i + 1) * embed_x) * upfactor,
            (py + j * embed_y) * upfactor:(py + (j + 1) * embed_y) * upfactor] = \
                out[px * upfactor:(kx - px) * upfactor, py * upfactor:(ky - py) * upfactor]

    return output[px * upfactor:(px + nx) * upfactor, py * upfactor:(py + ny) * upfactor]


if __name__ == '__main__':
    import numpy as np
    from utils.read_utils import read_mrc_with_hd
    import matplotlib.pyplot as plt

    a, _ = read_mrc_with_hd(
        '/mnt/Public/NL-SIM2/20220723-NL/cos7-2xkohinoor2.0-lifeact/c2-continue-488-300mw-12p-5ms-405-4p-5ms-1.41NA-1scyc-0.1p-250inten_20220724_002448/')
    b = kernel_test_model_3d(_, _, a, 217, 2)
    plt.figure()
    plt.imshow(b, 'gray')
