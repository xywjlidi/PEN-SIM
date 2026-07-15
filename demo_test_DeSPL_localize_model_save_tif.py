#########
#   Test demo for localization
#   model list:
#       UNet_localization3d_out2_b100_a0.01_min2_up3_20230411_refix
#           localization for exLambda 560 100nm beads
#   author: chy
#   date: 2024.4.22
#########

import os
import torch
import numpy as np
import tifffile
from utils.test_utils import get_localization_model_config
from utils.read_mrc import read_mrc_with_hd
from utils.split_test_model import kernel_test_model_3d
from utils.read_utils import load_recon_no_negative, load_recon_both, \
    is_specify_file, prctile_norm, mkdir, load_tif_img_origin, save_nz_tiff
from utils.test_utils import load_model, timeblock
import glob

if __name__ == "__main__":
    # 1.config
    # cpu/gpu device config
    # TODO: "cpu" or "cuda:n"
    device = torch.device("cuda:0")

    # model config
    # choose right beads: "80nm", "100nm", "120nm", "150nm", "240nm"
    env = "100nm"

    # test img config
    # img path
    # TODO: img should be (1)stabilized and then (2)processed by beadsimfilter.m.
    # TODO: change input path
    filelist = glob.glob(r'../demo_data/100nm_beads.tif')

    # save path
    save_to_input_path = True

    # output filename recommended when set save_to_input_path to True
    # save_dir_name = False
    # save_img_name = True
    # save_tag_name = False
    # tag = "560"

    # output filename recommended when set save_to_input_path to False
    # save_dir_name = True
    # save_img_name = False
    # save_tag_name = False
    # tag = "560"

    # input img start end step
    start = 0
    frame = 0  # 0= do predict to the end of img
    interval = 1

    # flag of save input and output
    # save_input = True
    # save_output = True

    # input img normalization config
    with_neg = False
    stretch_flag = True
    stretch_lower = 0
    stretch_upper = 100

    # tiling input img
    split_test = True
    cropSize = 256
    crop_depth = 10
    padding = 32
    padding_depth = 1

    # 2.test
    with torch.no_grad():
        log_dir, param, model_type = get_localization_model_config(env)
        model, upfactor, arch = param
        Net = load_model(arch)()

        path = os.path.join(log_dir, model, model_type + '.pth')

        weights = torch.load(path, map_location=device)
        Net.load_state_dict(weights['state_dict'])
        Net.to(device)

        for f in filelist:
            with timeblock("**********************************\n time for localizating %s : " % f):
                # load mrc
                if is_specify_file(f, '.mrc'):
                    img, _ = read_mrc_with_hd(f)
                # load tif
                elif is_specify_file(f, '.tif'):
                    img = load_tif_img_origin(f)
                    # TODO: a tr
                    if img.ndim == 2:
                        img = img[np.newaxis]
                else:
                    continue

                if with_neg:
                    img = load_recon_both(np.float32(img))
                else:
                    img = load_recon_no_negative(np.float32(img))

                nz, nx, ny = img.shape
                nt = np.int16(np.ceil(nz / interval))
                if frame != 0:
                    nt = frame
                img_clip = np.zeros((nt, nx, ny), np.float32)
                for t in range(nt):
                    img_clip[t] = img[(t + start) * interval]

                if stretch_flag:
                    img_clip = prctile_norm(img_clip, stretch_lower, stretch_upper)

                if split_test:
                    result = kernel_test_model_3d(Net, device, img_clip, (cropSize, cropSize, crop_depth),
                                                  (padding, padding, padding_depth),
                                                  upfactor=upfactor)
                else:
                    result = Net(torch.from_numpy(img_clip[np.newaxis, np.newaxis]).to(device))

                result = result.cpu().numpy().squeeze()

                save_nz_tiff(f[0:-4] + "_" + env + ".tif", np.float32(result))
