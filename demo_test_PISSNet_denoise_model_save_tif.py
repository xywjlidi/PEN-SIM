#########
#   Test demo for PRSSIM 2D denoise
#   model list:
#       2D single structure(488 channel):
#           "lifeact", "paxillin", "ensconsin", "sec61", "omm", "lamp1", "phb2"
#       2D double structure(488 channel, 560 channel fret reconstructed):
#           "lifeact-paxillin", "omm-ensconsin"
#       3D single structure(488 channel):
#           "ensconsin-3D", "laminb-3D"
#   author: chy
#   date: 2023.8.8
#########

import os
import torch
import numpy as np
from utils.read_utils import load_origin_img, save_nz_tiff, save_tiff
from utils.split_test_model import kernel_test_model_2d, kernel_test_model_3d
from utils.test_utils import load_model, get_prssim_model_config, timeblock
import glob

if __name__ == "__main__":
    # 1.config
    # cpu/gpu device config
    # TODO: "cpu" or "cuda:n"
    device = torch.device("cuda:0")

    # model config
    # model exist:
    # 2D single structure(488 channel):
    #           "lifeact", "paxillin", "ensconsin", "sec61", "omm", "lamp1", "phb2", "skl"(rolling default),"skl-prssim"
    # 2D double structure(488 channel, 560 channel fret reconstructed):
    #           marker                  488 channel         560 channel
    #           lifeact&paxillin-fret:  "lifeact-paxillin", "paxillin-fret"
    #           ensconsin&omm-fret:     "omm-ensconsin",    “omm-fret"
    #           ensconsin&lamp1-fret:   "ensconsin",        "ensconsin"
    #           skl&lamp1-fret:         “lamp1-skl",        "lamp1-skl"
    # 3D single structure(488 channel):
    #           "ensconsin-3D", "omm-3D", "lamp1-3D", "laminb-3D","lifeact-3D", "sec61-skl-3D"(used as sec61)
    # 3D double structure(488 channel):
    #           "sec61-skl-3D"
    # other model for comparison:
    # 3D single structure(488 channel):
    #           "lifeact-3D-Neighbor2Neighbor","lifeact-2D-Neighbor2Neighbor", "lifeact-N2V"
    # TODO: change model
    env = "lifeact"



    # TODO: change input path
    # img_pattern in unix file filename pattern matching
    filelist = glob.glob(r"../demo_data/Lifeact.tif")
    save_to_input_path = True

    # predicting img step
    test_interval = 1
    kernel2D = (512,512)
    padding2D = (64,64)
    kernel3D = (512,512,12)
    padding3D = (64,64,2)

    # 2.predict
    with torch.no_grad():
        # load model
        path, model, arch, model_type = get_prssim_model_config(env)
        Net = load_model(arch)()
        path = os.path.join('./', path, model, model_type + '.pth')

        weights = torch.load(path, map_location=device)
        Net.load_state_dict(weights['state_dict'])
        Net.to(device)
        print("model loaded: %s" % str(model))

        # match filename pattern
        for f in filelist:
            with timeblock("**********************************\n time for denoising %s : " % f):
                # load img
                img = load_origin_img(f)
                save_name = f[0:-4] + "_denoise_" + str(model) + ".tif"
                if len(save_name.split('\\')[-1]) > 20:
                    save_name = f[0:-4] + "_denoise_" + str(env) + ".tif"

                assert img.ndim == 2 or img.ndim == 3 or img.ndim == 4

                if img.shape[0] == 1:
                    img = img[0]
                # N2V and N2N need normal
                if env.find("N2V") > -1 or env.find("Neighbor") > -1:
                    img = img / np.max(np.abs(img))

                # do predict
                if env.find("3D") == -1:
                    if img.ndim == 2:
                        restored_t = kernel_test_model_2d(Net, device, img, kernel=kernel2D, overlap=padding2D, upfactor=1)
                        restored_t = restored_t.cpu().numpy().squeeze()

                        # save output
                        if f.endswith('mrc'):
                            save_tiff(save_name, np.rot90(np.float32(restored_t),1,(0,1)))
                        else:
                            save_tiff(save_name, np.float32(restored_t))

                    elif img.ndim == 3:
                        nz, nx, ny = img.shape
                        nt_aft_interval = nz // test_interval
                        if nt_aft_interval < 1:
                            nt_aft_interval = 1
                        restored_t = np.zeros((nt_aft_interval, nx, ny), np.float32)
                        # read stack img by step
                        for t in range(nt_aft_interval):
                            clip = img[t * test_interval]
                            result_t = kernel_test_model_2d(Net, device, clip, kernel=kernel2D, overlap=padding2D, upfactor=1)
                            result_t = result_t.cpu().numpy().squeeze()
                            restored_t[t] = result_t

                        # save output
                        if f.endswith('mrc'):
                            save_nz_tiff(save_name, np.rot90(np.float32(restored_t),1,(1,2)))
                        else:
                            save_nz_tiff(save_name, np.float32(restored_t))
                    elif img.ndim == 4:
                        nt, nz, nx, ny = img.shape
                        nt_aft_interval = nt // test_interval
                        if nt_aft_interval < 1:
                            nt_aft_interval = 1
                        restored_t = np.zeros((nt_aft_interval * nz, nx, ny), np.float32)
                        # read stack img by step
                        for t in range(nt_aft_interval):
                            for iz in range(nz):
                                clip = img[t * test_interval, iz]
                                result_t = kernel_test_model_2d(Net, device, clip, kernel=kernel2D, overlap=padding2D, upfactor=1)
                                result_t = result_t.cpu().numpy().squeeze()
                                restored_t[t*nz+iz] = result_t
                        # save output
                        if f.endswith('mrc'):
                            save_nz_tiff(save_name, np.rot90(np.float32(restored_t),1,(1,2)))
                        else:
                            save_nz_tiff(save_name, np.float32(restored_t))
                    else:
                        print("data dimension dont in 2,3 or 4")
                else:
                    if img.ndim == 2:
                        print("Img dimension do not equal 3: "+ f)
                    elif img.ndim == 3:
                        nz, nx, ny = img.shape
                        if test_interval != 1:
                            print("warning: predict 3D image by step "+str(test_interval))
                            nt_aft_interval = nz // test_interval
                            if nt_aft_interval < 1:
                                nt_aft_interval = 1
                            img_clip = np.zeros((nt_aft_interval, nx, ny), np.float32)
                            # read stack img by step
                            for t in range(nt_aft_interval):
                                img_clip[t] = img[t * test_interval]
                            img = img_clip

                        restored_t = kernel_test_model_3d(Net, device, img, kernel=kernel3D, padding=padding3D,
                                                        upfactor=1)
                        restored_t = restored_t.cpu().numpy().squeeze()

                        # save output
                        if f.endswith('mrc'):
                            save_nz_tiff(save_name, np.rot90(np.float32(restored_t),1,(1,2)))
                        else:
                            save_nz_tiff(save_name, np.float32(restored_t))
                    elif img.ndim == 4:
                        nt, nz, nx, ny = img.shape
                        nt_aft_interval = nt // test_interval
                        if nt_aft_interval < 1:
                            nt_aft_interval = 1

                        restored_t = np.zeros((nt_aft_interval * nz, nx, ny), np.float32)
                        # read stack img by step
                        for t in range(nt_aft_interval):
                            clip = img[t * test_interval]
                            result_t = kernel_test_model_3d(Net, device, clip,kernel=kernel3D, padding=padding3D,
                                                            upfactor=1)
                            result_t = result_t.cpu().numpy().squeeze()
                            restored_t[t * nz: (t+1)*nz] = result_t

                        # save output
                        if f.endswith('mrc'):
                            save_nz_tiff(save_name, np.rot90(np.float32(restored_t),1,(1,2)))
                        else:
                            save_nz_tiff(save_name, np.float32(restored_t))
                    else:
                        print("data dimension dont in 2,3 or 4")
