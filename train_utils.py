import os
import torch
from read_utils.read_utils import save_nz_tiff, save_tiff, load_3d_mrc_dataset, load_mrc_dataset_origin,\
    load_tif_img, is_specify_file, load_tif_img_origin, load_3d_mrc_dataset_origin, load_tif_2d_img_origin
from utils.split_test_model import kernel_test_model_3d, kernel_test_model_2d
import numpy as np

def mkdirs(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(epoch, iter, state_dict, optimizer, path):
    torch.save({'epoch': epoch,
                'iter': iter,
                'state_dict': state_dict,
                'optimizer': optimizer}, path)

def load_config(name):
    r"""
    A dirty hack to load a module from a string input

    Returns:
        A pointer to the loaded module
    """
    strCmd = "from config import " + name + " as config"
    exec(strCmd)
    return eval('config')

def load_dataset(name):
    r"""
    A dirty hack to load a module from a string input

    Returns:
        A pointer to the loaded module
    """
    strCmd = "from utils.dataset import " + name + " as Dataset"
    exec(strCmd)
    return eval('Dataset')


def load_model(name):
    r"""
    A dirty hack to load a module from a string input

    Returns:
        A pointer to the loaded module
    """
    strCmd = "from model.model import " + name + " as UNet"
    exec(strCmd)
    return eval('UNet')

def load_config(name):
    r"""
    A dirty hack to load a module from a string input

    Returns:
        A pointer to the loaded module
    """
    strCmd = "from config import " + name + " as config"
    exec(strCmd)
    return eval('config')


def test_and_save(model, device, iter, opt, img_path, save_img, save_name='default', img_dir=None):
    if img_dir is None:
        img_dir = os.path.join("log", opt.env_name, "img")

    with torch.no_grad():
        if is_specify_file(save_img, ".mrc"):
            if opt.is3D:
                load_function = load_3d_mrc_dataset_origin
            else:
                load_function = load_mrc_dataset_origin
        else:
            if opt.is3D:
                load_function = load_tif_img_origin
            else:
                load_function = load_tif_2d_img_origin
        input_t = load_function(os.path.join(img_path,save_img))

        if not opt.input_img_neg:
            input_t = np.where(input_t < 0, 0, input_t)
        if opt.normal:
            input_t = input_t / np.max(np.abs(input_t))

        # test split input img
        if opt.split_test > 1:
            if input_t.ndim == 2:
                re = kernel_test_model_2d(model, device, input_t, kernel=(512, 512),
                                      overlap=(64, 64),
                                      upfactor=opt.upfactor)
            elif input_t.ndim == 3:
                re = kernel_test_model_3d(model, device, input_t, kernel=(512, 512, 12),
                                          padding=(64, 64, 0),
                                          upfactor=opt.upfactor)
        else:
            input_t = torch.tensor(input_t[:, np.newaxis]).to(device)
            re = model(input_t)
            re = re.squeeze().cpu().numpy()
            input_t = input_t.squeeze().cpu().numpy()

        if len(re.shape) == 3:
            save_function = save_nz_tiff
        else:
            save_function = save_tiff

        filename = os.path.splitext(save_img)[0]+'_Res_it' + str(iter + 1).zfill(6) + '.tif'
        if save_name != 'default':
            filename = os.path.splitext(save_img)[0] + '_' + save_name + '.tif'
        save_function(os.path.join(img_dir, filename), re)
        if iter + 1 == opt.save_checkpoint:
            save_function(os.path.join(img_dir, os.path.splitext(save_img)[0]+'_val_input.tif'), input_t)
            if opt.gt_dir is not None:
                target_t = load_function(os.path.join(opt.val_dir, opt.gt_dir, save_img))
                save_function(os.path.join(img_dir, os.path.splitext(save_img)[0]+'_val_gt.tif'), target_t)