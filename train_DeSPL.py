import os
import datetime
import torch
import torch.optim as optim
import pickle
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import time
from train_utils import mkdir, save_checkpoint, load_config, load_model, load_dataset
from utils.loss import Conv_OTF_MSE, L1Norm, batch_PSNR, Conv_Kernel_MSE, get_gauss_kernel, Logsum, \
    batch_PSNR_linear_transform, Conv_Kernel_MSE_3D, batch_PSNR_linear_transform_3D
from read_utils.read_utils import save_nz_tiff, save_tiff, load_3d_mrc_dataset, load_tif_img
from utils.split_test_model import kernel_test_model_2d, kernel_test_model_3d
import argparse
import scipy.io as scio
import numpy as np
import sys


def test_and_save(model, device, iter, opt, save_name='default'):
    with torch.no_grad():
        if opt.img_type == '.mrc':
            load_function = load_3d_mrc_dataset
        else:
            load_function = load_tif_img
        input_t = load_function(os.path.join(opt.val_dir, opt.input_dir, opt.save_img), opt.input_img_neg)

        if opt.split_test == 1:
            re = kernel_test_model_3d(model, device, input_t, kernel=(opt.noise_crop, opt.noise_crop, 8),
                                          padding=(16, 16, 0),
                                          upfactor=opt.upfactor)
            re = re.numpy()
        elif opt.split_test == 0:
            input_t = torch.tensor(input_t[np.newaxis, np.newaxis, :, :]).to(device)
            re = model(input_t)
            re = re.squeeze().cpu().numpy()
            input_t = input_t.squeeze().cpu().numpy()

        if len(re.shape) == 3:
            save_function = save_nz_tiff
        else:
            save_function = save_tiff

        filename = 'Loc_it' + str(iter + 1).zfill(6) + '.tif'
        if save_name != 'default':
            filename = save_name + '.tif'
        save_function(
            os.path.join(img_dir, filename), re)
        if iter + 1 == opt.save_checkpoint:
            target_t = load_function(os.path.join(opt.val_dir, opt.gt_dir, opt.save_img), opt.input_img_neg)
            save_function(os.path.join(img_dir, 'val_gt.tif'), target_t)
            save_function(os.path.join(img_dir, 'val_input.tif'), input_t)

    #     save_checkpoint(epoch + 1, iter, UNet.state_dict(), optimizer.state_dict(),
    #                     os.path.join(log_dir, "model_epoch_{}.pth".format(iter + 1)))

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    prog = argparse.ArgumentParser()
    prog.add_argument('--opt', type=str)
    args = prog.parse_args()

    opt = load_config(args.opt)._C
    device = torch.device(opt.device)

    log_dir = os.path.join("./log", opt.env_name)
    print(opt.env_name)
    # model_dir = os.path.join(log_dir,"model")
    if os.path.isfile(os.path.join(log_dir, 'model_best.pth') and not opt.resume):
        in_str = input(opt.env_name + " model is trained.\noverwrite it? (y/n)")
        if in_str != "y":
            sys.exit(1)
    mkdir(log_dir)
    img_dir = os.path.join(log_dir, "img")
    mkdir(img_dir)

    log_name = os.path.join(log_dir, datetime.datetime.now().isoformat()[:13] + "log.txt")
    with open(log_name, 'a') as f:
        f.write(str(vars(opt)))

    #####1 Load Data
    Dataset = load_dataset(opt.dataset)
    train_dataset = Dataset(opt.train_dir, opt)
    train_length = train_dataset.get_len()

    opt.augment_size = opt.val_augment_size
    val_dataset = Dataset(opt.val_dir, opt)
    val_length = val_dataset.get_len()
    print("env: %s train: %s val: %s" % (opt.env_name, opt.train_dir, opt.val_dir))
    print("start train...")
    with open(log_name, 'a') as f:
        f.write("env: %s train: %s val: %s" % (opt.env_name, opt.train_dir, opt.val_dir))
        f.write("start train...")

    #####2 Init Model
    UNet = load_model(opt.arch)()
    UNet.apply(initialize_weights)
    # if opt.arch == "UNet_scheme1":
    #     UNet = UNet_scheme1()
    # elif opt.arch == "UNet":
    #     UNet = UNet()
    # elif opt.arch == "UNet_scheme2":
    #     UNet = UNet_scheme2()
    # elif opt.arch == "AutoEncoder":
    #     UNet = AutoEncoder()
    # elif opt.arch == "EncoderDecoder_scheme2":
    #     UNet = EncoderDecoder_scheme2()
    UNet.to(device)
    optimizer = optim.AdamW(UNet.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    start_epoch = 0
    # criterion
    # OTF = torch.from_numpy(scio.loadmat(opt.OTF_path)['OTF']).cuda()
    if opt.kernel == 'psf':
        PSF = scio.loadmat(opt.OTF_path)['PSF']
        L = opt.kernel_size
        H, W = PSF.shape
        start = (H - L) // 2 + 1
        kernel = PSF[start:start + L, start:start + L]
    elif opt.kernel == 'gauss':
        kernel = get_gauss_kernel(opt.kernel_size, opt.sigma, opt.dxy)
    elif opt.kernel == 'none':
        kernel = np.ones((1, 1))
    kernel = torch.FloatTensor(kernel[np.newaxis, np.newaxis, :, :]).to(device)

    criterion1 = Conv_Kernel_MSE_3D

    if opt.penalty == 'logsum':
        criterion2 = Logsum
    else:
        criterion2 = L1Norm

    ####Load weights
    if opt.resume:
        pretrained = torch.load(os.path.join(log_dir,"model_latest.pth"))
        UNet.load_state_dict(pretrained['state_dict'])
        start_epoch = pretrained['epoch']
        optimizer.load_state_dict(pretrained['optimizer'])
        for p in optimizer.param_groups: lr = p['lr']

    if opt.scheduler == 'steplr':
        scheduler = StepLR(optimizer, step_size=opt.decay_step, last_epoch=start_epoch - 1, gamma=0.5)
    #####3 Training...
    pklProfile = {"loss_mse": [], "loss_l1": [], "loss": [], "psnr": [], "lr": [], "val_loss": [], "val_psnr": [],
                  "best_epoch": 0, "best_iter": 0, "best_psnr": 0}
    iter = 0
    for epoch in range(opt.nepoch):
        start_time = time.time()
        for iter_ in range(train_length):
            optimizer.zero_grad()
            target, input_ = train_dataset.get_data(iter_)
            target = target.to(device)
            input_ = input_.to(device)
            restored = UNet(input_)
            loss_mse = criterion1(restored, target, kernel)
            loss_l1 = criterion2(restored)
            loss = loss_mse + opt.alpha * loss_l1
            loss.backward()
            optimizer.step()

            pklProfile["lr"].append(optimizer.state_dict()['param_groups'][0]['lr'])
            pklProfile["loss_mse"].append(loss_mse.item())
            pklProfile["loss_l1"].append(loss_l1.item())
            pklProfile["loss"].append(loss.item())
            # pklProfile["psnr"].append(batch_PSNR(restored, target).item()/opt.batch_size)

            if (iter + 1) % opt.val_checkpoint == 0:
                with torch.no_grad():
                    val_loss = 0
                    val_psnr = 0
                    for j in range(val_length):
                        target_t, input_t = val_dataset.get_data(j)
                        target_t = target_t.to(device)
                        input_t = input_t.to(device)
                        re = UNet(input_t)
                        val_loss += criterion1(re, target_t, kernel,test=True).item() + opt.alpha * criterion2(restored).item()
                        val_psnr += batch_PSNR(re, target_t).item()
                        #val_psnr += batch_PSNR_linear_transform(re, target_t, device).item()
                    val_psnr /= val_dataset.get_full_len()

                    pklProfile["val_loss"].append(val_loss)
                    pklProfile["val_psnr"].append(val_psnr)

                    if val_psnr > pklProfile["best_psnr"]:
                        pklProfile["best_epoch"] = epoch + 1
                        pklProfile["best_iter"] = iter + 1
                        pklProfile["best_psnr"] = val_psnr
                        save_checkpoint(epoch + 1, iter + 1, UNet.state_dict(), optimizer.state_dict(),
                                        os.path.join(log_dir, "model_best.pth"))
                        if iter > opt.warmup_iter:
                            test_and_save(UNet, device, iter, opt, save_name='model_best')
                    print("Ep %d it %d  \tPSNR: %.4f\tVal Loss: %.4f\t\t\tbest_ep %d\tbest_it %d\tbest_psnr %.4f" %
                          (epoch + 1, iter + 1, val_psnr, val_loss, pklProfile["best_epoch"], pklProfile["best_iter"],
                           pklProfile["best_psnr"]))
                    with open(log_name, 'a') as f:
                        f.write(
                            "Ep %d it %d  \tPSNR: %.4f\tVal Loss: %.4f\t\t\tbest_ep %d\tbest_it %d\tbest_psnr %.4f\n" %
                            (
                                epoch + 1, iter + 1, val_psnr, val_loss, pklProfile["best_epoch"],
                                pklProfile["best_iter"],
                                pklProfile["best_psnr"]))

                if (iter + 1) % opt.save_checkpoint == 0:
                    test_and_save(UNet, device, iter, opt)
                    # save_checkpoint(epoch + 1, iter + 1, UNet.state_dict(), optimizer.state_dict(),
                    #                 os.path.join(log_dir, "model_iter" + str(iter + 1).zfill(6) + ".pth"))
                if opt.kernel == 'gauss' and (iter + 1) % opt.change_step == 0:
                    if opt.sigma > opt.sigma_min:
                        opt.sigma *= opt.gamma
                        if opt.sigma < opt.sigma_min:
                            opt.sigma = opt.sigma_min
                        with open(log_name, 'a') as f:
                            f.write("change sigma to: " + str(opt.sigma))
                        print("change sigma to: " + str(opt.sigma))
                        kernel = get_gauss_kernel(opt.kernel_size, opt.sigma, opt.dxy)
                        kernel = torch.FloatTensor(kernel[np.newaxis, np.newaxis, :, :]).to(device)
                        # with open(log_name, 'a') as f:
                        #     f.write("no guass filter")
                        # print("no guass filter")
                        # kernel = torch.ones((1, 1, 1, 1)).to(device)

            iter += 1
        if opt.scheduler == 'steplr':
            scheduler.step()
        print("-----------------")
        print("Epoch: %d\tTime: %.4f\tLearningRate: %.7f" % (
            epoch + 1, time.time() - start_time, optimizer.state_dict()['param_groups'][0]['lr']))
        print("-----------------")
        with open(log_name, 'a') as f:
            f.write("-----------------\n")
            f.write("Epoch: %d\tTime: %.4f\tLearningRate: %.7f\n" % (
                epoch + 1, time.time() - start_time, optimizer.state_dict()['param_groups'][0]['lr']))
            f.write("-----------------\n")
        with open(os.path.join(log_dir, "model_loss.pkl"), 'wb') as handle:
            pickle.dump(pklProfile, handle, protocol=pickle.HIGHEST_PROTOCOL)

        save_checkpoint(opt.nepoch, 0, UNet.state_dict(), optimizer.state_dict(),
                        os.path.join(log_dir, "model_latest.pth"))
