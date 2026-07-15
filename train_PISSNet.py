import os
import datetime
import torch
import torch.optim as optim
import pickle
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, MultiStepLR
import time
from train_utils import mkdir, save_checkpoint, load_config, load_model, load_dataset
from utils.loss import CharbonnierLoss, batch_PSNR
# import config.train_unet_3d_2DNLSIM_time_lapse_supervise_denoise as config
from read_utils.read_utils import is_specify_file, load_3d_mrc_dataset_origin, load_mrc_dataset_origin,\
    load_tif_2d_img_origin, load_tif_img_origin, save_tiff, save_nz_tiff
from utils.split_test_model import kernel_test_model_3d, kernel_test_model_2d
import sys
import numpy as np
import argparse


# import wandb

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

        ## differ from standard
        if opt.is3D:
            input_t = input_t[:-1]

        # test split input img
        if opt.split_test > 1:
            if input_t.ndim == 2:
                re = kernel_test_model_2d(model, device, input_t, kernel=(512, 512),
                                      overlap=(64, 64),
                                      upfactor=opt.upfactor)
            elif input_t.ndim == 3:
                re = kernel_test_model_3d(model, device, input_t, kernel=(512, 512, 12),
                                          padding=(64, 64, 2),
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
    print("env: %s train: %s input: %s" % (opt.env_name, opt.train_dir, opt.input_dir))
    # model_dir = os.path.join(log_dir,"model")
    if os.path.isfile(os.path.join(log_dir, 'model_best.pth') and not opt.resume):
        in_str = input(opt.env_name + " model is trained.\noverwrite it? (y/n)")
        if in_str != "y":
            sys.exit(1)
    mkdir(log_dir)
    img_dir = os.path.join(log_dir, "img")
    mkdir(img_dir)

    log_name = os.path.join(log_dir, datetime.datetime.now().strftime('%Y-%m-%dT%H') + "log.txt")
    with open(log_name, "w") as f:
        f.write(str(vars(opt)))

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="PRSSIMNLSIM",
    #
    #     # track hyperparameters and run metadata
    #     config=opt,
    #     name=opt.env_name,
    #     resume='allow'
    # )

    #####1 Load Data
    Dataset = load_dataset(opt.dataset)
    train_dataset = Dataset(opt.train_dir, opt)
    train_length = train_dataset.get_len()

    # opt.augment_size = opt.val_augment_size
    # val_dataset = train_dataset
    val_length = opt.val_augment_size // opt.batch_size

    print("start train...")
    with open(log_name, 'a') as f:
        f.write("env: %s train: %s input: %s" % (opt.env_name, opt.train_dir, opt.input_dir))
        f.write("start train...")

    #####2 Init Model
    arch = load_model(opt.arch)
    UNet = arch().to(device)
    optimizer = optim.AdamW(UNet.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    start_epoch = 0

    ####Load weights
    if opt.resume:
        pretrained = torch.load(os.path.join(log_dir, "model_latest.pth"))
        UNet.load_state_dict(pretrained['state_dict'])
        start_epoch = pretrained['epoch']
        optimizer.load_state_dict(pretrained['optimizer'])
        for p in optimizer.param_groups: lr = p['lr']
        iter = pretrained['iter']
        pklProfile = np.load(os.path.join(log_dir, "model_loss.pkl"))
    else:
        iter = 0
        pklProfile = {"loss": [],
                      "psnr": [], "lr": [], "val_loss": [],
                      "val_psnr": [],
                      "best_epoch": 0, "best_iter": 0,
                      "best_psnr": 0}
    best_psnr = 0
    if opt.scheduler == 'steplr':
        scheduler = StepLR(optimizer, step_size=opt.decay_step, last_epoch=start_epoch - 1, gamma=0.5)
    elif opt.scheduler == 'cawr':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=opt.decay_step, T_mult=opt.cawr_mult,
                                                last_epoch=start_epoch - 1)
    elif opt.scheduler == 'multisteplr':
        scheduler = MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.5)
    else:
        scheduler = StepLR(optimizer, step_size=10000, last_epoch=start_epoch - 1, gamma=0.5)

    #####3 Training...
    print("using device: " + opt.device)
    for epoch in range(opt.nepoch):
        start_time = time.time()
        for iter_ in range(train_length):
            optimizer.zero_grad()
            target, input_ = train_dataset.get_data(iter_)
            target = target.to(device)
            input_ = input_.to(device)
            restored = UNet(input_)
            loss_Sp = CharbonnierLoss(restored, target)

            loss = loss_Sp
            loss.backward()
            optimizer.step()
            # psnr = batch_PSNR(restored.cpu(), target.cpu()).item() / opt.batch_size

            # pklProfile["lr"].append(optimizer.state_dict()['param_groups'][0]['lr'])
            # pklProfile["loss"].append(loss.item())
            # pklProfile["psnr"].append(psnr)

            # wandb.log({"lr": optimizer.state_dict()['param_groups'][0]['lr'], "loss": loss.item(),
            #            # "psnr": psnr
            #            })

            # if opt.warmup and iter > opt.warmup_iter and loss.item() > opt.upper_loss:
            #     print("iter %s loss %s" % (iter, loss.item()))
            # else:

            if (iter + 1) % opt.val_checkpoint == 0:
                with torch.no_grad():
                    val_loss, val_loss_Sp, val_loss_Freq = 0,0,0
                    val_psnr = 0
                    for j in range(val_length):
                        target_t, input_t = train_dataset.get_data(j)
                        target_t = target_t.to(device)
                        input_t = input_t.to(device)
                        re = UNet(input_t)
                        loss_Sp = CharbonnierLoss(re, target_t).item()

                        val_loss += loss_Sp
                        val_psnr += batch_PSNR(re.cpu(), target_t.cpu()).item()
                    val_psnr /= val_length * opt.batch_size

                    # pklProfile["val_loss"].append(val_loss)
                    # pklProfile["val_psnr"].append(val_psnr)

                    # wandb.log({"val_loss": val_loss,"val_psnr": val_psnr})

                    if val_psnr > pklProfile["best_psnr"]:
                        pklProfile["best_epoch"] = epoch + 1
                        pklProfile["best_iter"] = iter + 1
                        pklProfile["best_psnr"] = val_psnr
                        # save_checkpoint(epoch + 1, iter + 1, UNet.state_dict(), optimizer.state_dict(),
                        #                 os.path.join(log_dir, "model_best.pth"))
                        # if iter > opt.warmup_iter:
                        #     img_path = os.path.join(opt.val_dir, opt.input_dir[len(opt.input_dir) - 1])
                        #     for img in os.listdir(img_path):
                        #         test_and_save(UNet, device, iter, opt, img_path, img)

                    print("Ep %d it %d  \tPSNR: %.4f\tVal Loss Sp: %.4f\tVal Loss Freq: %.4f\t\tbest_ep %d\tbest_it %d\tbest_psnr %.4f" %
                          (epoch + 1, iter + 1, val_psnr, val_loss_Sp, val_loss_Freq,
                           pklProfile["best_epoch"], pklProfile["best_iter"], pklProfile["best_psnr"]))
                    with open(log_name, 'a') as f:
                        f.write(
                            "Ep %d it %d  \tPSNR: %.4f\tVal Loss: %.4f\t\t\tbest_ep %d\tbest_it %d\tbest_psnr %.4f\n" %
                            (
                                epoch + 1, iter + 1, val_psnr, val_loss, pklProfile["best_epoch"],
                                pklProfile["best_iter"],
                                pklProfile["best_psnr"]))

            if (iter + 1) % opt.save_checkpoint == 0:
                img_path = os.path.join(opt.val_dir, opt.input_dir[len(opt.input_dir)-1])
                for img in os.listdir(img_path):
                    test_and_save(UNet, device, iter, opt, img_path, img)
                save_checkpoint(epoch + 1, iter + 1, UNet.state_dict(), optimizer.state_dict(),
                            os.path.join(log_dir, "model_it"+ str(iter+1).zfill(6) + ".pth"))
            iter += 1
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

    save_checkpoint(opt.nepoch, 0, UNet.state_dict(), optimizer.state_dict(), os.path.join(log_dir, "model_latest.pth"))
