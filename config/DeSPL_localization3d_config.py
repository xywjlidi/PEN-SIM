from config.config import BasicConfig

_C = BasicConfig()

# TODO
_C.env_name = "TFM/DeSPL_localization3d_out2_b100_a0.01_min2_up3_20230411_refix"
_C.arch = "UNet_3D_up3"
_C.resume = False
_C.device = 'cuda:0'

# TODO
_C.train_dir = '/mnt/data1/chy/UNet/data/train_bead/TrainData_488_1.41rNA_300nmbeads/train'
_C.val_dir = '/mnt/data1/chy/UNet/data/train_bead/TrainData_488_1.41rNA_300nmbeads/val'
_C.haslabel = False

# TODO
_C.gt_dir = '0.3beads_20230411_refix_nc_position'
# TODO
_C.input_dir = 'TIRF560_0.1beads_20230411_3D_refix'
_C.upfactor = 3
_C.zupfactor = 1
_C.img_type = ".tif"
_C.normal = True
_C.input_img_neg = False
_C.img_frame = 0  # 0=all
_C.is3D = True

_C.dataset = 'Dataset_3dnt_augment_eachit_extend_baseline'
# TODO
_C.rotate = False
_C.rotate90 = True
_C.flip = True
_C.frontground_filter = False

_C.OTF_path = './data/OTF/OTF.mat'
_C.penalty = 'l1norm'
# TODO
_C.alpha = 0.01

_C.kernel = 'gauss'
_C.kernel_size = 25
_C.dxy = 25
_C.sigma = 8
# TODO
_C.sigma_min = 2
_C.change_step = 3000
_C.gamma = 0.9

# TODO
_C.augment_size = 20000
_C.val_augment_size = 500
_C.noise_crop = 64
_C.clean_crop = _C.noise_crop * _C.upfactor
_C.crop_depth = 8
_C.nepoch = 50

_C.batch_size = 4

_C.scheduler = 'steplr'
_C.decay_step = 15
_C.lr = 0.0001
_C.warmup = True
_C.warmup_iter = 10000

# TODO
_C.val_checkpoint = 300
_C.save_checkpoint = 2500
# TODO
_C.save_img = '1.tif'
_C.split_test = 1  # 0=no 1=yes

_C.normal_max = 0.999
_C.normal_min = 0.1
