from config.config import BasicConfig

_C = BasicConfig()

# TODO
_C.env_name = "denoise_PISSNet_filter_lifeact_NL"
_C.arch = "UNet_2D_N2N"
_C.resume = False
_C.device = 'cuda:0'

# TODO
_C.train_dir = r'./data/denoising/filter_lifeact_NL/'
# TODO
_C.val_dir = r'./data/denoising/filter_lifeact_NL/'

# TODO
_C.input_dir = ['20220616/step1','20220616/step2','20220616/step3','20220616/step4'
                ,'20220707/step2','20220707/step3','20220707/step4'
                ,'20230605/step1','20230605/step2','20230605/step3','20230605/step4'
                ,'val']
_C.gt_dir = None
_C.pairs = ['view-1_SIrecon.mrc','view-2_SIrecon.mrc','view-3_SIrecon.mrc','view-4_SIrecon.mrc']
_C.upfactor = 1
_C.is3D = False
_C.img_type = ".mrc"
_C.img_frame = 0
_C.input_img_neg = True
_C.normal = False
#
_C.dataset = 'Dataset_n2n_augment_eachit_in_cache_extend_baseline'
# TODO split fg vs bg
_C.rotate = False
_C.flip = False
_C.frontground_filter = False # filer for 3dnt n2n is cropped by set center at point in mask
# TODO
_C.augment_size = 10000  # 1536 // 128 = 400  ans * 2 * 20 = 20000
_C.val_augment_size = 500
_C.noise_crop = 128
_C.clean_crop = _C.noise_crop * _C.upfactor
_C.crop_depth = None

_C.nepoch = 50
_C.batch_size = 4

# Network init weight
_C.init_type = "kaiming_uniform"
_C.init_bn_type = "uniform"
_C.init_gain = 0.2

# TODO
_C.opti = 'adam'
_C.opti_decay = 0.5
_C.scheduler = 'multisteplr'
_C.milestones = [30000, 40000, 50000, 60000]
_C.decay_step = 10  # decay_step or T-0
_C.cawr_mult = 2  # int
_C.lr = 5e-5
_C.warmup = True
_C.warmup_iter = 1000
_C.upper_loss = 1

_C.val_checkpoint = 100
_C.save_checkpoint = 5000
# TODO
_C.save_img = ['20220616_c6_step1.mrc','20220616_c6_step2.mrc','20220616_c6_step3.mrc','20220616_c6_step4.mrc',
               '20220707_c1_step2.mrc','20220707_c1_step3.mrc','20220707_c1_step4.mrc',
               '20230605_c62_step1.mrc','20230605_c62_step2.mrc', '20230605_c62_step3.mrc', '20230605_c62_step4.mrc']
_C.split_test = 16
_C.split_padding = 16

_C.normal_max = 0.999
_C.normal_min = 0.01
