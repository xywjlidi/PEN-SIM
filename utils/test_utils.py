from contextlib import contextmanager
import time

def load_model(name):
    r"""
    A dirty hack to load a module from a string input

    Returns:
        A pointer to the loaded module
    """
    strCmd = "from model.model import " + name + " as Net"
    exec(strCmd)
    return eval('Net')

def get_localization_model_config(name):
    log_dir = "./log"
    param = {"80nm":["DeSPL_localization3d_out2_b80_a0.01_min2_up3_20240416_refix",3,"UNet_3D_up3"],
            "100nm":["DeSPL_localization3d_out2_b100_a0.01_min2_up3_20230411_refix",3,"UNet_3D_up3"],
             "120nm": ["DeSPL_localization3d_out2_b120_a0.01_min2_up2_20240416_refix", 2, "UNet_3D_up2"],
             "150nm": ["DeSPL_localization3d_out2_b150_a0.01_min2_up2_20230411_refix", 2, "UNet_3D_up2"],
             "240nm": ["DeSPL_localization3d_out2_b240_a0.01_min2_up1_20240416_refix", 1, "UNet_3D"],
           }
    model_type = "model_best"

    return log_dir, param[name], model_type

def get_PISSNet_model_config(name):
    log_dir = "log"

    envlist = {"lifeact":"denoise_PISSNet_filter_lifeact_NL",
                "paxillin":"denoise_PISSNet_20230710_paxillin_NL",
                "ensconsin":"denoise_PISSNet_20230710_ensconsin_NL",
                "sec61":"denoise_PISSNet_20230712_18_sec61_NL",
                "omm":"denoise_PISSNet_20230718_OMM_NL_mix",
                "lamp1":"denoise_PISSNet_20230718_lamp1_NL_2_4",
                "phb2":"denoise_PISSNet_20230718_PHB2_NL_mix",
               "skl": "denoise_PISSNet_20230729_skl_NL_rolling_step8_noc1c2",
               "skl-PISSNet": "denoise_PISSNet_20230718_skl_NL",

                "lifeact-paxillin":"denoise_PISSNet_20230729_lifeact_paxillin_NL_mix",
               "paxillin-fret": "denoise_PISSNet_20230729_lifeact_paxillin_NL_ccd2",
                "omm-ensconsin":"denoise_PISSNet_20230725_OMM_Ensconsin_NL_mix",
               "omm-fret": "denoise_PISSNet_20230725_OMM_Ensconsin_ccd2",
               "lamp1-skl":"denoise_PISSNet_20231020_lamp1_skl_NL_c1_rolling_step8",

                "ensconsin-3D":"denoise_PISSNet_3D_20230725_7ang_ensconsin_NL",
                "laminb-3D":"denoise_PISSNet_3D_20231026_laminB_NL",
                "omm-3D": "denoise_PISSNet_3D_20230912_OMM_NL",
                "lamp1-3D":"denoise_PISSNet_3D_20230912_lamp1_NL",
               "lifeact-3D":"denoise_PISSNet_3D_20231026_lifeact_NL",

               "sec61-skl-3D": "denoise_PISSNet_3D_20231219_ER_NL",

               "lifeact-3D-Neighbor2Neighbor": "Neighbor2Neighbor_0429_UNet3D_lifeact_20231026",
               "lifeact-2D-Neighbor2Neighbor": "Neighbor2Neighbor2D_lifeact_3DNL",
               "lifeact-N2V": "Noise2Void_lifeact3DNL",

               }
    # PRS-SIM
    model_type = "model_it100000"
    arch = "UNet_2D_N2N"
    if name.find("3D") > -1:
        arch = "UNet_3D_N2N"

    # Neighbor2Neighbor
    if name.find("3D-Neighbor2Neighbor") > -1:
        model_type = "model_it020000"
    elif name.find("2D-Neighbor2Neighbor") > -1:
        model_type = "epoch_model_082"
        arch = "UNet"
    # Noise2Void
    elif name.find("N2V") > -1:
        model_type = "model_epoch0150"
        arch = "ResNet"

    return log_dir, envlist[name], arch, model_type

@contextmanager
def timeblock(label, debug=1):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        if debug:
            print('{} : {}'.format(label, end - start))
