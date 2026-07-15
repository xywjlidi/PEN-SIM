# Deep Learning with PEN-SIM

## Set up environment

python version 3.7, pytorch version 1.8.0   
`$ pip install requirements.txt`  

## PISSNet for PEN-SIM
1. preprocess training dataset  
a. modify the variance `filelist` in Line 21 in `./matlab_code_for_PISSNet/PISS_sim_nonlinear.m` or `./matlab_code_for_PISSNet/PISS_sim_3Dnonlinear.m`  
b. run `./matlab_code_for_PISSNet/PISS_sim_nonlinear.m` or `./matlab_code_for_PISSNet/PISS_sim_3Dnonlinear.m`  
c. reconstruct raw SIM images groups `*-views[1-4]` by parameter estimated in origin raw SIM images.

2. train PISSNet  
a. modify options in `./config/PENSIM2D_PISSNet.py` or `./config/NLSIM3D_PISSNet.py`   
b. 
`$ python train_PISSNet.py --opt PENSIM2D_PISSNet` or  
`$ python train_PISSNet.py --opt PENSIM3D_PISSNet`  

3. predict PEN-SIM images  
a. modify `env` variance in Line 43 and `filelist` variance 49 in Line in `./demo_test_PISSNet_denoise_model_save_tif.py` or   
`./demo_test_PISSNet_denoise_model_save_mrc.py`.  
b. `$ python demo_test_PISSNet_denoise_model_save_tif.py` or  
`$ python demo_test_PISSNet_denoise_model_save_mrc.py`  
 
 
Note: `./demo_test_PISSNet_denoise_model_save_tif.py` only support file size below 4GB.

## DeSPL for PEN-SIM
1. preprocess training dataset  
    a. get the simulated beads images  
    i. `./matlab_code_for_DeSPL/simulate_SIM_vedio.m`  
    ii. reconstruct beads raw SIM images.  
    iii. `./matlab_code_for_DeSPL/merge_tform.m`  
    iv. `./matlab_code_for_DeSPL/process_tform_rigid.m`

    b. get the groundtruth of simulated beads images  
        localize the the origin beads SIM images by traditional algorithm.  

2. train DeSPL  
a. modify options in `./config/DeSPL_localization3d_config.py`   
b. `$ python train_DeSPL.py --opt DeSPL_localization3d_config` 

3. predict beads PEN-SIM images  
a. reconstruct beads raw SIM images.  
b. stablize images.  
c. run `./matlab_code_for_DeSPL/beadssimfilter.m`  
d. modify `env` variance in Line 34, `filelist` variance in Line 40 in `./demo_test_DeSPL_localize_model_save_tif.py`.  
Or modify `env` variance in Line 34 and `filelist` variance in Line 40 and `headerlist` variance in Line 41 in `./demo_test_DeSPL_localize_model_save_mrc.py`.  
e. `$ python demo_test_DeSPL_localize_model_save_tif.py` or  
`$ python demo_test_DeSPL_localize_model_save_mrc.py`  
 
Note: `./demo_test_DeSPL_localize_model_save_tif.py` only support file size below 4GB.
