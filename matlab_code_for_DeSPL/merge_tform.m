clc
clear all
close all
addpath(genpath('./file_utils'));

%经过尝试，最优的效果是先用similarity计算出来一个初始t矩阵，然后在用这个初始值，用affine计算出来精确的t矩阵，imregtform是产生t矩阵，imregister是直接计算图片，两者在参数一样时的结果是一致的
%%0是SIM的原始图加和为TIRF图；
file_type=2;%等于1就是mrc，等于2就是tif文件

normalization_judge=1;%等于1就把两个作为标准的bead图调节到亮度一样，然后merge
pixel_cal_nor=400;%使用多少个最亮的pixel计算来计算图片的平均灰度
normalization_rate=0.2;%归一化后最大亮度设定为饱和亮度的比例，应该低于1，防止高于平均灰度的像素过饱和
%%1提取一定的帧数；
%%2处理stop不好导致的图片不够9/15/25倍数；
%%3是不同z拍摄的tirf转化为2D的tirf
%%4是3ang2ph的6图转4图
%%5是图像flip
%%6是图像缩放

SIM_3D=0;%等于1，就表示处理的是3D数据。
bead=1;%等于1就是展示两个相机的细胞器的merge图
bead_nz1=1;
bead_nz2=1;%主要用于3D数据的处理，2D数据中这两个数就都填1

fitstep_rate=5;%步长=原始步长/fitstep_rate
fit_time=1000;%一般填上3和500就够了，拟合不出来就填上5和1000
%%%%%基本参数
ndirs0=3;
nphases0=3;%原始的方向和相位
Pic_Bit=16;%图像的bit数


gamma_value=1;%已经验证与image-J的参数完全一致
gf_pixel=0;

show_flag = 0;%0=invisible, 1=show

%%%%%%%%%%%%%bead数据读取以及T变换矩阵生成
%不是直接的录像文件，是包含录像的文件夹
file_bead_c2=['/mnt/data1/chy/UNet/data/train_bead/TrainData_488_1.41rNA_300nmbeads/20230411_TFM_300nm/TIRF560_0.3to0.15_recrop/' ...
    '0.3to0.15beads_200signal_c10_step3-1-22_crop_L.tif'];
file_bead_c1=['/mnt/data1/chy/UNet/data/train_bead/TrainData_488_1.41rNA_300nmbeads/20230411_TFM_300nm/TIRF560_0.3to0.15_recrop/' ...
    'c10_step3-1-22_crop.tif'];

if file_type==1;
    [header_bead_c2, data_bead_c1] = XxReadMRC(file_bead_c1);
    nx_bead_c1=double(header_bead_c2(1));
    ny_bead_c1=double(header_bead_c2(2));
    nz_bead_c1=double(header_bead_c2(3));
    data_bead_c1=reshape(data_bead_c1, [nx_bead_c1,ny_bead_c1,nz_bead_c1]);
    data_bead_c1=double(data_bead_c1);
    data_bead_c1=data_bead_c1(:,:,bead_nz1);%防止有多张的bead录像
    data_bead_c1(data_bead_c1<0)=0;    

    [header_bead_c2, data_bead_c2] = XxReadMRC(file_bead_c2);
    nx_bead_c2=double(header_bead_c2(1));
    ny_bead_c2=double(header_bead_c2(2));
    nz_bead_c2=double(header_bead_c2(3));
    data_bead_c2=reshape(data_bead_c2, [nx_bead_c2,ny_bead_c2,nz_bead_c2]);
    data_bead_c2=double(data_bead_c2);
    data_bead_c2=data_bead_c2(:,:,bead_nz2);%防止有多张的bead录像
    data_bead_c2(data_bead_c2<0)=0;    
else
    data_bead_c1=double(imread(file_bead_c1));
    data_bead_c2=double(imread(file_bead_c2));    
end

data_bead_c2 = imresize(data_bead_c2, size(data_bead_c1));

if normalization_judge==1
    data_sort_bead_c1=data_bead_c1;
    data_sort_bead_c1=data_sort_bead_c1(:);
    data_sort_bead_c1=sort(data_sort_bead_c1);
    gray_max_mean_bead_c1=mean(data_sort_bead_c1(end-pixel_cal_nor:end));
    %normalization_rate_temp=(normalization_rate*2^Pic_Bit)/gray_max_mean
    data_bead_c1=data_bead_c1*(normalization_rate*2^Pic_Bit)/gray_max_mean_bead_c1;
    gray_max_bead_c1=max(max(data_bead_c1));

    if gray_max_bead_c1>2^Pic_Bit
        str = sprintf('max gray %d bigger than 65535,cut normalization_rate',gray_max_bead_c1);
        disp(str);
        return
    else
    end

    data_sort_bead_c2=data_bead_c2;
    data_sort_bead_c2=data_sort_bead_c2(:);
    data_sort_bead_c2=sort(data_sort_bead_c2);
    gray_max_mean_bead_c2=mean(data_sort_bead_c2(end-pixel_cal_nor:end));
    %normalization_rate_temp=(normalization_rate*2^Pic_Bit)/gray_max_mean
    data_bead_c2=data_bead_c2*(normalization_rate*2^Pic_Bit)/gray_max_mean_bead_c2;
    gray_max_bead_c2=max(max(data_bead_c2));

    if gray_max_bead_c2>2^Pic_Bit
        str = sprintf('max gray %d bigger than 65535,cut normalization_rate',gray_max_bead_c2);
        disp(str);
        return
    else
    end
else
end

% Monomodal images have similar brightness and contrast.
% Multimodal images have different brightness and contrast. The images can come from two different types of devices.
[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = optimizer.InitialRadius/fitstep_rate;%优化步长变短
optimizer.MaximumIterations = fit_time;%优化的迭代次数增加
% "translation"	Translation transformation
% "rigid"	Rigid transformation: translation and rotation
% "similarity"	Similarity transformation: translation, rotation, and isotropic scaling
% "affine"	Affine transformation: translation, rotation, anisotropic scaling, and shearing
tform_initial = imregtform(data_bead_c1,data_bead_c2,'similarity',optimizer,metric);%用similarity的变换方式做初始配准，在这里imregtform把变化矩阵输出
tform_rigid = imregtform(data_bead_c1,data_bead_c2,'affine',optimizer,metric,'InitialTransformation',tform_initial);%计算出精确的t矩阵

file_bead_out=[file_bead_c1(1:end-4),'_T_matrix.mat'];
file_bead_out_ID=fopen(file_bead_out,'w+');
% fprintf(file_bead_out,'affine2d',tform_rigid)
% Save(file_bead_out,'-struct',’structname’,’val1’,’val2’,…)

save(file_bead_out,'tform_rigid');%把work space的内容保存下来
fclose(file_bead_out_ID);
