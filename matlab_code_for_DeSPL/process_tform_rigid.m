clc
clear all
close all
addpath(genpath('/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/matlab/file_utils'));

SIM_3D=0;%等于1，就表示处理的是3D数据。
gamma_value=1;%已经验证与image-J的参数完全一致
gf_pixel=0;

load('/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/UNet/img/localization/300nm-degradation-100nm-TFM/20230817_TFM_300nm/20230817_152106_Stablized_T_matrix.mat');
pathname = '/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/UNet/img/localization/300nm-degradation-100nm-TFM/20230817_TFM_300nm/';
c_list = dir(pathname);

filename_pattern = 'Stablized.tif';

for i=3:length(c_list)
    if ~contains(c_list(i).name,filename_pattern)
        continue
    end
    %% read
    file_cell_c1=[pathname '/' c_list(i).name];

    info = imfinfo(fullfile(file_cell_c1));
    data=zeros(info(1).Height,info(1).Width,numel(info));
    for j=1:numel(info)
        data(:,:,j)=double(imread(fullfile(file_cell_c1),j));
    end

    %% process cell
    [nx_cell_c2,ny_cell_c2,nz_cell_c2]=size(data);    
    data_cell_c1_move=zeros(nx_cell_c2,ny_cell_c2,nz_cell_c2);
    for frame_num=1:1:nz_cell_c2
        if isnan(data(:,:,frame_num))==1
        else
            Rfixed_cell = imref2d(size(data(:,:,frame_num)));%imref2d限制变换后的图像与参考图像有相同的坐标分布
            data_cell_c1_move(:,:,frame_num) = imwarp(data(:,:,frame_num),tform_rigid,'OutputView',Rfixed_cell);%imwarp函数执行几何变换，当然依据则是tformSimilarity的变换矩阵了。

            if SIM_3D==1
            else
                data_cell_c1_move(:,:,frame_num)=fliplr(data_cell_c1_move(:,:,frame_num));%上下翻转图像以便和原始图方向一致,只有2D数据才需要翻转        end
            end

            data_cell_c1_move(:,:,frame_num)=data_cell_c1_move(:,:,frame_num).^gamma_value;%经过验证已经验证与image-J的参数完全一致
            if gf_pixel==0;
            else
                gausFilter = fspecial('gaussian', [5,5], gf_pixel);
                %gausFilter = fspecial('disk', 1);
                data_cell_c1_move(:,:,frame_num)= imfilter(data_cell_c1_move(:,:,frame_num), gausFilter);
            end
        end
    end
    
    data_target = single(fliplr(data_cell_c1_move(:,:,:)));%上下翻转图像以便和原始图方向一致,只有2D数据才需要翻转

    file_out=[file_cell_c1(1:end-4),'_for_merge-ga',num2str(gamma_value,'%.2f'),'-gfMT',num2str(gf_pixel,'%.2f'),'.tif'];
    handleout=fopen(file_out,'w+');
%         imwrite(data_target,file_out)
    save_tiff_z_single(data_target, file_out);
end