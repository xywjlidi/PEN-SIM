clc
clear
addpath(genpath('./matlab_utils'));

% path = '/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/UNet/data/train_bead/TrainData_488_1.41rNA_300nmbeads/origin_20220907/0.24to0.12_w0.01_DC10_0.6/';
% path = '/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/UNet/data/train_bead/TestData_560_1.41rNA_100nmbeads/SIrecon/';

path = '/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/UNet/data/train_bead/TestData_560_1.41rNA_100nmbeads/20230426/TFM_edc10%_40x/';

% filename = 'TIRF560_0.3to0.15beads_200signal_19_SIrecon.mrc';
f_tag = 'c5_Stablized_crop.tif';

list = dir(path);
end_number = size(list);
for img_number = 3:end_number
    filename = list(img_number).name;
    
    if ~contains(filename,f_tag)
        continue
    end
    
    if strcmp(filename(end-2:end),'mrc') ==1
        [header, temp] = XxReadMRC([path filesep filename]);
        temp = reshape(temp, [header(1) header(2) header(3)]);
        Nx = header(1);
        Ny = header(2);
        Nz = header(3);
    else        
        info = imfinfo(fullfile([path filesep filename]));
        Nx = single(info(1).Height);
        Ny = single(info(1).Width);
        Nz = single(numel(info));

        temp=zeros(info(1).Height,info(1).Width,numel(info));
        for j=1:numel(info)
            temp(:,:,j)=double(imread(fullfile([path filesep filename]),j));
        end
    end

    sigma = 10;
    DC = 0.6;

    guass_kernel = fspecial('gaussian', [double(Nx), double(Ny)], sigma);
    guass_kernel = guass_kernel/max(guass_kernel(:));
    filter = 1-guass_kernel*DC;

    img = zeros(Nx,Ny,Nz);
    for z = 1:Nz
        img(:,:,z) = real(ifft2(ifftshift(filter.*fftshift(fft2(temp(:,:,z))))));
    end
    save_tiff_z_single(single(img), [path filename(1:end-4) '_DC' num2str(sigma) '_' num2str(DC) '.tif'])
%     save_tiff_single(img, [path filename(1:end-4) '_DC' num2str(sigma) '_' num2str(DC) '.tif'])
end