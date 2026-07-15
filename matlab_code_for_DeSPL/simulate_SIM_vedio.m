clear;clc;
addpath(genpath('./file_utils'));
%% config
% path = '/mnt/Public/OlympusSIM/xsz/20221025-TFM/GA0.94-EDC50%-300nmbeads-10x/halo-lifeact/c5_20221025_233000';
% path = '/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/UNet/img/localization/300nm-degradation-100nm-TFM/20230817_TFM_300nm/';
% savepath = '/media/li-lab/1c1d7fee-cb9a-46f4-87fa-98f6216d5d0e/chy/UNet/img/localization/300nm-degradation-100nm-TFM/20230817_TFM_300nm/';

path = '/mnt/data1/chy/UNet/data/train_bead/TrainData_488_1.41rNA_300nmbeads/20230411_TFM_300nm/dataset_recrop/';
savepath = '/mnt/data1/chy/UNet/data/train_bead/TrainData_488_1.41rNA_300nmbeads/20230411_TFM_300nm/dataset_recrop/';
filename_pattern = 'crop.tif';
bead_size = 300 * 1e-3;
simulate_bead_size = 150 * 1e-3;
input_pixel_size = 0.0303;
simulate_pixel_size = 0.0606;


filelist = dir(path);
end_number = size(filelist,1);
parfor img_number = 3:end_number
% for img_number = 3:end_number
    if ~contains(filelist(img_number).name,filename_pattern)
        continue
    end
    
    filename = filelist(img_number).name;
    ext = split(filename,'.');
    if strcmp(ext(end),'mrc')
        [header, img] = XxReadMRC([path filesep filename]);
        Nx = single(header(1));
        Ny = single(header(2));
        Nz = single(header(3));
        img = reshape(img, [Nx Ny Nz]);
    else
    %     img = imread([path filesep filename]);

        info = imfinfo(fullfile([path filesep filename]));
        Nx = single(info(1).Height);
        Ny = single(info(1).Width);
        Nz = single(numel(info));

        img=zeros(info(1).Height,info(1).Width,numel(info));
        for j=1:numel(info)
            img(:,:,j)=double(imread(fullfile([path filesep filename]),j));
        end

    %     [Nx, Ny, Nz] = size(img);
    end

    ExNA = 1.41;
    DetNA = 1.17;
    exLambda = 0.560;
    emLambda = 0.609;
    % exLambda = 0.488;
    % emLambda = 0.525;

    signal = 200;
    background = 100; % mean value of gaussian noise 
    sigma_gs = 3.5; % std of gaussian noise

    upfactor = bead_size / simulate_bead_size;
    dxy = input_pixel_size / upfactor;
    downfactor = double(dxy / simulate_pixel_size);
    SegSize = fix(Nx * downfactor);
    Nx = SegSize / downfactor;
    Ny = Nx;
    %% preprocess

    % define SIM parameters
    angle_k0 = [0.0908 -0.9564 -2.0036];
    ndirs = length(angle_k0);
    nphases = 3;
    phase_space = 2 * pi / nphases;
    ls = exLambda * 0.5 / ExNA;
    xx = dxy * (-Nx / 2 : Nx / 2 - 1);
    yy = dxy * (-Ny / 2 : Ny / 2 - 1);
    [X,Y] = meshgrid(xx,yy);

    % generate SIM pattern
    pattern = zeros(Ny, Nx, ndirs * nphases);
    for id = 1:ndirs
        alpha = angle_k0(id);
        for ip = 0:nphases-1
            kxL = 2 * pi / exLambda * ExNA * cos(alpha);
            kyL = 2 * pi / exLambda * ExNA * sin(alpha);
            kxR = -2 * pi / exLambda * ExNA * cos(alpha);
            kyR = -2 * pi / exLambda * ExNA * sin(alpha);
            phOffset = ip * phase_space;
            interBeam = exp(1i*(kxL*X + kyL*Y + phOffset)) + exp(1i*(kxR*X + kyR*Y));
            interBeam = XxNorm(abs(interBeam) .^ 2);
            pattern(:, :, (id-1)*nphases+ip+1) = interBeam;
        end
    end

    % define psf
    % psf_sigma = emLambda * 0.5 / DetNA / 2.355 / dxy;
    % psf_size_lr = round(psf_sigma * 3) * 2 + 1;
    % psf_lr = fspecial('gaussian', [psf_size_lr, psf_size_lr], psf_sigma);
    % psf_lr = psf_lr / max(psf_lr(:));
    % save_tiff_uint16(uint16(round(65535*XxNorm(psf_lr))), [savepath filesep  num2str(bead_size) 'to' ...
    %         num2str(simulate_bead_size) 'beads_psf.tif'])

    % define experimental OTF
    OTF_path = '/mnt/data1/chy/UNet/data/train_bead/TrainData_488_1.41rNA_300nmbeads/batfile/TIRF560_cam2_step1_001_z35_OTF.mrc';
    dkx = 1 / (Nx * dxy);
    dky = 1 / (Ny * dxy);
    [headerotf, rawOTF] = XxReadMRC(OTF_path);
    nxotf = single(headerotf(1));
    nyotf = single(headerotf(2));
    nzotf = single(headerotf(3));
    dkrotf = single(typecast(headerotf(11),'single'));
    dkr = min(dkx,dky);
    diagdist = ceil(sqrt((Nx/2)^2+(Ny/2)^2)+1);
    OTF = complex(rawOTF(1:2:end),rawOTF(2:2:end));
    x = (0:dkrotf:(nxotf-1)*dkrotf)';
    xi = (0:dkr:(nxotf-1)*dkrotf)';
    OTF = interp1(x,OTF,xi,'spline');
    sizeOTF = max(size(OTF));
    OTF(sizeOTF+1:diagdist)=0;

    dx = (-Nx/2:1:Nx/2-1)*dkx;
    dy = (-Ny/2:1:Ny/2-1)*dky;
    [dX,dY] = meshgrid(dx,dy);
    rdist = sqrt(dX.^2+dY.^2);
    otflen = max(size(OTF))-1;
    OTF = interp1(0:dkr:otflen*dkr, OTF, rdist, 'spline');

    % header
    HeaderPath = 'TemplateMRCHeader.mrc';
    header = XxReadMRCHeader(HeaderPath);
    header(1) = int32(SegSize);% pixel size
    header(2) = int32(SegSize);% pixel size
    header(3) = int32(Nz * ndirs * nphases);% nz
    header(4) = int32(6);% data type 6=uint16
    header(11) = typecast(single(simulate_pixel_size), 'int32');% dx
    header(12) = typecast(single(simulate_pixel_size), 'int32');% dy
    header(46) = typecast([int16(Nz), int16(1)], 'int32');%nTimes
    header(50) = typecast([int16(1), int16(exLambda*1000)], 'int32');%lambda
    %%
    % p = parpool(10);
    img_process = uint16(zeros(SegSize, SegSize, ndirs*nphases*Nz));
    for frame_number = 1:Nz
        display(['processing img '  num2str(img_number) ' frame:' num2str(frame_number) ])
        img_n = img(1:Nx,1:Ny,frame_number);
        img_n(find(img_n<0))=0;
        im_gt = zeros(SegSize, SegSize, ndirs*nphases);
        im_noise = zeros(SegSize, SegSize, ndirs*nphases);
        img_signal = zeros(Nx, Ny, ndirs*nphases);
    %     img_signal = zeros(SegSize, SegSize, ndirs*nphases);
        img_pattern = zeros(Nx, Ny, ndirs*nphases);
        for i = 1:ndirs*nphases
            % step 1: add SIM pattern
            img_pattern(:, :, i) = pattern(:, :, i) .* img_n;

    %         otf = fftshift(fft2(ifftshift(img_pattern)));
    %         temp = zeros(Nx,Ny);
    %         start = floor((Nx-SegSize)/2);
    %         temp(start+1:start+SegSize, start+1:start+SegSize) = otf;
    %         im_gt(:,:,i) = abs(fftshift(ifft2(ifftshift(temp))));

            % step 2: resize OTF 
            img_signal(:,:,i) = signal * real(ifft2(ifftshift(OTF.*fftshift(fft2(img_pattern(:, :, i))))));
            im_gt(:, :, i) = imresize(img_signal(:, :, i), [SegSize, SegSize]);
            im_noise(:, :, i) = imresize(poissrnd(img_signal(:, :, i)) +...
                    normrnd(background, sigma_gs, size(img_signal(:, :, i))), [SegSize, SegSize]);
    %             
    %         figure();
    %         subplot(2,2,1);
    %         imshow(img_pattern(:, :, i),[]);
    %         subplot(2,2,2);
    %         imshow(fftshift(fft2(ifftshift(img_pattern(:,:,i)))),[]);
    %         subplot(2,2,3);
    %         imshow(im_gt(:,:,i),[]);
    %         subplot(2,2,4);
    %         imshow(fftshift(fft2(ifftshift(im_gt(:,:,i)))),[]);
        end

    %     img_pattern = uint16(round(65535*XxNorm(img_pattern)));
    %     img_signal = uint16(round(65535*XxNorm(img_signal)));
    %     im_gt = uint16(round(65535*XxNorm(im_gt)));
    %     im_noise = uint16(round(65535*XxNorm(im_noise)));

        img_process(:,:,(frame_number-1)*ndirs*nphases+1:frame_number*ndirs*nphases) = uint16(round(65535*XxNorm(im_noise)));
    end
        % step 5: save
        XxWriteMRC_SmallEndian(rot90(img_process,3), [savepath filesep num2str(bead_size) 'to' ...
            num2str(simulate_bead_size) 'beads_' num2str(signal) 'signal_' filename(1:end-4) '.mrc'], header);
    %     save_tiff_z_uint16(im_gt, [savepath filesep num2str(bead_size) 'to' num2str(simulate_bead_size) 'beads_gt.tif'])
    %     save_tiff_z_uint16(img_signal, [savepath filesep num2str(bead_size) 'to' num2str(simulate_bead_size) 'beads_origin.tif'])
    %     save_tiff_z_uint16(img_pattern, [savepath filesep num2str(bead_size) 'to' num2str(simulate_bead_size) 'beads_pattern.tif'])
    %     save_tiff_single(single(fftshift(ifft2(ifftshift(OTF)))), [savepath filesep  num2str(bead_size) 'to' ...
    %         num2str(simulate_bead_size) 'beads_psf.tif'])
    % delete(p);
end