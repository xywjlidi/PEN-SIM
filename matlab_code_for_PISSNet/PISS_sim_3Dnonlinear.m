clear;
close all;

clear;
close all;

addpath(genpath('./XxMatlabUtils'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

view_count = 4;
load_otf_flag = true;
ss_flag = true; %% sub-pixel registration
interp_flag = true; %% interpolation in up-sampling

n_dirs = 5;
n_phases = 9;
        
%% select file

file_path = '.\data\denoising\ER_3DNL\231219_sec61_skl\train-cam2';

file_list = dir(file_path);
save_dir = file_dir;
for j = 3:size(file_list, 1)

    file_name = file_list(j).name;
    if contains(file_name, 'views')
        continue
    elseif contains(file_name, 'SIrecon')
        continue
    end
    save_file_name = [file_name(1:end-4),'-views.mrc'];
    if isfile([save_dir,'/',save_file_name])
        continue
    end

    % raw file
    [header, data] = XxReadMRC([file_dir,'/',file_name]);
    Nx = double(header(1));
    Ny = double(header(2));
    N_slice = double(header(3));
    temp = typecast(header(46),'int16');
    Nt = double(temp(1));
    Nz = N_slice / (n_phases*n_dirs) / Nt;

    data = reshape(data,[Nx*Ny*n_phases*n_dirs*Nz,Nt]);

    img_raw = reshape(data,[Nx,Ny,n_phases*n_dirs,Nz,Nt]);

    header_out = header;
    for view_id = 1: 1: view_count
        if view_id == 1
            img_raw_v = img_raw(1:2:end,1:2:end,:,:,:);
        elseif view_id == 2
            img_raw_v = img_raw(2:2:end,1:2:end,:,:,:);
        elseif view_id == 3
            img_raw_v = img_raw(2:2:end,2:2:end,:,:,:);
        elseif view_id == 4
            img_raw_v = img_raw(1:2:end,2:2:end,:,:,:);
        else
            print('view id cannot be larger than 5')
        end
        img_raw_v = reshape(img_raw_v,[size(img_raw_v,1),size(img_raw_v,2),size(img_raw,3)*size(img_raw,4)*size(img_raw,5)]);

        if interp_flag
            img_raw_v = imresize(img_raw_v,2,'bicubic');
        else
            img_raw_v = imresize(img_raw_v,2,'nearest');
        end

        if ss_flag
            if view_id == 1
                img_raw_v = imtranslate(img_raw_v,[-0.5,-0.5]);
            elseif view_id == 2
                img_raw_v = imtranslate(img_raw_v,[-0.5,0.5]);
            elseif view_id == 3
                img_raw_v = imtranslate(img_raw_v,[0.5,0.5]);
            elseif view_id == 4
                img_raw_v = imtranslate(img_raw_v,[0.5,-0.5]);
            else
                print('view id cannot be larger than 4');
            end
        end

        img_raw_v = reshape(img_raw_v,[size(img_raw_v,1),size(img_raw_v,2),size(img_raw,3),size(img_raw,4),size(img_raw,5)]);

        data_view = uint16(img_raw_v);
        data = cat(1,data(:),data_view(:));

    end

    header_out(3) = int32(N_slice*(view_count+1));%nSlices
    header_out(46) = typecast([int16(Nt*(view_count+1)), int16(0)], 'int32');%nTimes


    handle = fopen([save_dir,'/',save_file_name],'w+');
    handle = XxWriteMRC_SmallEndian(handle, data(:), header_out);
    fclose(handle);

end