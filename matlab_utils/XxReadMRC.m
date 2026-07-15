function [header, data] = XxReadMRC(file)

handle = fopen(file,'r');
header = int32(fread(handle,256,'int32'));

if header(4)>7              %%%%%%% big endian
    frewind(handle);
    header=int32(fread(handle,256,'int32','b'));
    
    if nargin==1
        switch header(4)
            case 6
                rawimage=uint16(fread(handle,header(1)*header(2)*header(3),'uint16','b'));
            case 4
                fseek(handle, header(24), 'cof');
                %           ftell(handle)
                rawimage=single(fread(handle,header(1)*header(2)*header(3)*2,'single','b'));
            case 2
                fseek(handle, header(24), 'cof');
                rawimage=single(fread(handle,header(1)*header(2)*header(3),'single','b'));
        end
        data=rawimage;
    end
    fclose(handle);
    
else
    if nargin==1
        switch header(4)
            case 6
                rawimage=uint16(fread(handle,double(header(1))*double(header(2))*double(header(3)),'uint16'));
            case 4
                fseek(handle, header(24), 'cof');
                rawimage=single(fread(handle,double(header(1))*double(header(2))*double(header(3))*2,'single'));
            case 2
                fseek(handle, header(24), 'cof');
                rawimage=single(fread(handle,double(header(1))*double(header(2))*double(header(3)),'single'));
        end
        data=rawimage;
    end
    fclose(handle);
end

end

