function [] = save_tiff_z_single(img, path)
    [h,w,z] = length(img);
    t = Tiff(path,'w');
    tag.Photometric = 1;
    tag.BitsPerSample=32;
    tag.SampleFormat = Tiff.SampleFormat.IEEEFP;
%     tag.SamplesPerPixel = 1;
    tag.ImageLength=h;
    tag.ImageWidth=w;
    tag.PlanarConfiguration=Tiff.PlanarConfiguration.Chunky;
    t.setTag(tag);
%     write(t,img);
    for ii = 1:z
        t.setTag(tag);
        write(t,img(:,:,ii));
        writeDirectory(t);
    end
    close(t);
end