import numpy as np
import imageio
from read_utils.utils import prctile_norm
import os
import math

def data_loader(images_path, data_path, gt_path, height, width, batch_size, 
                prctile_norm_flag=0,bn=0):
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        img = imageio.imread(path).astype(np.float32)
        path_gt = path.replace(data_path, gt_path) 
        #print(path_gt)
        gt = imageio.imread(path_gt).astype(np.float32)
        if prctile_norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / 65535
            gt = gt / 65535
        image_batch.append(img)
        gt_batch.append(gt)
    sr_ratio = gt.shape[0]//img.shape[0]
    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)
    if bn:
        image_batch = (image_batch - 0.5) / 0.5
        gt_batch = (gt_batch - 0.5) / 0.5
    image_batch = image_batch.reshape((batch_size, height, width, 1))
    gt_batch = gt_batch.reshape((batch_size, height*sr_ratio, width*sr_ratio, 1))

    return image_batch, gt_batch

def data_loader3D(images_path, data_path, gt_path, height, width, depth, 
                  batch_size, N2N_flag, depth_first_flag, half_side_flag,
                  simu_flag, norm_flag=1):
    batch_images_path = np.random.choice(images_path, size=batch_size)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        img = imageio.imread(path).astype(np.float)
        path_gt = path.replace(data_path, gt_path) #str.replace(old, new)
        gt = np.array(imageio.mimread(path_gt)).astype(np.float)
        if not depth_first_flag:
            gt = np.transpose(gt,(1,2,0))
        if norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / 65535
            gt = gt / 65535
        image_batch.append(img)
        gt_batch.append(gt)

    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)
    
    if not half_side_flag:
        if depth_first_flag:
            image_batch = image_batch.reshape((batch_size, 1, height, width, 1))
            image_batch = np.concatenate((np.zeros([batch_size, int((depth-1)/2), height, width, 1]), image_batch),axis=1)
            image_batch = np.concatenate((image_batch,np.zeros([batch_size, int((depth-1)/2), height, width, 1])),axis=1)
    
            if N2N_flag or simu_flag:
                gt_batch = gt_batch.reshape((batch_size, depth, height, width, 1))
            else:
                gt_batch = gt_batch.reshape((batch_size, depth, height*2, width*2, 1))
        else:
            image_batch = image_batch.reshape((batch_size, height, width, 1, 1))
            image_batch = np.concatenate((np.zeros([batch_size, height, width, int((depth-1)/2), 1]), image_batch),axis=3)
            image_batch = np.concatenate((image_batch,np.zeros([batch_size, height, width, int((depth-1)/2), 1])),axis=3)
    
            if N2N_flag or simu_flag:
                gt_batch = gt_batch.reshape((batch_size, height, width, depth, 1))
            else:
                gt_batch = gt_batch.reshape((batch_size, height*2, width*2, depth, 1))
    else:
        if depth_first_flag:
            image_batch = image_batch.reshape((batch_size, 1, height, width, 1))
            image_batch = np.concatenate((image_batch,np.zeros([batch_size, depth-1, height, width, 1])),axis=1)
    
            if N2N_flag or simu_flag:
                gt_batch = gt_batch.reshape((batch_size, depth, height, width, 1))
            else:
                gt_batch = gt_batch.reshape((batch_size, depth, height*2, width*2, 1))
        else:
            image_batch = image_batch.reshape((batch_size, height, width, 1, 1))
            image_batch = np.concatenate((image_batch,np.zeros([batch_size, height, width, depth-1, 1])),axis=3)
    
            if N2N_flag or simu_flag:
                gt_batch = gt_batch.reshape((batch_size, height, width, depth, 1))
            else:
                gt_batch = gt_batch.reshape((batch_size, height*2, width*2, depth, 1))
    return image_batch, gt_batch

def data_loader3D_MI(images_path, data_path, gt_path, height, width, depth, 
                  batch_size, depth_first_flag, half_side_flag,
                  norm_flag, interp_flag, insert_slice, zero_insert,
                  down_sample_rate,test_flag):
    batch_images_path = np.random.choice(images_path, size=batch_size)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        img_raw = np.array(imageio.mimread(path)).astype(np.float)
        path_gt = path.replace(data_path, gt_path) #str.replace(old, new)
        gt = np.array(imageio.mimread(path_gt)).astype(np.float)
        
        img_raw = img_raw
        gt = gt
        
        if not depth_first_flag:
            img_raw = np.transpose(img_raw,(1,2,0))
            gt = np.transpose(gt,(1,2,0))
        height_gt = gt.shape[0]
        sr_ratio = height_gt//height
        if norm_flag:
            img_raw = prctile_norm(img_raw)
            gt = prctile_norm(gt)
        else:
            img_raw = img_raw / 65535
            gt = gt / 65535
        
        if depth_first_flag:
            img = np.zeros((depth,height,width))
            for n in range(insert_slice,depth-insert_slice,down_sample_rate):
                img[n,:,:] = img_raw[(n-insert_slice)//down_sample_rate,:,:]
            if insert_slice>0 and not zero_insert:
                for n in range(insert_slice):
                    img[n,:,:] = img[insert_slice,:,:]
                for n in range(depth-insert_slice, depth):
                    img[n,:,:] = img[depth-insert_slice-1,:,:]
            if interp_flag:
                for n in range(insert_slice,depth-insert_slice-down_sample_rate,down_sample_rate):
                    for i in range(down_sample_rate-1):
                        img[n+i+1,:,:] = (img[n,:,:]+img[n+down_sample_rate,:,:])/2

        else:
            img = np.zeros((height,width,depth))
            for n in range(insert_slice,depth-insert_slice,down_sample_rate):
                img[:,:,n] = img_raw[:,:,(n-insert_slice)//down_sample_rate]
            if insert_slice>0 and not zero_insert:
                for n in range(insert_slice):
                    img[:,:,n] = img[:,:,insert_slice]
                for n in range(depth-insert_slice, depth):
                    img[:,:,n] = img[:,:,depth-insert_slice-1]
            if interp_flag:
                for n in range(insert_slice,depth-insert_slice-down_sample_rate,down_sample_rate):
                    for i in range(down_sample_rate-1):
                        img[:,:,n+i+1] = (img[:,:,n]+img[:,:,n+down_sample_rate])/2
        
        if not test_flag:
            gt_raw=gt
            if depth_first_flag:
                gt = np.zeros((depth,height,width))
                for n in range(insert_slice,depth-insert_slice,down_sample_rate):
                    gt[n,:,:] = gt_raw[(n-insert_slice)//down_sample_rate,:,:]
            else:
                gt = np.zeros((height,width,depth))
                for n in range(insert_slice,depth-insert_slice,down_sample_rate):
                    gt[:,:,n] = gt_raw[:,:,(n-insert_slice)//down_sample_rate]
        
        image_batch.append(img)
        gt_batch.append(gt)

    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)
    
    if depth_first_flag:
        image_batch = image_batch.reshape((batch_size, depth, height, width, 1))
        if test_flag:
            gt_batch = gt_batch.reshape((batch_size, depth-2*insert_slice, height*sr_ratio, width*sr_ratio, 1))
        else:
            gt_batch = gt_batch.reshape((batch_size, depth, height*sr_ratio, width*sr_ratio, 1))
    else:
        image_batch = image_batch.reshape((batch_size, height, width, depth, 1))
        if test_flag:
            gt_batch = gt_batch.reshape((batch_size, height*sr_ratio, width*sr_ratio, depth-2*insert_slice, 1))
        else:
            gt_batch = gt_batch.reshape((batch_size, height*sr_ratio, width*sr_ratio, depth, 1))

    return image_batch, gt_batch

def data_loader3D_mitosis(images_path, data_path, gt_path, height, width, depth, 
                  batch_size, img_version, norm_flag=0):
    batch_images_path = np.random.choice(images_path, size=batch_size)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        img = np.array(imageio.mimread(path)).astype(np.float)
        img = np.transpose(img,(1,2,0))
        
        gt = img[:,:,depth*2:depth*4:2]
        if img_version==1:
            img = img[:,:,0:-1:2]
        elif img_version==2:
            img = img[:,:,2*depth+1:4*depth:2]
        
        if norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / 65535
            gt = gt / 65535
        image_batch.append(img)
        gt_batch.append(gt)

    # for item in image_batch:
    #     print(item.shape)
    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)
    
    image_batch = image_batch.reshape((batch_size, height, width, depth, -1))
    gt_batch = gt_batch.reshape((batch_size, height, width, depth, 1))
    
    return image_batch, gt_batch

def data_loader3D_superfast(images_path, data_path, gt_path, height, width, depth, 
                  batch_size, time_points, norm_flag=0):
    batch_images_path = np.random.choice(images_path, size=batch_size)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        img = np.array(imageio.mimread(path)).astype(np.float)
        img = np.transpose(img,(1,2,0))
        img.reshape([height,width,depth,-1])
        img = img[:,:,:,0:time_points]
        
        gt = img[:,:,0:time_points//2]
        img = img[:,:,time_points//2+1:-1]
        
        if norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / 65535
            gt = gt / 65535
        image_batch.append(img)
        gt_batch.append(gt)

    # for item in image_batch:
    #     print(item.shape)
    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)
    
    image_batch = image_batch.reshape((batch_size, height, width, depth, -1))
    gt_batch = gt_batch.reshape((batch_size, height, width, depth, -1))
    
    return image_batch, gt_batch

def DataLoader_3D_tif(images_path, data_path, gt_path, 
                  batch_size, norm_flag):
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        path_gt = path.replace(data_path, gt_path) #str.replace(old, new)
        while not os.path.exists(path_gt):
            path= np.random.choice(images_path, size=1, replace=False)
        img = np.array(imageio.mimread(path)).astype(np.float32)
        gt = np.array(imageio.mimread(path_gt)).astype(np.float32)
        
        if norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / 65535
            gt = gt / 65535
        
        image_batch.append(img)
        gt_batch.append(gt)

    image_batch = np.array(image_batch).astype(np.float32)
    gt_batch = np.array(gt_batch).astype(np.float32)
    
    return image_batch, gt_batch

def DataLoader_3D_tif_crop(images_path, data_path, gt_path, 
                           input_x, input_y, input_z,
                  batch_size, norm_flag):
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        path_gt = path.replace(data_path, gt_path) #str.replace(old, new)
        while not os.path.exists(path_gt):
            path= np.random.choice(images_path, size=1, replace=False)
        img = np.array(imageio.mimread(path)).astype(np.float32)
        gt = np.array(imageio.mimread(path_gt)).astype(np.float32)
        if input_z<img.shape[0]:
            z = np.random.randint(0,img.shape[0]-input_z)
            img = img[z:z+input_z,...]
            gt = gt[z:z+input_z,...]
        if input_y<img.shape[1]:
            y = np.random.randint(0,img.shape[1]-input_y)
            img = img[:,y:y+input_y,:]
            gt = gt[:,y:y+input_y,:] 
        if input_x<img.shape[2]:
            x = np.random.randint(0,img.shape[2]-input_x)
            img = img[...,x:x+input_x]
            gt = gt[...,x:x+input_x]
        
        if norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / 65535
            gt = gt / 65535
        
        image_batch.append(img)
        gt_batch.append(gt)

    image_batch = np.array(image_batch).astype(np.float32)
    gt_batch = np.array(gt_batch).astype(np.float32)
    
    return image_batch, gt_batch

def DataLoader_3D_tif_ch1(images_path, data_path, gt_path, 
                  batch_size, norm_flag):
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        path_gt = path.replace(data_path, gt_path) #str.replace(old, new)
        while not os.path.exists(path_gt):
            path= np.random.choice(images_path, size=1, replace=False)
        img = np.array(imageio.mimread(path)).astype(np.float32)
        img = img[16*3:16*4,:,:]
        gt = np.array(imageio.mimread(path_gt)).astype(np.float32)
        gt = gt[16*3:16*4,:,:]
        
        if norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / 65535
            gt = gt / 65535
        
        image_batch.append(img)
        gt_batch.append(gt)

    image_batch = np.array(image_batch).astype(np.float32)
    gt_batch = np.array(gt_batch).astype(np.float32)
    
    return image_batch, gt_batch

def DataLoader_3D_tif_nogt(images_path, data_path, 
                  batch_size, norm_flag):
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    image_batch = []

    for path in batch_images_path:
        
        img = np.array(imageio.mimread(path)).astype(np.float32)
        
        if norm_flag:
            img = prctile_norm(img)
        else:
            img = img / 65535
        
        image_batch.append(img)

    image_batch = np.array(image_batch).astype(np.float32)
    
    return image_batch

def augment_img(img,mode):
    if mode==1:
        img = np.flipud(np.rot90(img))
    elif mode==2:
        img = np.flipud(img)
    elif mode==3:
        img = np.rot90(img,k=3)
    elif mode==4:
        img = np.flipud(np.rot90(img,k=2))
    elif mode==5:
        img = np.rot90(img)
    elif mode==6:
        img = np.rot90(img,k=2)
    elif mode==7:
        img = np.flipud(np.rot90(img,k=3))
        
    return img

def DataLoader_3D_npy(images_path, data_path, gt_path, 
                  batch_size, norm_flag):
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        path_gt = path.replace(data_path, gt_path) #str.replace(old, new)
        while not os.path.exists(path_gt):
            path= np.random.choice(images_path, size=1, replace=False)
        img = np.load(path)
        gt = np.load(path_gt)
        
        #augment
        mode = np.random.randint(1,7)
        img = augment_img(img,mode)
        gt = augment_img(gt,mode)
        
        #crop
        x_ind = math.floor((1004-128)*np.random.rand(1))
        y_ind = math.floor((1004-128)*np.random.rand(1))
        img = img[x_ind:x_ind+128,y_ind:y_ind+128]
        gt = gt[x_ind:x_ind+128,y_ind:y_ind+128]
        
        if norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / np.max(img)
            gt = gt / np.max(gt)
        
        image_batch.append(img)
        gt_batch.append(gt)

    image_batch = np.array(image_batch).astype(np.float32)
    gt_batch = np.array(gt_batch).astype(np.float32)
    
    return image_batch, gt_batch