import numpy as np
import csv
import os
import tifffile

if __name__ == '__main__':
    csv_path = '/mnt/Public/chy/localize/train_data/20230411_TFM_300nm_recrop/adjust/'
    save_path = '../data/train_bead/TrainData_488_1.41rNA_300nmbeads/20230411_TFM_300nm/nc_position_recrop/'

    upfactor = 1

    nz = 8
    nx = ny = 1152 * upfactor

    for f in os.listdir(csv_path):
        if f.endswith(".csv"):
            img = np.zeros((nz, nx, ny))

            fin = csv.reader(open(os.path.join(csv_path, f)))
            for lines in fin:
                if lines[0] == 'frame':
                    continue
                else:
                    # img[int(lines[0]) - 1, int(np.round((float(lines[2]) - 0.5) * upfactor)), int(
                    #     np.round((float(lines[1]) - 0.5) * upfactor))] = 1
                    img[int(lines[0]) - 1, int(float(lines[2]) * upfactor), int(float(lines[1]) * upfactor)] = 1

                    # img[int(lines[0]) - 1, int(float(lines[2]) * upfactor), int(float(lines[1]) * upfactor)] = 1

            tifffile.imsave(os.path.join(save_path, f.split('_set')[0] +'.tif'), img)
            # tifffile.imsave(os.path.join(save_path, f.split('_set')[0] + '_up' + str(upfactor) + '_floor.tif'), img)
